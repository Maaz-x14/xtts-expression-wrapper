"""
ExpressionWrapper — Urdu XTTS v2 fine-tune backend (Auralis serving layer).

Key upgrades over the original TTS.api version:
  - Loads a LOCAL fine-tuned checkpoint (model.pth + config.json + vocab.json)
  - Latent cache: gpt_cond_latent + speaker_embedding precomputed at startup,
    never recomputed per request
  - Multi-clip conditioning: pass all clips for an emotion or a single best clip
    (controlled by USE_MULTI_CLIP flag or per-call override)
  - Text chunking: Auralis handles XTTS's 250-char GPT context limit internally
  - Language: "ur" (Urdu fine-tune)

tag_parser.py and reference_map.py are UNCHANGED — this is a drop-in backend swap.
"""

import os
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from typing import Optional

os.environ["COQUI_TOS_AGREED"] = "1"

# ---------------------------------------------------------------------------
# Auralis imports — pip install auralis
# ---------------------------------------------------------------------------
from auralis import TTS as AuralisTTS, TTSRequest

from src.tag_parser import parse_input
from src.reference_map import get_reference_path, list_clips, verify_all_clips

# ---------------------------------------------------------------------------
# Paths — update MODEL_DIR to wherever Agri-TTS lives on your machine
# ---------------------------------------------------------------------------
MODEL_DIR   = Path(os.environ.get("XTTS_MODEL_DIR", "Agri-TTS"))
OUTPUT_DIR  = Path(__file__).parent.parent / "output"
SAMPLE_RATE = 24000  # XTTS v2 native output rate

# ---------------------------------------------------------------------------
# Defaults — flip USE_MULTI_CLIP to False to use single best clip per emotion
# ---------------------------------------------------------------------------
USE_MULTI_CLIP   = True   # True → all available clips; False → DEFAULT_CLIP_ID
LANGUAGE         = "ur"
ENABLE_CHUNKING  = True   # handles XTTS 250-char GPT context limit


class ExpressionWrapper:
    """
    Auralis-backed expressive TTS wrapper for the Urdu XTTS v2 fine-tune.

    Latent cache is built at __init__ time for every emotion × clip combination
    present on disk. Synthesis calls are pure inference — no reference audio
    re-encoding at request time.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        use_multi_clip: bool = USE_MULTI_CLIP,
        enable_chunking: bool = ENABLE_CHUNKING,
    ):
        self.model_dir      = Path(model_dir)
        self.use_multi_clip = use_multi_clip
        self.enable_chunking = enable_chunking

        self._validate_checkpoint()
        self._check_clips()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WRAPPER] Device      : {device}")
        print(f"[WRAPPER] Model dir   : {self.model_dir}")
        print(f"[WRAPPER] Multi-clip  : {self.use_multi_clip}")
        print(f"[WRAPPER] Chunking    : {self.enable_chunking}")
        print(f"[WRAPPER] Loading fine-tuned XTTS v2...")

        self.tts = AuralisTTS().from_pretrained(
            str(self.model_dir),           # path to config.json + model.pth
            gpt_model=str(self.model_dir / "model.pth"),
            tokenizer_model=str(self.model_dir / "vocab.json"),
        )

        OUTPUT_DIR.mkdir(exist_ok=True)

        # Build latent cache at startup
        print("[WRAPPER] Precomputing latent cache...")
        self._latent_cache: dict[str, tuple] = {}
        self._build_latent_cache()
        print(f"[WRAPPER] Cache ready — {len(self._latent_cache)} entries.")
        print("[WRAPPER] Ready.\n")

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def _validate_checkpoint(self):
        required = ["config.json", "model.pth", "vocab.json"]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"[WRAPPER] Missing checkpoint files in {self.model_dir}: {missing}\n"
                f"Set XTTS_MODEL_DIR env var or pass model_dir= to point at Agri-TTS/"
            )

    def _check_clips(self):
        missing = [e for e, ok in verify_all_clips().items() if not ok]
        if missing:
            print(f"[WRAPPER] WARNING — missing reference clips for: {missing}")
            print(f"[WRAPPER] Add Urdu clips from SEMOUR+ to reference_clips/<emotion>/")

    # ------------------------------------------------------------------
    # Latent cache
    # ------------------------------------------------------------------

    def _cache_key(self, emotion: str, multi: bool) -> str:
        return f"{emotion}::{'multi' if multi else 'single'}"

    def _build_latent_cache(self):
        """
        Precompute (gpt_cond_latent, speaker_embedding) for every emotion.

        Multi-clip path  : all available wavs → Auralis computes one fused latent
                           (internally: speaker_embedding averaged, gpt_cond_latent
                           from concatenated audio — matches raw XTTS get_conditioning_latents)
        Single-clip path : DEFAULT_CLIP_ID wav only
        """
        from src.reference_map import ALL_BASE_EMOTIONS, list_clips, get_reference_path

        for emotion in sorted(ALL_BASE_EMOTIONS):
            # --- multi-clip entry ---
            if self.use_multi_clip:
                clips = list_clips(emotion)
                if clips:
                    key = self._cache_key(emotion, multi=True)
                    self._latent_cache[key] = self._compute_latents(clips, emotion, "multi")

            # --- single-clip entry (always cache — needed for fallback) ---
            path = get_reference_path(emotion)  # returns DEFAULT_CLIP_ID
            if path.exists():
                key = self._cache_key(emotion, multi=False)
                self._latent_cache[key] = self._compute_latents([path], emotion, "single")

    def _compute_latents(self, clip_paths: list[Path], emotion: str, mode: str):
        """Compute and return (gpt_cond_latent, speaker_embedding) for a clip set."""
        t0 = time.time()
        try:
            latents = self.tts.get_conditioning_latents(
                audio_path=[str(p) for p in clip_paths]
            )
            elapsed = time.time() - t0
            print(f"  [{emotion}:{mode}] {len(clip_paths)} clip(s) → {elapsed:.1f}s")
            return latents
        except Exception as e:
            print(f"  [{emotion}:{mode}] FAILED: {e} — skipping.")
            return None

    def _get_latents(self, emotion: str):
        """
        Return cached latents for emotion.
        Prefers multi-clip if USE_MULTI_CLIP and available, falls back gracefully.
        """
        if self.use_multi_clip:
            key = self._cache_key(emotion, multi=True)
            latents = self._latent_cache.get(key)
            if latents is not None:
                return latents

        key = self._cache_key(emotion, multi=False)
        latents = self._latent_cache.get(key)
        if latents is not None:
            return latents

        # Last resort: neutral
        print(f"[WRAPPER] No cached latents for '{emotion}'. Falling back to neutral.")
        return self._latent_cache.get(self._cache_key("neutral", multi=False))

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synth_text(
        self,
        text: str,
        gpt_cond_latent,
        speaker_embedding,
        tmp_path: Path,
    ) -> np.ndarray:
        """Synthesize one text segment using precomputed latents."""
        request = TTSRequest(
            text=text,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_splitting=self.enable_chunking,
            # Sampling defaults — expose as params if you want to sweep these
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            repetition_penalty=5.0,
        )
        output = self.tts.generate_speech(request)
        # Auralis returns audio as numpy array or writes to file depending on version
        if isinstance(output, np.ndarray):
            audio = output.astype(np.float32)
        else:
            # If it returns a path or bytes, write then read back
            output.save(str(tmp_path))
            audio, _ = sf.read(str(tmp_path))
            audio = audio.astype(np.float32)
        return audio

    def _make_silence(self, duration_ms: int) -> np.ndarray:
        n_samples = int(SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.float32)

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if abs(speed - 1.0) < 1e-6:
            return audio
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)

    def synthesize(
        self,
        tagged_input: str,
        output_filename: str = "output.wav",
        multi_clip_override: Optional[bool] = None,
    ) -> Path:
        """
        Parse tagged input and synthesize final audio.

        Args:
            tagged_input        : e.g. "<happy>سلام، آپ کیسے ہیں؟ <pause=300ms></happy>"
            output_filename     : wav filename inside output/
            multi_clip_override : override instance-level USE_MULTI_CLIP for this call

        Returns:
            Path to generated wav file.
        """
        emotion, segments = parse_input(tagged_input)
        output_path = OUTPUT_DIR / output_filename
        tmp_path    = OUTPUT_DIR / "_tmp_seg.wav"

        # Resolve which latents to use
        latents = self._get_latents(emotion)
        if latents is None:
            raise RuntimeError(
                f"No latents available for '{emotion}' and neutral fallback. "
                f"Check that reference_clips/ has at least neutral.wav."
            )
        gpt_cond_latent, speaker_embedding = latents

        mode = "multi" if (multi_clip_override if multi_clip_override is not None else self.use_multi_clip) else "single"
        print(f"[WRAPPER] Emotion     : {emotion}")
        print(f"[WRAPPER] Clip mode   : {mode}")
        print(f"[WRAPPER] Segments    : {len(segments)}")
        print()

        audio_parts = []

        for i, seg in enumerate(segments):
            if seg["type"] == "pause":
                ms = seg["duration_ms"]
                print(f"  [{i+1}] pause     {ms}ms")
                audio_parts.append(self._make_silence(ms))

            elif seg["type"] == "text":
                text  = seg["content"]
                speed = seg["speed"]
                label = f"speed={speed}" if abs(speed - 1.0) > 1e-6 else "normal"
                print(f"  [{i+1}] text      [{label}]  \"{text}\"")
                print(f"        Synthesizing...")
                audio = self._synth_text(text, gpt_cond_latent, speaker_embedding, tmp_path)
                audio = self._apply_speed(audio, speed)
                audio_parts.append(audio)

        if tmp_path.exists():
            tmp_path.unlink()

        final = np.concatenate(audio_parts)
        sf.write(str(output_path), final, SAMPLE_RATE)

        print(f"\n[WRAPPER] Done → {output_path}")
        return output_path
