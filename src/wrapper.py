"""
ExpressionWrapper — Urdu XTTS v2 fine-tune, Coqui TTS backend with latent caching.

Auralis dropped — incompatible with Coqui trainer checkpoint format (needs safetensors
+ HF-style config.json). Coqui TTS 0.22.0 loads the checkpoint natively.

Latent caching: tts_model.get_conditioning_latents() is called once per emotion at
startup. (gpt_cond_latent, speaker_embedding) tensors are stored in self._cache.
Synthesis calls inference() directly with precomputed tensors — zero re-encoding
per request.

Text chunking: handled manually — split on sentence boundaries before passing to
inference(), since XTTS GPT context is ~250 chars.
"""

import os
import re
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from src.tag_parser import parse_input
from src.reference_map import get_reference_path, verify_all_clips, ALL_BASE_EMOTIONS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR   = Path(os.environ.get("XTTS_MODEL_DIR", "Agri-TTS"))
OUTPUT_DIR  = Path(__file__).parent.parent / "output"
SAMPLE_RATE = 24000
LANGUAGE    = "ur"

# XTTS GPT context limit — split text segments longer than this
CHUNK_CHAR_LIMIT = 200


class ExpressionWrapper:
    """
    Coqui XTTS v2 fine-tune wrapper with manual latent caching.

    Startup: loads model, computes (gpt_cond_latent, speaker_embedding) for
             every emotion that has a ref.wav, stores in self._cache.
    Synthesis: calls tts_model.inference() directly with cached tensors.
               No reference audio re-encoding at request time.
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        temperature: float = 0.75,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 5.0,
        length_penalty: float = 1.0,
    ):
        self.model_dir          = Path(model_dir)
        self.temperature        = temperature
        self.top_p              = top_p
        self.top_k              = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty     = length_penalty

        self._validate_checkpoint()
        self._check_clips()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WRAPPER] Device   : {self.device}")
        print(f"[WRAPPER] Model    : {self.model_dir}")
        print(f"[WRAPPER] Loading XTTS v2 fine-tune...")

        config = XttsConfig()
        config.load_json(str(self.model_dir / "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config,
            checkpoint_dir=str(self.model_dir),   # <-- add this line
            checkpoint_path=str(self.model_dir / "model.pth"),
            vocab_path=str(self.model_dir / "vocab.json"),
            eval=True,
        )
        self.model.to(self.device)

        OUTPUT_DIR.mkdir(exist_ok=True)

        print("[WRAPPER] Building latent cache...")
        self._cache: dict[str, tuple] = {}
        self._build_cache()
        print(f"[WRAPPER] Cache ready — {len(self._cache)} emotions.\n")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _validate_checkpoint(self):
        required = ["config.json", "model.pth", "vocab.json"]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing in {self.model_dir}: {missing}\n"
                f"Set XTTS_MODEL_DIR env var to point at Agri-TTS/"
            )

    def _check_clips(self):
        missing = [e for e, ok in verify_all_clips().items() if not ok]
        if missing:
            print(f"[WRAPPER] WARNING — no ref.wav for: {missing}")
            print(f"[WRAPPER] Run: python scripts/select_clips.py --semour-dir SEMOUR+_data --build --actor 5")

    # ------------------------------------------------------------------
    # Latent cache
    # ------------------------------------------------------------------

    def _build_cache(self):
        """
        Compute (gpt_cond_latent, speaker_embedding) for each emotion's ref.wav.
        get_conditioning_latents() concatenates audio internally for gpt_cond_latent
        and averages speaker_embedding — same mechanism as multi-clip XTTS conditioning.
        """
        for emotion in sorted(ALL_BASE_EMOTIONS):
            try:
                ref_path = get_reference_path(emotion)
            except FileNotFoundError:
                continue

            t0 = time.time()
            try:
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[str(ref_path)],
                    gpt_cond_len=self.model.config.gpt_cond_len,
                    gpt_cond_chunk_len=self.model.config.gpt_cond_chunk_len,
                    max_ref_length=self.model.config.max_ref_len,
                    sound_norm_refs=self.model.config.sound_norm_refs,
                )
                self._cache[emotion] = (gpt_cond_latent, speaker_embedding)
                print(f"  [{emotion:10s}] {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  [{emotion:10s}] FAILED: {e}")

    def _get_latents(self, emotion: str) -> tuple:
        if emotion in self._cache:
            return self._cache[emotion]
        print(f"[WRAPPER] No latents for '{emotion}' — falling back to neutral")
        if "neutral" in self._cache:
            return self._cache["neutral"]
        raise RuntimeError("Latent cache empty. Check reference_clips/.")

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text at sentence boundaries to stay within XTTS's GPT context limit.
        Tries punctuation splits first, falls back to hard char-limit splits.
        """
        if len(text) <= CHUNK_CHAR_LIMIT:
            return [text]

        # Split on Urdu/Arabic sentence-ending punctuation + common Latin stops
        parts = re.split(r'(?<=[۔؟!.?])\s+', text)
        chunks, current = [], ""
        for part in parts:
            if len(current) + len(part) + 1 <= CHUNK_CHAR_LIMIT:
                current = (current + " " + part).strip()
            else:
                if current:
                    chunks.append(current)
                # If single part is still too long, hard-split
                while len(part) > CHUNK_CHAR_LIMIT:
                    chunks.append(part[:CHUNK_CHAR_LIMIT])
                    part = part[CHUNK_CHAR_LIMIT:]
                current = part
        if current:
            chunks.append(current)
        return chunks

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _make_silence(self, duration_ms: int) -> np.ndarray:
        return np.zeros(int(SAMPLE_RATE * duration_ms / 1000), dtype=np.float32)

    def _synth_text(self, text: str, gpt_cond_latent, speaker_embedding) -> np.ndarray:
        """Synthesize one text chunk using precomputed latents."""
        chunks = self._chunk_text(text)
        parts = []
        for chunk in chunks:
            out = self.model.inference(
                text=chunk,
                language=LANGUAGE,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                enable_text_splitting=False,  # we handle splitting ourselves
            )
            audio = torch.tensor(out["wav"]).numpy().astype(np.float32)
            parts.append(audio)
        return np.concatenate(parts)

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if abs(speed - 1.0) < 1e-6:
            return audio
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)

    def synthesize(
        self,
        tagged_input: str,
        output_filename: str = "output.wav",
    ) -> Path:
        """
        Parse tagged input and synthesize to wav.

        Args:
            tagged_input   : e.g. "<happy>السلام علیکم <pause=300ms> آپ کیسے ہیں؟</happy>"
            output_filename: filename inside output/

        Returns:
            Path to generated wav.
        """
        emotion, segments = parse_input(tagged_input)
        output_path = OUTPUT_DIR / output_filename
        gpt_cond_latent, speaker_embedding = self._get_latents(emotion)

        print(f"[WRAPPER] Emotion  : {emotion}")
        print(f"[WRAPPER] Segments : {len(segments)}")

        parts = []
        for i, seg in enumerate(segments):
            if seg["type"] == "pause":
                print(f"  [{i+1}] pause {seg['duration_ms']}ms")
                parts.append(self._make_silence(seg["duration_ms"]))

            elif seg["type"] == "text":
                speed = seg["speed"]
                print(f"  [{i+1}] text  speed={speed}  \"{seg['content']}\"")
                audio = self._synth_text(seg["content"], gpt_cond_latent, speaker_embedding)
                audio = self._apply_speed(audio, speed)
                parts.append(audio)

        final = np.concatenate(parts)
        sf.write(str(output_path), final, SAMPLE_RATE)
        print(f"\n[WRAPPER] → {output_path}")
        return output_path