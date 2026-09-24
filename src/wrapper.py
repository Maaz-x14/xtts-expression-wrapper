"""
ExpressionWrapper — XTTS v2, Coqui TTS backend with latent caching.

One instance = one model (base|finetuned) + one language (english|urdu), fixed
at construction — matches how you're running the dual-model comparison.
Testing phase only — Auralis not used. Base loads via Coqui's TTS.api
(auto-download from Coqui/HF); fine-tuned loads from a local checkpoint dir.
Both converge to a raw Xtts instance on self.model, so every downstream call
(get_conditioning_latents, inference) is identical regardless of which loaded.

Latent caching: get_conditioning_latents() called once per emotion at startup for
the selected language. (gpt_cond_latent, speaker_embedding) cached in self._cache.

Text chunking: manual, sentence-boundary split before inference() — XTTS GPT
context is ~250 chars.
"""

import os
import re
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, basic_cleaners

from src.tag_parser import parse_input
from src.reference_map import get_reference_path, verify_all_clips, ALL_BASE_EMOTIONS

# ---------------------------------------------------------------------------
# Monkeypatch: XTTS's tokenizer (TTS/tts/layers/xtts/tokenizer.py) hardcodes a
# language allowlist in preprocess_text() that raises NotImplementedError for
# any language outside {ar, cs, de, en, es, fr, hu, it, nl, pl, pt, ru, tr,
# zh, ko, ja, hi}. Urdu ('ur') isn't in that list, even though this fine-tuned
# checkpoint was explicitly trained on Urdu. Patched here — not in
# site-packages — so it survives reinstalls/upgrades.
#
# basic_cleaners (lowercase + whitespace collapse, no transliteration) mirrors
# exactly what Coqui's own maintainers do for 'hi' (Hindi) — same "not yet
# implemented, use the neutral fallback" pattern. multilingual_cleaners was
# deliberately avoided: it calls expand_numbers_multilingual() -> num2words(),
# and num2words' Urdu support is unverified — not worth risking a second,
# harder-to-diagnose failure mid-pipeline.
# ---------------------------------------------------------------------------
_original_preprocess_text = VoiceBpeTokenizer.preprocess_text


def _patched_preprocess_text(self, txt, lang):
    if lang == "ur":
        return basic_cleaners(txt)
    return _original_preprocess_text(self, txt, lang)


VoiceBpeTokenizer.preprocess_text = _patched_preprocess_text

FINETUNED_MODEL_DIR = Path(os.environ.get("XTTS_MODEL_DIR", Path(__file__).parent.parent / "Agri-TTS"))
OUTPUT_DIR_ROOT      = Path(__file__).parent.parent / "output"
SAMPLE_RATE          = 24000

MODEL_KEYS     = {"base", "finetuned"}
LANGUAGE_CODES = {"english": "en", "urdu": "ur"}
CHUNK_CHAR_LIMIT = 200


class ExpressionWrapper:
    def __init__(
        self,
        model: str = "finetuned",
        language: str = "urdu",
        temperature: float = 0.75,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 5.0,
        length_penalty: float = 1.0,
    ):
        if model not in MODEL_KEYS:
            raise ValueError(f"Unsupported model '{model}'. Use: {sorted(MODEL_KEYS)}")
        if language not in LANGUAGE_CODES:
            raise ValueError(f"Unsupported language '{language}'. Use: {sorted(LANGUAGE_CODES)}")

        self.model_key = model
        self.model_dir = FINETUNED_MODEL_DIR if model == "finetuned" else None
        self.language  = language
        self.lang_code = LANGUAGE_CODES[language]

        self.temperature        = temperature
        self.top_p              = top_p
        self.top_k              = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty     = length_penalty

        self._validate_checkpoint()
        self._check_clips()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WRAPPER] Model    : {self.model_key}")
        print(f"[WRAPPER] Language : {self.language} ({self.lang_code})")
        print(f"[WRAPPER] Device   : {self.device}")

        if self.model_key == "base":
            print("[WRAPPER] Loading stock XTTS v2 (auto-download on first run)...")
            from TTS.api import TTS as CoquiTTS
            tts_api = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            self.model = tts_api.synthesizer.tts_model
            self.model.to(self.device)
        else:
            print(f"[WRAPPER] Loading fine-tuned checkpoint ({self.model_dir})...")
            config = XttsConfig()
            config.load_json(str(self.model_dir / "config.json"))
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(
                config,
                checkpoint_dir=str(self.model_dir),
                checkpoint_path=str(self.model_dir / "model.pth"),
                vocab_path=str(self.model_dir / "vocab.json"),
                eval=True,
            )
            self.model.to(self.device)

        self.output_dir = OUTPUT_DIR_ROOT / self.language
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("[WRAPPER] Building latent cache...")
        self._cache: dict[str, tuple] = {}
        self._build_cache()
        print(f"[WRAPPER] Cache ready — {len(self._cache)} emotions.\n")

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _validate_checkpoint(self):
        """Only applies to the local fine-tuned checkpoint — base auto-downloads."""
        if self.model_key != "finetuned":
            return
        required = ["config.json", "model.pth", "vocab.json"]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing in {self.model_dir}: {missing}\n"
                f"Set XTTS_MODEL_DIR or place checkpoint in {FINETUNED_MODEL_DIR}/"
            )

    def _check_clips(self):
        missing = [e for e, ok in verify_all_clips(self.language).items() if not ok]
        if missing:
            print(f"[WRAPPER] WARNING — no ref.wav for ({self.language}): {missing}")
            print(f"[WRAPPER] Run: python scripts/build_reference_clips.py --language {self.language}")

    # ------------------------------------------------------------------
    # Latent cache
    # ------------------------------------------------------------------

    def _build_cache(self):
        for emotion in sorted(ALL_BASE_EMOTIONS):
            try:
                ref_path = get_reference_path(emotion, self.language)
            except FileNotFoundError:
                continue

            t0 = time.time()
            try:
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[str(ref_path)],
                    gpt_cond_len=self.model.config.gpt_cond_len,
                    gpt_cond_chunk_len=6,      # was: self.model.config.gpt_cond_chunk_len (4)
                    max_ref_length=10,         # was: self.model.config.max_ref_len (30)
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
        if len(text) <= CHUNK_CHAR_LIMIT:
            return [text]
        parts = re.split(r'(?<=[۔؟!.?])\s+', text)
        chunks, current = [], ""
        for part in parts:
            if len(current) + len(part) + 1 <= CHUNK_CHAR_LIMIT:
                current = (current + " " + part).strip()
            else:
                if current:
                    chunks.append(current)
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
        chunks = self._chunk_text(text)
        parts = []
        for chunk in chunks:
            out = self.model.inference(
                text=chunk,
                language=self.lang_code,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                enable_text_splitting=False,
            )
            audio = torch.tensor(out["wav"]).numpy().astype(np.float32)
            parts.append(audio)
        return np.concatenate(parts)

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        if abs(speed - 1.0) < 1e-6:
            return audio
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)

    def synthesize(self, tagged_input: str, output_filename: str = "output.wav") -> Path:
        emotion, segments = parse_input(tagged_input)
        output_path = self.output_dir / output_filename
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