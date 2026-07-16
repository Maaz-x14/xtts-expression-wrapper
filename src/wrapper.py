import os
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path

# Must be set before TTS import
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

from src.tag_parser import parse_input
from src.reference_map import get_reference_path, verify_all_clips

OUTPUT_DIR  = Path(__file__).parent.parent / "output"
MODEL_NAME  = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000  # XTTS-v2 native output rate


class ExpressionWrapper:
    """
    Level 1 + Stage 2 wrapper around XTTS-v2.

    Handles:
      - Emotion routing via reference audio swap
      - Inline pause / break / silence via silence insertion
      - Inline fast / slow via librosa time-stretch
    """

    def __init__(self):
        self._check_clips()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WRAPPER] Device    : {device}")
        print(f"[WRAPPER] Loading XTTS-v2 model...")
        self.tts = TTS(MODEL_NAME).to(device)
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("[WRAPPER] Ready.")

    def _check_clips(self):
        missing = [e for e, ok in verify_all_clips().items() if not ok]
        if missing:
            print(f"[WRAPPER] WARNING — missing reference clips: {missing}")
            print(f"[WRAPPER] Run: python scripts/download_ravdess.py")

    def _synth_text(self, text: str, reference_path: Path, tmp_path: Path) -> np.ndarray:
        """Synthesize one text segment, return float32 numpy array."""
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(reference_path),
            language="en",
            file_path=str(tmp_path),
        )
        audio, _ = sf.read(str(tmp_path))
        return audio.astype(np.float32)

    def _make_silence(self, duration_ms: int) -> np.ndarray:
        """Generate silence as zero-filled float32 array."""
        n_samples = int(SAMPLE_RATE * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.float32)

    def _apply_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Time-stretch audio by speed factor.
          speed > 1.0 → faster (shorter)
          speed < 1.0 → slower (longer)
        """
        if abs(speed - 1.0) < 1e-6:
            return audio
        return librosa.effects.time_stretch(audio, rate=speed)

    def synthesize(self, tagged_input: str, output_filename: str = "output.wav") -> Path:
        """
        Parse tagged input and synthesize final audio.

        Args:
            tagged_input   : e.g. "<happy>Hello <pause=300ms> how are you?</happy>"
            output_filename: wav filename inside output/

        Returns:
            Path to generated wav file.
        """
        emotion, segments = parse_input(tagged_input)
        reference_path    = get_reference_path(emotion)
        output_path       = OUTPUT_DIR / output_filename
        tmp_path          = OUTPUT_DIR / "_tmp_seg.wav"

        print(f"[WRAPPER] Emotion   : {emotion}")
        print(f"[WRAPPER] Reference : {reference_path.name}")
        print(f"[WRAPPER] Segments  : {len(segments)}")
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
                print(f"        Synthesizing... (CPU: 30–90s per segment)")
                audio = self._synth_text(text, reference_path, tmp_path)
                audio = self._apply_speed(audio, speed)
                audio_parts.append(audio)

        if tmp_path.exists():
            tmp_path.unlink()

        final = np.concatenate(audio_parts)
        sf.write(str(output_path), final, SAMPLE_RATE)

        print(f"\n[WRAPPER] Done → {output_path}")
        return output_path
