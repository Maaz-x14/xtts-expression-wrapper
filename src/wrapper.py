import os
import torch
from pathlib import Path

# Must be set before TTS import — accepts Coqui XTTS-v2 license automatically.
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS

from src.tag_parser import parse_input
from src.reference_map import get_reference_path, verify_all_clips

OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class ExpressionWrapper:
    """
    Level 1 wrapper around XTTS-v2.
    Parses expression tags and routes to the appropriate reference audio clip.
    Does not modify model weights or internals.
    """

    def __init__(self):
        self._check_clips()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WRAPPER] Device: {device}")
        print(f"[WRAPPER] Loading XTTS-v2 model (first run downloads ~1.8GB)...")
        self.tts = TTS(MODEL_NAME).to(device)
        OUTPUT_DIR.mkdir(exist_ok=True)
        print("[WRAPPER] Ready.")

    def _check_clips(self):
        status = verify_all_clips()
        missing = [e for e, exists in status.items() if not exists]
        if missing:
            print(f"[WRAPPER] WARNING — missing reference clips: {missing}")
            print(f"[WRAPPER] Run: python scripts/download_ravdess.py")

    def synthesize(self, tagged_input: str, output_filename: str = "output.wav") -> Path:
        """
        Parse tagged input and synthesize audio.

        Args:
            tagged_input: string like "<happy>Hello world</happy>"
            output_filename: wav filename inside output/

        Returns:
            Path to generated audio file.
        """
        emotion, text = parse_input(tagged_input)
        reference_path = get_reference_path(emotion)
        output_path = OUTPUT_DIR / output_filename

        print(f"[WRAPPER] Emotion     : {emotion}")
        print(f"[WRAPPER] Text        : {text}")
        print(f"[WRAPPER] Reference   : {reference_path.name}")
        print(f"[WRAPPER] Output      : {output_path}")
        print(f"[WRAPPER] Synthesizing... (CPU: expect 30–90s per sentence)")

        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(reference_path),
            language="en",
            file_path=str(output_path),
        )

        print(f"[WRAPPER] Done → {output_path}")
        return output_path
