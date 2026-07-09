from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

# Maps each supported emotion to its reference clip filename.
# whisper uses RAVDESS 'calm' emotion as proxy — closest available.
EMOTION_TO_CLIP: dict[str, Path] = {
    "neutral": REFERENCE_DIR / "neutral.wav",
    "happy":   REFERENCE_DIR / "happy.wav",
    "sad":     REFERENCE_DIR / "sad.wav",
    "angry":   REFERENCE_DIR / "angry.wav",
    "whisper": REFERENCE_DIR / "whisper.wav",
}

def get_reference_path(emotion: str) -> Path:
    """
    Return the Path to the reference wav for the given emotion.
    Falls back to neutral if the clip doesn't exist on disk.
    """
    path = EMOTION_TO_CLIP.get(emotion)

    if path is None:
        print(f"[REFMAP] Emotion '{emotion}' not in map. Falling back to neutral.")
        return EMOTION_TO_CLIP["neutral"]

    if not path.exists():
        print(f"[REFMAP] Clip not found: {path}. Falling back to neutral.")
        fallback = EMOTION_TO_CLIP["neutral"]
        if not fallback.exists():
            raise FileNotFoundError(
                f"Neutral fallback clip not found at {fallback}. "
                f"Run: python scripts/download_ravdess.py"
            )
        return fallback

    return path


def verify_all_clips() -> dict[str, bool]:
    """
    Check which emotion clips are present on disk.
    Returns a dict of emotion → exists.
    """
    return {emotion: path.exists() for emotion, path in EMOTION_TO_CLIP.items()}
