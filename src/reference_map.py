"""
reference_map.py — Emotion → reference audio clip routing.

Clip structure on disk (produced by scripts/select_clips.py --build):
  reference_clips/
    neutral.wav          ← flat fallback (copy of neutral/ref.wav)
    angry/
      ref.wav            ← concatenated composite of 4 longest clips (~5–6s)
    happy/
      ref.wav
    sad/
      ref.wav
    ... (one ref.wav per emotion)

Source: SEMOUR+ clips, pre-concatenated at prep time.
Concatenation happens ONCE in select_clips.py, not at inference time.
wrapper.py loads ref.wav → computes latents → caches. Done.
"""

from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

# All supported emotions (internal keys)
ALL_BASE_EMOTIONS = {
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
}

# Sub-emotion aliases → base emotion
EMOTION_ALIASES: dict[str, str] = {
    "joy": "happy", "excitement": "happy", "contentment": "happy",
    "grief": "sad", "loneliness": "sad", "disappointment": "sad",
    "rage": "angry", "frustration": "angry", "irritation": "angry",
    "anxiety": "fearful", "nervousness": "fearful", "panic": "fearful",
    "contempt": "disgust", "revulsion": "disgust", "disdain": "disgust",
    "shock": "surprised", "amazement": "surprised", "disbelief": "surprised",
    "serenity": "calm", "relaxation": "calm", "boredom": "calm",
    "whisper": "neutral",  # whisper falls back to neutral ref
}


def _resolve(emotion: str) -> str:
    return EMOTION_ALIASES.get(emotion.lower(), emotion.lower())


def get_reference_path(emotion: str) -> Path:
    """
    Return Path to the reference wav for the given emotion.
    Falls back to neutral if not found.
    """
    emotion = _resolve(emotion)

    path = REFERENCE_DIR / emotion / "ref.wav"
    if path.exists():
        return path

    # flat neutral.wav fallback
    flat = REFERENCE_DIR / "neutral.wav"
    if flat.exists():
        print(f"[REFMAP] '{emotion}/ref.wav' not found — falling back to neutral.wav")
        return flat

    raise FileNotFoundError(
        f"No reference clip found for '{emotion}' and neutral fallback missing.\n"
        f"Run: python scripts/select_clips.py --semour-dir SEMOUR+_data --build --actor <ID>"
    )


def list_clips(emotion: str) -> list[Path]:
    """
    Return available reference paths for an emotion.
    With the concatenated-composite structure, this is always a 1-item list.
    Kept for API compatibility with wrapper.py.
    """
    path = get_reference_path(emotion)
    return [path]


def verify_all_clips() -> dict[str, bool]:
    """Check which base emotions have a ref.wav on disk."""
    return {
        emotion: (REFERENCE_DIR / emotion / "ref.wav").exists()
        for emotion in sorted(ALL_BASE_EMOTIONS)
    }
