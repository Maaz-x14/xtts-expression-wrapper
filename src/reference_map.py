"""
reference_map.py — emotion + language → reference audio clip routing.

Clip structure on disk (produced by scripts/build_reference_clips.py):
  reference_clips/
    english/
      neutral.wav          ← flat fallback (copy of neutral/ref.wav)
      angry/ref.wav        ← concatenated composite
      ...                  ← one ref.wav per emotion
    urdu/
      neutral.wav
      angry/ref.wav
      ...

One actor per language (RAVDESS Actor 01 for English, SEMOUR+ Actor 5 for Urdu) —
no clip_id / multi-actor selection. Concatenation happens ONCE in
build_reference_clips.py, not at inference time.
"""

from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

SUPPORTED_LANGUAGES = {"english", "urdu"}

ALL_BASE_EMOTIONS = {
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
}

EMOTION_ALIASES: dict[str, str] = {
    "joy": "happy", "excitement": "happy", "contentment": "happy",
    "grief": "sad", "loneliness": "sad", "disappointment": "sad",
    "rage": "angry", "frustration": "angry", "irritation": "angry",
    "anxiety": "fearful", "nervousness": "fearful", "panic": "fearful",
    "contempt": "disgust", "revulsion": "disgust", "disdain": "disgust",
    "shock": "surprised", "amazement": "surprised", "disbelief": "surprised",
    "serenity": "calm", "relaxation": "calm", "boredom": "calm",
    "whisper": "neutral",
}


def _resolve(emotion: str) -> str:
    return EMOTION_ALIASES.get(emotion.lower(), emotion.lower())


def _check_language(language: str) -> str:
    language = language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Use: {sorted(SUPPORTED_LANGUAGES)}")
    return language


def get_reference_path(emotion: str, language: str) -> Path:
    """Return Path to the reference wav for the given emotion + language. Falls back to neutral."""
    language = _check_language(language)
    emotion = _resolve(emotion)

    path = REFERENCE_DIR / language / emotion / "ref.wav"
    if path.exists():
        return path

    flat = REFERENCE_DIR / language / "neutral.wav"
    if flat.exists():
        print(f"[REFMAP] '{language}/{emotion}/ref.wav' not found — falling back to neutral.wav")
        return flat

    raise FileNotFoundError(
        f"No reference clip found for '{emotion}' ({language}) and neutral fallback missing.\n"
        f"Run: python scripts/build_reference_clips.py --language {language}"
    )


def list_clips(emotion: str, language: str) -> list[Path]:
    """Kept for API compatibility with wrapper.py — always a 1-item list (composite ref.wav)."""
    return [get_reference_path(emotion, language)]


def verify_all_clips(language: str) -> dict[str, bool]:
    """Check which base emotions have a ref.wav on disk for the given language."""
    language = _check_language(language)
    return {
        emotion: (REFERENCE_DIR / language / emotion / "ref.wav").exists()
        for emotion in sorted(ALL_BASE_EMOTIONS)
    }