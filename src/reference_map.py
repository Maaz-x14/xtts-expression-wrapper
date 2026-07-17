from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

# Maps each BASE emotion (one with an actual reference clip) to its wav file.
# whisper is a manually-provided clip, perceptually distinct from calm
# (hushed/breathy vs. relaxed/slow) — kept separate, not RAVDESS-derived.
EMOTION_TO_CLIP: dict[str, Path] = {
    "neutral":   REFERENCE_DIR / "neutral.wav",
    "calm":      REFERENCE_DIR / "calm.wav",
    "happy":     REFERENCE_DIR / "happy.wav",
    "sad":       REFERENCE_DIR / "sad.wav",
    "angry":     REFERENCE_DIR / "angry.wav",
    "fearful":   REFERENCE_DIR / "fearful.wav",
    "disgust":   REFERENCE_DIR / "disgust.wav",
    "surprised": REFERENCE_DIR / "surprised.wav",
    "whisper":   REFERENCE_DIR / "whisper.wav",
}

# Sub-emotion tag -> closest base emotion. These have no clip of their own;
# they resolve to a base emotion's clip via _resolve_alias() below.
# Speculative mapping (not validated by listening tests yet — that's Step 2).
EMOTION_ALIASES: dict[str, str] = {
    # -> happy
    "joy": "happy",
    "excitement": "happy",
    "contentment": "happy",
    # -> sad
    "grief": "sad",
    "loneliness": "sad",
    "disappointment": "sad",
    # -> angry
    "rage": "angry",
    "frustration": "angry",
    "irritation": "angry",
    # -> fearful
    "anxiety": "fearful",
    "nervousness": "fearful",
    "panic": "fearful",
    # -> disgust
    "contempt": "disgust",
    "revulsion": "disgust",
    "disdain": "disgust",
    # -> surprised
    "shock": "surprised",
    "amazement": "surprised",
    "disbelief": "surprised",
    # -> calm
    "serenity": "calm",
    "relaxation": "calm",
}


def _resolve_alias(emotion: str) -> str:
    """Resolve a sub-emotion alias to its base emotion. No-op if already base."""
    return EMOTION_ALIASES.get(emotion, emotion)


def get_reference_path(emotion: str) -> Path:
    """
    Return the Path to the reference wav for the given emotion.
    Resolves sub-emotion aliases to their base emotion first.
    Falls back to neutral if the clip doesn't exist on disk.
    """
    emotion = _resolve_alias(emotion)
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
    Check which BASE emotion clips are present on disk.
    (Aliases aren't checked — they resolve to a base emotion's clip.)
    Returns a dict of emotion -> exists.
    """
    return {emotion: path.exists() for emotion, path in EMOTION_TO_CLIP.items()}