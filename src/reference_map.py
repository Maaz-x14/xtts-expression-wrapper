from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

DEFAULT_CLIP_ID = "s01_r01"

# Emotions with multiple clips (subfoldered: reference_clips/<emotion>/<clip_id>.wav)
MULTI_CLIP_EMOTIONS = {
    "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
}

# Emotions with exactly one clip, flat in reference_clips/ (no subfolder,
# nothing to select between).
SINGLE_CLIP_EMOTIONS: dict[str, Path] = {
    "neutral": REFERENCE_DIR / "neutral.wav",
    "whisper": REFERENCE_DIR / "whisper.wav",  # manual clip, not RAVDESS
}

# All available clip_ids for multi-clip emotions (2 statements x 2 reps).
AVAILABLE_CLIP_IDS = [f"s{s}_r{r}" for s in ("01", "02") for r in ("01", "02")]

# Sub-emotion tag -> closest base emotion. No clip of their own; resolved via
# _resolve_alias() before lookup. Speculative mapping (Step 2 listening pass
# validated base emotions only, not these aliases).
EMOTION_ALIASES: dict[str, str] = {
    "joy": "happy", "excitement": "happy", "contentment": "happy",
    "grief": "sad", "loneliness": "sad", "disappointment": "sad",
    "rage": "angry", "frustration": "angry", "irritation": "angry",
    "anxiety": "fearful", "nervousness": "fearful", "panic": "fearful",
    "contempt": "disgust", "revulsion": "disgust", "disdain": "disgust",
    "shock": "surprised", "amazement": "surprised", "disbelief": "surprised",
    "serenity": "calm", "relaxation": "calm",
}

# Union of every base emotion (multi- or single-clip), used by tag_parser.py
# to build SUPPORTED_EMOTION_TAGS.
ALL_BASE_EMOTIONS = MULTI_CLIP_EMOTIONS | set(SINGLE_CLIP_EMOTIONS.keys())


def _resolve_alias(emotion: str) -> str:
    """Resolve a sub-emotion alias to its base emotion. No-op if already base."""
    return EMOTION_ALIASES.get(emotion, emotion)


def get_reference_path(emotion: str, clip_id: str | None = None) -> Path:
    """
    Return the Path to the reference wav for the given emotion.

    Args:
        emotion : base emotion or sub-emotion alias (resolved automatically)
        clip_id : which of the 4 clips to use for multi-clip emotions, e.g.
                  "s01_r02". Defaults to DEFAULT_CLIP_ID ("s01_r01"), which
                  matches prior single-clip behavior exactly. Ignored for
                  single-clip emotions (neutral, whisper).

    Falls back to neutral if the emotion/clip combination doesn't exist.
    """
    emotion = _resolve_alias(emotion)

    if emotion in SINGLE_CLIP_EMOTIONS:
        path = SINGLE_CLIP_EMOTIONS[emotion]
        if not path.exists():
            return _fallback_to_neutral(f"Clip not found: {path}.")
        return path

    if emotion in MULTI_CLIP_EMOTIONS:
        resolved_clip_id = clip_id or DEFAULT_CLIP_ID
        if resolved_clip_id not in AVAILABLE_CLIP_IDS:
            print(f"[REFMAP] Unknown clip_id '{resolved_clip_id}' for '{emotion}'. "
                  f"Available: {AVAILABLE_CLIP_IDS}. Falling back to {DEFAULT_CLIP_ID}.")
            resolved_clip_id = DEFAULT_CLIP_ID
        path = REFERENCE_DIR / emotion / f"{resolved_clip_id}.wav"
        if not path.exists():
            return _fallback_to_neutral(f"Clip not found: {path}.")
        return path

    return _fallback_to_neutral(f"Emotion '{emotion}' not in map.")


def _fallback_to_neutral(reason: str) -> Path:
    print(f"[REFMAP] {reason} Falling back to neutral.")
    fallback = SINGLE_CLIP_EMOTIONS["neutral"]
    if not fallback.exists():
        raise FileNotFoundError(
            f"Neutral fallback clip not found at {fallback}. "
            f"Run: python scripts/download_ravdess.py"
        )
    return fallback


def list_clips(emotion: str) -> list[Path]:
    """
    Return all available clip Paths for a given base emotion (aliases resolved).
    Single-clip emotions return a 1-item list. Missing files are excluded.
    Useful for A/B testing across clip_ids.
    """
    emotion = _resolve_alias(emotion)

    if emotion in SINGLE_CLIP_EMOTIONS:
        path = SINGLE_CLIP_EMOTIONS[emotion]
        return [path] if path.exists() else []

    if emotion in MULTI_CLIP_EMOTIONS:
        paths = [REFERENCE_DIR / emotion / f"{cid}.wav" for cid in AVAILABLE_CLIP_IDS]
        return [p for p in paths if p.exists()]

    return []


def verify_all_clips() -> dict[str, bool]:
    """
    Check which BASE emotion clips are present on disk.
    For multi-clip emotions, True only if ALL 4 clips exist (strict check --
    partial downloads should be visible, not silently masked).
    Aliases aren't checked -- they resolve to a base emotion's clip(s).
    """
    status: dict[str, bool] = {}

    for emotion, path in SINGLE_CLIP_EMOTIONS.items():
        status[emotion] = path.exists()

    for emotion in MULTI_CLIP_EMOTIONS:
        paths = [REFERENCE_DIR / emotion / f"{cid}.wav" for cid in AVAILABLE_CLIP_IDS]
        status[emotion] = all(p.exists() for p in paths)

    return status