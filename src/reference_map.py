"""
reference_map.py — Emotion → reference audio clip routing.

Clip structure on disk:
  reference_clips/
    neutral.wav          ← single clip (flat)
    whisper.wav          ← single clip (flat)
    happy/               ← multi-clip emotions (subfoldered)
      s01_r01.wav
      s01_r02.wav
      s02_r01.wav
      s02_r02.wav
    angry/ ...
    sad/   ...
    (etc.)

Source for Urdu emotion clips: SEMOUR+ (27,640 utterances, 8 emotions, 24 speakers).
Pick 2–4 clips per emotion from the same speaker for consistent conditioning.
RAVDESS English clips can still be used as a temporary cross-lingual placeholder
until Urdu clips are sourced — quality will be lower but the pipeline runs.
"""

from pathlib import Path

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

DEFAULT_CLIP_ID = "s01_r01"

# Emotions with multiple clips (subfoldered: reference_clips/<emotion>/<clip_id>.wav)
MULTI_CLIP_EMOTIONS = {
    "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
}

# Emotions with exactly one clip, flat in reference_clips/
SINGLE_CLIP_EMOTIONS: dict[str, Path] = {
    "neutral": REFERENCE_DIR / "neutral.wav",
    "whisper": REFERENCE_DIR / "whisper.wav",
}

# Clip IDs available per multi-clip emotion (2 statements × 2 reps — matches RAVDESS
# and SEMOUR+ naming convention used in download scripts).
AVAILABLE_CLIP_IDS = [f"s{s}_r{r}" for s in ("01", "02") for r in ("01", "02")]

# Sub-emotion aliases → base emotion
EMOTION_ALIASES: dict[str, str] = {
    # Happiness family
    "joy": "happy", "excitement": "happy", "contentment": "happy",
    # Sadness family
    "grief": "sad", "loneliness": "sad", "disappointment": "sad",
    # Anger family
    "rage": "angry", "frustration": "angry", "irritation": "angry",
    # Fear family
    "anxiety": "fearful", "nervousness": "fearful", "panic": "fearful",
    # Disgust family
    "contempt": "disgust", "revulsion": "disgust", "disdain": "disgust",
    # Surprise family
    "shock": "surprised", "amazement": "surprised", "disbelief": "surprised",
    # Calm family
    "serenity": "calm", "relaxation": "calm",
}

ALL_BASE_EMOTIONS = MULTI_CLIP_EMOTIONS | set(SINGLE_CLIP_EMOTIONS.keys())


def _resolve_alias(emotion: str) -> str:
    return EMOTION_ALIASES.get(emotion, emotion)


def get_reference_path(emotion: str, clip_id: str | None = None) -> Path:
    """
    Return the Path to the reference wav for the given emotion (single clip).
    For multi-clip conditioning use list_clips() instead.

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
                  f"Falling back to {DEFAULT_CLIP_ID}.")
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
            f"Add a neutral Urdu reference clip from SEMOUR+."
        )
    return fallback


def list_clips(emotion: str) -> list[Path]:
    """
    Return all available clip Paths for a given emotion (aliases resolved).
    Used by wrapper.py for multi-clip latent conditioning.
    Single-clip emotions return a 1-item list. Missing files are excluded.
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
    Check which base emotion clips exist on disk.
    For multi-clip emotions: True only if ALL 4 clips present (strict).
    """
    status: dict[str, bool] = {}

    for emotion, path in SINGLE_CLIP_EMOTIONS.items():
        status[emotion] = path.exists()

    for emotion in MULTI_CLIP_EMOTIONS:
        paths = [REFERENCE_DIR / emotion / f"{cid}.wav" for cid in AVAILABLE_CLIP_IDS]
        status[emotion] = all(p.exists() for p in paths)

    return status