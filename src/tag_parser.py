import re

from src.reference_map import ALL_BASE_EMOTIONS, EMOTION_ALIASES

# Outer tags a user is allowed to write. Base emotions (have clip(s)) +
# sub-emotion aliases (resolved to a base emotion's clip in reference_map.py).
# Built from reference_map.py so this file never drifts out of sync with it.
SUPPORTED_EMOTION_TAGS = ALL_BASE_EMOTIONS | set(EMOTION_ALIASES.keys())

PAUSE_DEFAULTS_MS = {
    "break":   500,
    "silence": 1000,
}

SPEED_VALUES = {
    "fast": 1.5,
    "slow": 0.7,
}

_INLINE_PATTERN = re.compile(
    r'(<pause=(\d+)(?:ms)?>|<break>|<silence>|<(fast|slow)>(.*?)</\3>)',
    re.DOTALL,
)


def _parse_emotion_tag(text: str) -> tuple[str, str]:
    """
    Extract outer emotion tag and inner content.
    Returns (emotion, inner_content).
    Falls back to neutral if tag missing or unsupported.
    Note: emotion returned here may be an alias (e.g. "joy") — resolution to
    a base emotion happens downstream in reference_map.get_reference_path().
    """
    match = re.match(r'^<(\w+)>(.*)</\1>$', text.strip(), re.DOTALL)

    if not match:
        print("[PARSER] No emotion tag found. Defaulting to neutral.")
        return "neutral", text.strip()

    emotion = match.group(1).lower()
    inner   = match.group(2).strip()

    if emotion not in SUPPORTED_EMOTION_TAGS:
        print(f"[PARSER] Unknown emotion <{emotion}>. "
              f"Supported: {sorted(SUPPORTED_EMOTION_TAGS)}. Defaulting to neutral.")
        return "neutral", inner

    return emotion, inner


def _parse_segments(inner: str) -> list[dict]:
    """
    Split inner content into ordered list of segments.

    Segment types:
      {"type": "text",  "content": str, "speed": float}
      {"type": "pause", "duration_ms": int}
    """
    segments = []
    cursor = 0

    for match in _INLINE_PATTERN.finditer(inner):
        before = inner[cursor:match.start()].strip()
        if before:
            segments.append({"type": "text", "content": before, "speed": 1.0})

        full = match.group(0)

        if full.startswith("<pause="):
            segments.append({"type": "pause", "duration_ms": int(match.group(2))})

        elif full == "<break>":
            segments.append({"type": "pause", "duration_ms": PAUSE_DEFAULTS_MS["break"]})

        elif full == "<silence>":
            segments.append({"type": "pause", "duration_ms": PAUSE_DEFAULTS_MS["silence"]})

        else:  # <fast> or <slow>
            tag_name = match.group(3)
            content  = match.group(4).strip()
            if content:
                segments.append({
                    "type":    "text",
                    "content": content,
                    "speed":   SPEED_VALUES[tag_name],
                })

        cursor = match.end()

    remaining = inner[cursor:].strip()
    if remaining:
        segments.append({"type": "text", "content": remaining, "speed": 1.0})

    if not segments:
        raise ValueError("No content found after parsing tags.")

    return segments


def parse_input(text: str) -> tuple[str, list[dict]]:
    """
    Full parse of a tagged input string.

    Returns:
        emotion  : str        — a SUPPORTED_EMOTION_TAGS member (base or alias);
                                 caller must resolve aliases via reference_map
                                 before using this as a lookup key elsewhere.
        segments : list[dict] — ordered list of text/pause segments
    """
    emotion, inner = _parse_emotion_tag(text)
    segments = _parse_segments(inner)
    return emotion, segments