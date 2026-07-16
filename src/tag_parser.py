import re

SUPPORTED_EMOTION_TAGS = {"neutral", "happy", "sad", "angry", "whisper"}

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

    Example:
      input  → 'Hello <pause=300ms> how are <fast>you</fast> today?'
      output → [
          {"type": "text",  "content": "Hello",   "speed": 1.0},
          {"type": "pause", "duration_ms": 300},
          {"type": "text",  "content": "how are", "speed": 1.0},
          {"type": "text",  "content": "you",     "speed": 1.5},
          {"type": "text",  "content": "today?",  "speed": 1.0},
      ]
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
        emotion  : str        — one of SUPPORTED_EMOTION_TAGS
        segments : list[dict] — ordered list of text/pause segments

    Supported format:
        <happy>Hello <pause=300ms> how are <fast>you</fast> today?</happy>
    """
    emotion, inner = _parse_emotion_tag(text)
    segments = _parse_segments(inner)
    return emotion, segments
