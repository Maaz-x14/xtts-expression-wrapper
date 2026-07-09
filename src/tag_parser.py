import re

SUPPORTED_TAGS = {"neutral", "happy", "sad", "angry", "whisper"}

def parse_input(text: str) -> tuple[str, str]:
    """
    Parse a tagged input string into (emotion, clean_text).

    Expected format: <happy>Hello world</happy>

    Rules:
    - Tag must be opening + closing pair: <tag>...</tag>
    - If no tag found → fallback to neutral
    - If tag found but not in SUPPORTED_TAGS → warn, fallback to neutral
    - Whitespace inside tags is stripped
    """
    pattern = r"<(\w+)>(.*?)</\1>"
    match = re.search(pattern, text.strip(), re.DOTALL)

    if not match:
        print(f"[PARSER] No valid tag found. Defaulting to neutral.")
        return "neutral", text.strip()

    emotion = match.group(1).lower()
    content = match.group(2).strip()

    if emotion not in SUPPORTED_TAGS:
        print(f"[PARSER] Unknown tag <{emotion}>. Supported: {sorted(SUPPORTED_TAGS)}. Defaulting to neutral.")
        return "neutral", content

    if not content:
        print(f"[PARSER] Tag <{emotion}> is empty. Nothing to synthesize.")
        raise ValueError(f"Empty text content inside <{emotion}> tag.")

    return emotion, content
