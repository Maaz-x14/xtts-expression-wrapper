"""
prepare_urdu_clips.py — Select and copy Urdu reference clips from SEMOUR+
into the reference_clips/<emotion>/ structure expected by reference_map.py.

Usage:
    python scripts/prepare_urdu_clips.py --semour-dir /path/to/SEMOUR+

SEMOUR+ folder structure assumed:
    SEMOUR+/
        angry/
            speaker_01/
                ...wav files...
            speaker_02/
            ...
        happy/
        sad/
        ...

Strategy:
    - Pick ONE speaker (default: first available, ideally female for naturalness)
    - Take 4 clips per emotion (2 utterances × 2 takes → s01_r01, s01_r02, s02_r01, s02_r02)
    - Copy into reference_clips/<emotion>/s{X}_r{Y}.wav

Adjust SPEAKER_PREFERENCE and UTTERANCE_INDICES to taste.
"""

import argparse
import shutil
from pathlib import Path

# ---- Configuration --------------------------------------------------------

# Emotion name mapping: SEMOUR+ folder name → our internal key
# Adjust if SEMOUR+ uses different folder names
EMOTION_MAP = {
    "angry":     "angry",
    "disgust":   "disgust",
    "fear":      "fearful",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprise":  "surprised",
    "calm":      "calm",
}

# Prefer a single speaker for consistent voice identity across emotions.
# Set to None to auto-pick the first available speaker folder.
SPEAKER_PREFERENCE = None  # e.g. "speaker_01"

# How many clips to select per emotion (max 4, maps to s01_r01..s02_r02)
CLIPS_PER_EMOTION = 4

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"

# ---------------------------------------------------------------------------


def find_clips(emotion_dir: Path, speaker_pref: str | None, n: int) -> list[Path]:
    """Pick n wav files from emotion_dir, preferring a specific speaker subfolder."""
    if not emotion_dir.exists():
        return []

    # Try speaker preference first
    if speaker_pref:
        speaker_dir = emotion_dir / speaker_pref
        if speaker_dir.exists():
            wavs = sorted(speaker_dir.glob("*.wav"))[:n]
            if wavs:
                return wavs

    # Fall back: walk all subdirs, take first speaker found
    for subdir in sorted(emotion_dir.iterdir()):
        if subdir.is_dir():
            wavs = sorted(subdir.glob("*.wav"))[:n]
            if wavs:
                return wavs

    # Flat structure (no speaker subdirs)
    return sorted(emotion_dir.glob("*.wav"))[:n]


def main():
    parser = argparse.ArgumentParser(description="Prepare Urdu reference clips from SEMOUR+")
    parser.add_argument("--semour-dir", required=True, type=Path,
                        help="Path to SEMOUR+ root directory")
    parser.add_argument("--speaker", default=SPEAKER_PREFERENCE,
                        help="Preferred speaker subfolder name (default: auto)")
    parser.add_argument("--clips", type=int, default=CLIPS_PER_EMOTION,
                        help="Clips per emotion (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be copied without copying")
    args = parser.parse_args()

    semour = Path(args.semour_dir)
    if not semour.exists():
        print(f"[ERROR] SEMOUR+ dir not found: {semour}")
        return

    clip_ids = [f"s{s}_r{r}" for s in ("01", "02") for r in ("01", "02")]

    for semour_name, internal_name in EMOTION_MAP.items():
        emotion_dir = semour / semour_name
        clips = find_clips(emotion_dir, args.speaker, args.clips)

        if not clips:
            print(f"[SKIP] {internal_name}: no clips found in {emotion_dir}")
            continue

        dest_dir = REFERENCE_DIR / internal_name
        if not args.dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(clips):
            if i >= len(clip_ids):
                break
            dest = dest_dir / f"{clip_ids[i]}.wav"
            if args.dry_run:
                print(f"[DRY] {src} → {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"[COPY] {src.name} → {dest}")

        print(f"[OK] {internal_name}: {len(clips)} clip(s) prepared")

    # Handle neutral separately (single flat clip)
    neutral_src = REFERENCE_DIR / "neutral" / "s01_r01.wav"
    neutral_dst = REFERENCE_DIR / "neutral.wav"
    if neutral_src.exists() and not neutral_dst.exists():
        if args.dry_run:
            print(f"[DRY] {neutral_src} → {neutral_dst}")
        else:
            shutil.copy2(neutral_src, neutral_dst)
            print(f"[COPY] neutral.wav (flat copy for fallback)")

    print("\n[DONE] Reference clips ready.")
    print(f"       Run: python main.py \"<happy>السلام علیکم</happy>\"")


if __name__ == "__main__":
    main()
