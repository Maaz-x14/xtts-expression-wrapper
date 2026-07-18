"""
Download RAVDESS speech dataset and extract multiple reference clips per emotion.

RAVDESS filename format: 03-01-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav
  Modality  : 03 = audio-only
  Channel   : 01 = speech
  Emotion   : 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
  Intensity : 01=normal 02=strong (neutral has no strong variant)
  Statement : 01="Kids are talking by the door"  02="Dogs are sitting by the door"
  Rep       : 01 or 02
  Actor     : 01-24

Scope (Level 2 multi-clip): Actor 01, both statements (01/02), both reps (01/02),
strong intensity -> 4 clips per emotion, for the 7 emotions that have a strong
intensity variant (calm, happy, sad, angry, fearful, disgust, surprised).

neutral has only 1 clip (normal intensity, no strong variant exists) and stays
flat at reference_clips/neutral.wav -- not subfoldered, nothing to select between.

whisper.wav is a manually-provided clip, entirely outside RAVDESS/this script.

Output layout:
  reference_clips/
    neutral.wav                  <- unchanged, flat, single clip
    whisper.wav                  <- unchanged, flat, manual clip (untouched by this script)
    calm/s01_r01.wav, calm/s01_r02.wav, calm/s02_r01.wav, calm/s02_r02.wav
    happy/... (same 4-clip pattern)
    sad/...
    angry/...
    fearful/...
    disgust/...
    surprised/...

Source: https://zenodo.org/record/1188976
License: CC BY-NC-SA 4.0
"""

import io
import sys
import zipfile
from pathlib import Path

import requests

ZENODO_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)

OUTPUT_DIR = Path(__file__).parent.parent / "reference_clips"

RAVDESS_EMOTION_CODES = {
    "calm":      "02",
    "happy":     "03",
    "sad":       "04",
    "angry":     "05",
    "fearful":   "06",
    "disgust":   "07",
    "surprised": "08",
}

_STATEMENTS = ["01", "02"]
_REPS = ["01", "02"]

# Single-clip emotions handled separately (unchanged behavior from before)
SINGLE_CLIP_TARGETS: dict[str, str] = {
    "neutral": "03-01-01-01-01-01-01.wav",  # neutral, normal intensity (only option)
}
# NOTE: "whisper" excluded entirely -- separate manual clip, not RAVDESS, not
# touched by this script.


def _build_multi_clip_targets() -> dict[str, dict[str, str]]:
    """
    Build {emotion: {clip_id: ravdess_filename}} for the 7 multi-clip emotions.
    clip_id format: "s{statement}_r{rep}", e.g. "s01_r02".
    """
    targets: dict[str, dict[str, str]] = {}
    for emotion, code in RAVDESS_EMOTION_CODES.items():
        clips = {}
        for stmt in _STATEMENTS:
            for rep in _REPS:
                clip_id = f"s{stmt}_r{rep}"
                filename = f"03-01-{code}-02-{stmt}-{rep}-01.wav"
                clips[clip_id] = filename
        targets[emotion] = clips
    return targets


MULTI_CLIP_TARGETS = _build_multi_clip_targets()


def download_zip(url: str) -> bytes:
    print("[DOWNLOAD] Connecting to Zenodo...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks: list[bytes] = []

    for chunk in response.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            mb_done = downloaded / 1024 / 1024
            mb_total = total / 1024 / 1024
            print(f"\r[DOWNLOAD] {pct:5.1f}%  {mb_done:.0f} / {mb_total:.0f} MB", end="", flush=True)

    print()
    print(f"[DOWNLOAD] Complete — {downloaded / 1024 / 1024:.1f} MB received.")
    return b"".join(chunks)


def extract_clips(zip_data: bytes) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename_to_target: dict[str, tuple[str, str | None]] = {}
    for emotion, filename in SINGLE_CLIP_TARGETS.items():
        filename_to_target[filename] = (emotion, None)
    for emotion, clips in MULTI_CLIP_TARGETS.items():
        for clip_id, filename in clips.items():
            filename_to_target[filename] = (emotion, clip_id)

    print("[EXTRACT] Scanning zip contents...")
    found: dict[str, str] = {}

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            basename = Path(name).name
            if basename in filename_to_target:
                emotion, clip_id = filename_to_target[basename]
                key = emotion if clip_id is None else f"{emotion}/{clip_id}"
                found[key] = name

        if not found:
            print("[ERROR] No target files found in zip. Check RAVDESS zip structure.")
            sys.exit(1)

        for key, zip_path in found.items():
            if "/" in key:
                emotion, clip_id = key.split("/")
                emotion_dir = OUTPUT_DIR / emotion
                emotion_dir.mkdir(exist_ok=True)
                out_path = emotion_dir / f"{clip_id}.wav"
            else:
                out_path = OUTPUT_DIR / f"{key}.wav"

            with zf.open(zip_path) as src:
                out_path.write_bytes(src.read())
            print(f"[EXTRACT] {key:20s} <- {Path(zip_path).name}  ->  {out_path}")

    expected_keys = set(SINGLE_CLIP_TARGETS.keys()) | {
        f"{emotion}/{clip_id}"
        for emotion, clips in MULTI_CLIP_TARGETS.items()
        for clip_id in clips
    }
    missing = expected_keys - set(found.keys())
    if missing:
        print(f"[WARN] Could not locate clips for: {sorted(missing)}")
    else:
        print(f"[DONE] All {len(expected_keys)} reference clips extracted successfully.")
        print("[INFO] 'whisper.wav' is untouched — manage that clip manually.")


def main() -> None:
    print("=" * 55)
    print(" RAVDESS Reference Clip Downloader (multi-clip)")
    print(" Source : zenodo.org/record/1188976")
    print(" License: CC BY-NC-SA 4.0 (research use only)")
    print("=" * 55)

    zip_data = download_zip(ZENODO_URL)
    extract_clips(zip_data)

    print()
    print("[INFO] Reference clips are in: reference_clips/")
    print("[INFO] These are NOT committed to git (see .gitignore).")
    print("[INFO] Re-run this script after cloning on a new machine.")


if __name__ == "__main__":
    main()