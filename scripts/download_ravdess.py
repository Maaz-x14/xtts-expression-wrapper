"""
Download RAVDESS speech dataset and extract one reference clip per emotion.

RAVDESS filename format: 03-01-{emotion}-{intensity}-{statement}-{rep}-{actor}.wav
  Modality  : 03 = audio-only
  Channel   : 01 = speech
  Emotion   : 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
  Intensity : 01=normal 02=strong (neutral has no strong variant)
  Statement : 01="Kids are talking by the door"
  Rep       : 01 or 02
  Actor     : 01–24

We pick Actor 01, Statement 01, Rep 01 for all clips. Every actor performs
all 60 trials (7 emotions x 2 intensities, minus neutral's missing strong,
plus calm x2 and neutral x1 = 60), so Actor 01/Statement 01/Rep 01 exists
for every emotion below — no fallback logic needed.

'whisper' is kept as a separate manually-provided clip (not RAVDESS-derived,
perceptually distinct from calm — hushed/breathy vs. relaxed/slow) and is
NOT touched by this script.

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

# Maps our emotion label -> exact RAVDESS filename we want
TARGET_FILES: dict[str, str] = {
    "neutral":   "03-01-01-01-01-01-01.wav",  # neutral, normal intensity (only option)
    "calm":      "03-01-02-02-01-01-01.wav",  # calm, strong intensity
    "happy":     "03-01-03-02-01-01-01.wav",  # happy, strong intensity
    "sad":       "03-01-04-02-01-01-01.wav",  # sad, strong intensity
    "angry":     "03-01-05-02-01-01-01.wav",  # angry, strong intensity
    "fearful":   "03-01-06-02-01-01-01.wav",  # fearful, strong intensity
    "disgust":   "03-01-07-02-01-01-01.wav",  # disgust, strong intensity
    "surprised": "03-01-08-02-01-01-01.wav",  # surprised, strong intensity
    # NOTE: "whisper" is intentionally excluded — separate manual clip, not RAVDESS.
}


def download_zip(url: str) -> bytes:
    print("[DOWNLOAD] Connecting to Zenodo...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunks: list[bytes] = []

    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            mb_done = downloaded / 1024 / 1024
            mb_total = total / 1024 / 1024
            print(f"\r[DOWNLOAD] {pct:5.1f}%  {mb_done:.0f} / {mb_total:.0f} MB", end="", flush=True)

    print()  # newline after progress
    print(f"[DOWNLOAD] Complete — {downloaded / 1024 / 1024:.1f} MB received.")
    return b"".join(chunks)


def extract_clips(zip_data: bytes) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build reverse lookup: ravdess_filename -> our emotion label
    filename_to_emotion = {v: k for k, v in TARGET_FILES.items()}

    print("[EXTRACT] Scanning zip contents...")
    found: dict[str, str] = {}  # emotion -> zip internal path

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for name in zf.namelist():
            basename = Path(name).name
            if basename in filename_to_emotion:
                emotion = filename_to_emotion[basename]
                found[emotion] = name

        if not found:
            print("[ERROR] No target files found in zip. Check RAVDESS zip structure.")
            sys.exit(1)

        for emotion, zip_path in found.items():
            out_path = OUTPUT_DIR / f"{emotion}.wav"
            with zf.open(zip_path) as src:
                out_path.write_bytes(src.read())
            print(f"[EXTRACT] {emotion:10s} <- {Path(zip_path).name}  ->  {out_path}")

    missing = set(TARGET_FILES.keys()) - set(found.keys())
    if missing:
        print(f"[WARN] Could not locate clips for: {sorted(missing)}")
        print("[WARN] These emotions will fall back to neutral at runtime.")
    else:
        print(f"[DONE] All {len(TARGET_FILES)} reference clips extracted successfully.")
        print("[INFO] 'whisper.wav' is untouched — manage that clip manually.")


def main() -> None:
    print("=" * 55)
    print(" RAVDESS Reference Clip Downloader")
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