"""
scripts/build_reference_clips.py — Build per-emotion composite reference clips
for both languages, into reference_clips/<language>/<emotion>/ref.wav.

English : RAVDESS, Actor 01 (fixed — single actor per language, no --actor flag).
Urdu    : SEMOUR+, Actor 5  (fixed). Already built on disk for this project —
          this script re-runs cleanly / no-ops if files already exist, and only
          needs --semour-dir if you're rebuilding urdu/ from scratch.

Composite = 4 clips per emotion (2 statements × 2 reps), concatenated with a
short silence gap between them.

Usage:
    python scripts/build_reference_clips.py --language english
    python scripts/build_reference_clips.py --language urdu --semour-dir /path/to/SEMOUR+
    python scripts/build_reference_clips.py --language english --force
"""

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import requests

REFERENCE_DIR = Path(__file__).parent.parent / "reference_clips"
SILENCE_GAP_S = 0.3  # gap inserted between concatenated clips

# ---- English (RAVDESS) ----------------------------------------------------

RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)
RAVDESS_ACTOR = "01"

RAVDESS_EMOTION_CODES = {
    "neutral": "01", "calm": "02", "happy": "03", "sad": "04",
    "angry": "05", "fearful": "06", "disgust": "07", "surprised": "08",
}


def _ravdess_filenames(emotion_code: str) -> list[str]:
    intensity = "01" if emotion_code == "01" else "02"  # neutral has no strong variant
    return [
        f"03-01-{emotion_code}-{intensity}-{stmt}-{rep}-{RAVDESS_ACTOR}.wav"
        for stmt in ("01", "02")
        for rep in ("01", "02")
    ]


def _download_ravdess_zip() -> bytes:
    print("[DOWNLOAD] Connecting to Zenodo...")
    response = requests.get(RAVDESS_URL, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded, chunks = 0, []
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            print(f"\r[DOWNLOAD] {pct:5.1f}%  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB", end="", flush=True)
    print()
    return b"".join(chunks)


def _concat_with_gaps(wavs: list[np.ndarray], sr: int) -> np.ndarray:
    gap = np.zeros(int(sr * SILENCE_GAP_S), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        if i > 0:
            parts.append(gap)
        parts.append(w)
    return np.concatenate(parts)


def build_english(force: bool = False) -> None:
    out_root = REFERENCE_DIR / "english"

    if not force and all((out_root / e / "ref.wav").exists() for e in RAVDESS_EMOTION_CODES):
        print("[SKIP] reference_clips/english/ already fully built. Use --force to rebuild.")
        return

    zip_data = _download_ravdess_zip()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        namelist = {Path(n).name: n for n in zf.namelist()}

        for emotion, code in RAVDESS_EMOTION_CODES.items():
            dest_dir = out_root / emotion
            if dest_dir.joinpath("ref.wav").exists() and not force:
                print(f"[SKIP] {emotion} already built")
                continue

            wanted = _ravdess_filenames(code)
            wavs, sr = [], None
            for fname in wanted:
                if fname not in namelist:
                    print(f"[WARN] {emotion}: missing {fname} in zip")
                    continue
                with zf.open(namelist[fname]) as f:
                    data, file_sr = sf.read(io.BytesIO(f.read()))
                    sr = sr or file_sr
                    wavs.append(data.astype(np.float32))

            if not wavs:
                print(f"[SKIP] {emotion}: no clips found")
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            composite = _concat_with_gaps(wavs, sr)
            sf.write(str(dest_dir / "ref.wav"), composite, sr)
            print(f"[BUILD] {emotion:10s} ← {len(wavs)} clip(s)  →  {dest_dir/'ref.wav'}")

    neutral_ref = out_root / "neutral" / "ref.wav"
    neutral_flat = out_root / "neutral.wav"
    if neutral_ref.exists() and (force or not neutral_flat.exists()):
        shutil.copy2(neutral_ref, neutral_flat)
        print(f"[COPY] {neutral_flat}")

    print("[DONE] English reference clips ready.")


# ---- Urdu (SEMOUR+) ---------------------------------------------------------
# Already built on disk (Actor 5). Kept as the single source of truth in case
# it ever needs rebuilding from raw SEMOUR+ — not exercised by your current run.

SEMOUR_EMOTION_MAP = {
    "angry": "angry", "disgust": "disgust", "fear": "fearful", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "surprise": "surprised", "calm": "calm",
}
SEMOUR_SPEAKER = "speaker_05"  # Actor 5 — UNVERIFIED folder name, only matters on rebuild
SEMOUR_CLIPS_PER_EMOTION = 4


def _find_semour_clips(emotion_dir: Path, speaker: str, n: int) -> list[Path]:
    speaker_dir = emotion_dir / speaker
    if speaker_dir.exists():
        wavs = sorted(speaker_dir.glob("*.wav"))[:n]
        if wavs:
            return wavs
    return sorted(emotion_dir.glob("*.wav"))[:n]


def build_urdu(semour_dir, force: bool = False) -> None:
    out_root = REFERENCE_DIR / "urdu"

    if not force and all((out_root / e / "ref.wav").exists() for e in SEMOUR_EMOTION_MAP.values()):
        print("[SKIP] reference_clips/urdu/ already fully built. Use --force to rebuild.")
        return

    if semour_dir is None:
        print("[ERROR] Urdu clips not fully built and no --semour-dir given — cannot rebuild.")
        sys.exit(1)

    for semour_name, internal_name in SEMOUR_EMOTION_MAP.items():
        dest_dir = out_root / internal_name
        if dest_dir.joinpath("ref.wav").exists() and not force:
            print(f"[SKIP] {internal_name} already built")
            continue

        emotion_dir = semour_dir / semour_name
        clips = _find_semour_clips(emotion_dir, SEMOUR_SPEAKER, SEMOUR_CLIPS_PER_EMOTION)
        if not clips:
            print(f"[SKIP] {internal_name}: no clips found in {emotion_dir}")
            continue

        wavs, sr = [], None
        for c in clips:
            data, file_sr = sf.read(str(c))
            sr = sr or file_sr
            wavs.append(data.astype(np.float32))

        dest_dir.mkdir(parents=True, exist_ok=True)
        composite = _concat_with_gaps(wavs, sr)
        sf.write(str(dest_dir / "ref.wav"), composite, sr)
        print(f"[BUILD] {internal_name:10s} ← {len(wavs)} clip(s)  →  {dest_dir/'ref.wav'}")

    neutral_ref = out_root / "neutral" / "ref.wav"
    neutral_flat = out_root / "neutral.wav"
    if neutral_ref.exists() and (force or not neutral_flat.exists()):
        shutil.copy2(neutral_ref, neutral_flat)
        print(f"[COPY] {neutral_flat}")

    print("[DONE] Urdu reference clips ready.")


def main():
    parser = argparse.ArgumentParser(description="Build per-emotion composite reference clips.")
    parser.add_argument("--language", choices=["english", "urdu"], required=True)
    parser.add_argument("--semour-dir", type=Path, default=None,
                         help="Path to raw SEMOUR+ root (only needed to rebuild urdu clips from scratch)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if ref.wav already exists")
    args = parser.parse_args()

    if args.language == "english":
        build_english(force=args.force)
    else:
        build_urdu(args.semour_dir, force=args.force)


if __name__ == "__main__":
    main()