"""
Throwaway A/B script — synthesizes the test sentence using every clip_id
for a given emotion, so you can listen and compare convincingness.

Does NOT touch wrapper.py or main.py. Once a winning clip_id is picked
per emotion, that's when DEFAULT_CLIP_ID or a per-emotion default map
gets wired into reference_map.py properly.

Usage:
    python scripts/ab_test_clips.py happy
    python scripts/ab_test_clips.py happy --text "Custom sentence here"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf

from src.reference_map import list_clips, MULTI_CLIP_EMOTIONS
from src.tag_parser import parse_input
from src.wrapper import ExpressionWrapper, OUTPUT_DIR, SAMPLE_RATE

DEFAULT_TEST_SENTENCE = "Hello, how are you today?"


def main():
    parser = argparse.ArgumentParser(description="A/B test clip_ids for one emotion")
    parser.add_argument("emotion", help=f"One of: {sorted(MULTI_CLIP_EMOTIONS)}")
    parser.add_argument("--text", default=DEFAULT_TEST_SENTENCE, help="Test sentence")
    args = parser.parse_args()

    if args.emotion not in MULTI_CLIP_EMOTIONS:
        print(f"[ERROR] '{args.emotion}' has no multiple clips. "
              f"Choose from: {sorted(MULTI_CLIP_EMOTIONS)}")
        sys.exit(1)

    clips = list_clips(args.emotion)
    if not clips:
        print(f"[ERROR] No clips found on disk for '{args.emotion}'.")
        sys.exit(1)

    print(f"[AB TEST] Emotion : {args.emotion}")
    print(f"[AB TEST] Text    : \"{args.text}\"")
    print(f"[AB TEST] Clips   : {[c.stem for c in clips]}")
    print()

    wrapper = ExpressionWrapper()

    for clip_path in clips:
        clip_id = clip_path.stem  # e.g. "s01_r01"
        tagged_input = f"<{args.emotion}>{args.text}</{args.emotion}>"
        output_filename = f"ab_{args.emotion}_{clip_id}.wav"

        # Directly reuse the wrapper's internal synth, but override the
        # reference clip it picks — bypassing get_reference_path's default
        # so we hit this exact clip_id.
        print(f"[AB TEST] --- {clip_id} ---")
        _, segments = parse_input(tagged_input)

        tmp_path = OUTPUT_DIR / "_tmp_ab.wav"
        audio_parts = []
        for seg in segments:
            if seg["type"] == "pause":
                audio_parts.append(wrapper._make_silence(seg["duration_ms"]))
            else:
                audio = wrapper._synth_text(seg["content"], clip_path, tmp_path)
                audio = wrapper._apply_speed(audio, seg["speed"])
                audio_parts.append(audio)

        if tmp_path.exists():
            tmp_path.unlink()

        final = np.concatenate(audio_parts)
        out_path = OUTPUT_DIR / output_filename
        sf.write(str(out_path), final, SAMPLE_RATE)
        print(f"[AB TEST] -> {out_path}")
        print()

    print("[AB TEST] Done. Listen to output/ab_<emotion>_s*_r*.wav and compare.")


if __name__ == "__main__":
    main()