"""
Expressive TTS CLI — Level 1 Wrapper

Usage:
    python main.py "<happy>Hello, how are you?</happy>"
    python main.py "<sad>I miss you so much.</sad>" --output sad_test.wav
    python main.py "<angry>This is unacceptable!</angry>" --output angry_test.wav
    python main.py "<whisper>Can you keep a secret?</whisper>"
    python main.py "<neutral>The meeting is at three PM.</neutral>"

Supported tags: <neutral> <happy> <sad> <angry> <whisper>
Output lands in: output/
"""

import argparse
import sys
from src.wrapper import ExpressionWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expressive TTS wrapper around XTTS-v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help='Tagged input string. Example: "<happy>Hello world</happy>"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output filename inside output/ directory (default: output.wav)",
    )

    args = parser.parse_args()

    try:
        wrapper = ExpressionWrapper()
        output_path = wrapper.synthesize(args.input, args.output)
        print(f"\n[DONE] Audio saved: {output_path}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
