"""
Expressive TTS CLI — Stage 2 Wrapper

Supported outer tags  : <neutral> <happy> <sad> <angry> <whisper>
Supported inline tags : <pause=Xms>  <break>  <silence>  <fast>...</fast>  <slow>...</slow>

Usage examples:
    python main.py "<happy>Hello, how are you today?</happy>"
    python main.py "<sad>I miss you <pause=500ms> so much.</sad>"
    python main.py "<angry>This is <pause=300ms> completely unacceptable!</angry>"
    python main.py "<happy>Welcome <break> to the show <pause=200ms> everyone!</happy>"
    python main.py "<neutral>Please <slow>take your time</slow> with this.</neutral>"
    python main.py "<happy>Let me tell you <fast>this very quickly</fast> before I forget.</happy>"
    python main.py "<sad>I wanted to say <pause=400ms> <slow>goodbye</slow>.</sad>"

Output lands in: output/
"""

import argparse
import sys
from src.wrapper import ExpressionWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expressive TTS — Stage 2 wrapper with prosodic tags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help='Tagged input. Example: "<happy>Hello <pause=300ms> world!</happy>"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output filename inside output/ (default: output.wav)",
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
        print("\n[ABORT] Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
