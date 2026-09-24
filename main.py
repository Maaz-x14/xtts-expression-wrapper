"""
Expressive TTS — XTTS v2, dual-model / dual-language testing harness.

Usage:
    python main.py "<happy>Hello there</happy>" --model base --language english
    python main.py "<happy>السلام علیکم</happy>" --model finetuned --language urdu

Environment:
    XTTS_MODEL_DIR   override path to the fine-tuned checkpoint dir (default: ./Agri-TTS)

Output: output/<language>/
"""

import argparse
import sys
from src.wrapper import ExpressionWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expressive TTS — XTTS v2, base vs fine-tuned, English vs Urdu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=str, help='Tagged input. Example: "<happy>Hello</happy>"')
    parser.add_argument("--model", choices=["base", "finetuned"], default="finetuned",
                         help="Which checkpoint to load (default: finetuned)")
    parser.add_argument("--language", choices=["english", "urdu"], default="urdu",
                         help="Which reference-clip set + XTTS language code to use (default: urdu)")
    parser.add_argument("--output", type=str, default="output.wav",
                         help="Output filename inside output/<language>/ (default: output.wav)")

    args = parser.parse_args()

    try:
        wrapper = ExpressionWrapper(model=args.model, language=args.language)
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