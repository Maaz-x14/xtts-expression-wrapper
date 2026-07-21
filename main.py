"""
Expressive Urdu TTS CLI — XTTS v2 fine-tune + Auralis backend

Model    : Agri-TTS/ (local fine-tuned checkpoint)
Language : Urdu (ur)
Backend  : Auralis (latent-cached, multi-clip conditioning)

Environment:
    XTTS_MODEL_DIR   path to checkpoint dir (default: ./Agri-TTS)

Usage examples (Urdu):
    python main.py "<happy>السلام علیکم، آپ کیسے ہیں؟</happy>"
    python main.py "<sad>مجھے آپ کی یاد آتی ہے <pause=500ms> بہت زیادہ۔</sad>"
    python main.py "<angry>یہ بالکل <pause=300ms> ناقابل قبول ہے!</angry>"
    python main.py "<neutral>براہ کرم <slow>آہستہ آہستہ</slow> بولیں۔</neutral>"
    python main.py "<happy>سنیں <fast>یہ بات جلدی سے</fast> بتاتا ہوں۔</happy>"

Usage examples (Roman Urdu — works too):
    python main.py "<happy>Salam, aap kaise hain?</happy>"
    python main.py "<sad>Mujhe aap ki yaad aati hai <pause=500ms> bohat zyada.</sad>"

Output: output/
"""

import argparse
import sys
from src.wrapper import ExpressionWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expressive Urdu TTS — XTTS v2 fine-tune + Auralis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help='Tagged input. Example: "<happy>السلام علیکم</happy>"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output filename inside output/ (default: output.wav)",
    )
    parser.add_argument(
        "--single-clip",
        action="store_true",
        help="Use single reference clip instead of multi-clip conditioning",
    )

    args = parser.parse_args()

    try:
        wrapper = ExpressionWrapper(use_multi_clip=not args.single_clip)
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