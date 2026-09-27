"""
predict.py
==========
CLI entry point for running inference with a trained FDCS-Net V4 model.

Usage
-----
    python src/predict.py --model models/fdcsnet_v4_final.keras --image path/to/image.jpg
    python src/predict.py --model models/fdcsnet_v4_final.keras --dir path/to/folder/
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ and project root are on sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import resolve_and_load_model, predict_single_image
from configs.config import CLASSIFICATION_THRESHOLD, SUPPORTED_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="SynthLens: FDCS-Net V4 AI Image Detection CLI"
    )
    parser.add_argument(
        "--model", default=str(PROJECT_ROOT / "models" / "fdcsnet_v4_final.keras"),
        help="Path to saved .keras model file (default: models/fdcsnet_v4_final.keras)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--dir",   help="Path to a directory of images")
    parser.add_argument(
        "--threshold", type=float, default=CLASSIFICATION_THRESHOLD,
        help=f"Classification threshold (default: {CLASSIFICATION_THRESHOLD}, from configs/config.py)"
    )
    return parser.parse_args()


def run_on_image(model, image_path: Path | str, threshold: float) -> None:
    p = Path(image_path)
    result = predict_single_image(model, p, threshold=threshold)
    attn_str = ""
    if result.get("attention_weights"):
        aw = result["attention_weights"]
        attn_str = f" [Attn: S={aw['spatial']:.2f}, C={aw['color']:.2f}, F={aw['frequency']:.2f}]"
    print(
        f"  {p.name:<35s} "
        f"-> {result['label']:<14s} "
        f"(prob: {result['prediction']:.4f}, conf: {result['confidence']:.2%}){attn_str}"
    )


def main():
    args = parse_args()

    print(f"\nSynthLens Inference CLI | FDCS-Net V4")
    print(f"Loading model from: {args.model}")
    try:
        model = resolve_and_load_model(args.model)
    except Exception as e:
        print(f"\n[Error] Could not load model: {e}")
        sys.exit(1)

    print(f"Classification threshold: {args.threshold:.2f}\n")
    print(f"{'Image File':<35s}   {'Predicted Class':<14s}  Details")
    print("-" * 80)

    if args.image:
        run_on_image(model, args.image, args.threshold)
    else:
        dir_path = Path(args.dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"Directory not found: {args.dir}")
            sys.exit(1)

        images = [
            f for f in sorted(dir_path.iterdir())
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not images:
            print(f"No supported images found in {args.dir}")
            sys.exit(1)
        for img_path in images:
            run_on_image(model, img_path, args.threshold)

    print()


if __name__ == "__main__":
    main()