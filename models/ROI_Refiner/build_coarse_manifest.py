"""Run the existing 2D detector once and save coarse points for ROI training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from fire_detector import FireDetector
from train_week6 import load_records


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=root / "fire-detection-from-cctv")
    parser.add_argument("--model", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--output", type=Path, default=root / "fire-model-data" / "coarse_manifest.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    detector = FireDetector(args.model, device=args.device, threshold=args.threshold)
    records, _ = load_records(args.labels, args.dataset_root)
    output = {}
    for index, record in enumerate(records, 1):
        image = Image.open(record.image_path).convert("RGB")
        result = detector.detect(image, warmup=index == 1)
        output[record.image_path] = {
            "point": None if result.pixel is None else [float(result.pixel[0]), float(result.pixel[1])],
            "confidence": float(result.confidence),
            "detected": bool(result.detected),
            "size": [int(image.width), int(image.height)],
        }
        if index % 50 == 0 or index == len(records):
            print(f"processed={index}/{len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved={args.output} records={len(output)}")


if __name__ == "__main__":
    main()
