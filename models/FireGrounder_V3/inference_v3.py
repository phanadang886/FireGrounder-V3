"""
Inference for FireGrounder V3.

Examples:
    python inference_v3.py --mode image --input fire.jpg --checkpoint best_v3_fpn.pth
    python inference_v3.py --mode video --input fire.mp4 --checkpoint best_v3_fpn.pth
    python inference_v3.py --mode webcam --checkpoint best_v3_fpn.pth

The existing inference.py remains the entry point for V1/V2 checkpoints.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from firegrounder_v3 import FireGrounderV3, load_checkpoint_v3, val_transform_v3


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


@torch.no_grad()
def predict(
    model: FireGrounderV3,
    image: Image.Image,
    device: str,
) -> tuple[float, float, float]:
    """Return (p_fire, x_norm, y_norm)."""

    tensor = val_transform_v3(image.convert("RGB")).unsqueeze(0).to(device)
    output = model(tensor)
    if isinstance(output, dict):
        output = output["pred"]
    p_fire, x_norm, y_norm = output.squeeze(0).detach().cpu().tolist()
    return float(p_fire), float(x_norm), float(y_norm)


def draw_fire_base(
    frame_bgr: np.ndarray,
    p_fire: float,
    x_norm: float,
    y_norm: float,
    threshold: float = 0.5,
    extra_text: str = "",
) -> np.ndarray:
    """Draw the predicted fire-base point when confidence crosses threshold."""

    height, width = frame_bgr.shape[:2]
    p_fire = float(np.clip(p_fire, 0.0, 1.0))
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    if p_fire < threshold:
        text = f"NO FIRE  (conf={1.0 - p_fire:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame_bgr,
            (6, 6),
            (16 + text_width, 12 + text_height + 10),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame_bgr,
            text,
            (10, 12 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    else:
        px = int(round(x_norm * max(width - 1, 1)))
        py = int(round(y_norm * max(height - 1, 1)))
        radius = max(12, min(width, height) // 35)

        cv2.circle(frame_bgr, (px, py), radius + 4, (0, 0, 180), 2)
        overlay = frame_bgr.copy()
        cv2.circle(overlay, (px, py), radius, (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0, frame_bgr)
        arm = radius + 10
        cv2.line(frame_bgr, (px - arm, py), (px + arm, py), (255, 255, 255), 2)
        cv2.line(frame_bgr, (px, py - arm), (px, py + arm), (255, 255, 255), 2)
        cv2.circle(frame_bgr, (px, py), 3, (0, 255, 255), -1)

        lines = [
            f"FIRE DETECTED ({p_fire * 100:.1f}%) | Base: ({x_norm:.3f}, {y_norm:.3f})",
            f"Pixel: ({px}, {py})",
        ]
        for line_index, text in enumerate(lines):
            baseline = 28 + line_index * 24
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame_bgr,
                (6, baseline - text_height - 4),
                (12 + text_width, baseline + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame_bgr,
                text,
                (9, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if line_index == 0 else (0, 255, 255),
                2,
            )

    if extra_text:
        (text_width, _), _ = cv2.getTextSize(
            extra_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.putText(
            frame_bgr,
            extra_text,
            (max(8, width - text_width - 8), 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 255, 100),
            2,
        )
    return frame_bgr


def run_image(args: argparse.Namespace, model: FireGrounderV3, device: str) -> None:
    image_path = Path(args.input)
    image = Image.open(image_path).convert("RGB")
    p_fire, x_norm, y_norm = predict(model, image, device)

    print("\n[RESULT]")
    print(
        f"  Confidence : {p_fire * 100:.2f}% "
        f"({'FIRE' if p_fire >= args.threshold else 'NO FIRE'})"
    )
    if p_fire >= args.threshold:
        print(f"  Normalized : x={x_norm:.4f}  y={y_norm:.4f}")
        print(f"  Pixel      : x={x_norm * image.width:.1f}  y={y_norm * image.height:.1f}")

    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    frame = draw_fire_base(frame, p_fire, x_norm, y_norm, args.threshold)
    output_path = Path(args.output) if args.output else image_path.with_name(
        f"{image_path.stem}_v3_pred.jpg"
    )
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write output image: {output_path}")
    print(f"  Saved      : {output_path}")

    if args.show:
        cv2.imshow("FireGrounder V3 - press any key to close", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(args: argparse.Namespace, model: FireGrounderV3, device: str) -> None:
    source = 0 if args.mode == "webcam" else args.input
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 1 else 25.0
    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open output video: {args.output}")

    print(f"[INFO] Stream {width}x{height} | threshold={args.threshold:.2f}")
    print("[INFO] Press Q to quit when display is enabled")
    frame_count = 0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            p_fire, x_norm, y_norm = predict(model, image, device)
            frame_count += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            current_fps = frame_count / elapsed
            frame = draw_fire_base(
                frame,
                p_fire,
                x_norm,
                y_norm,
                args.threshold,
                extra_text=f"FPS {current_fps:.1f}",
            )
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("FireGrounder V3 - Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"[INFO] Processed {frame_count} frames")
    if args.output:
        print(f"[INFO] Saved video: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FireGrounder V3 inference")
    parser.add_argument("--mode", choices=("image", "video", "webcam"), required=True)
    parser.add_argument("--input", default=None, help="Image/video path; omitted for webcam")
    parser.add_argument("--checkpoint", required=True, help="V3 checkpoint .pth")
    parser.add_argument("--output", default=None, help="Output image/video path")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV display window")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if args.mode in ("image", "video") and not args.input:
        parser.error(f"--input is required for mode={args.mode}")
    device = resolve_device(args.device)
    model = load_checkpoint_v3(args.checkpoint, device)

    if args.mode == "image":
        run_image(args, model, device)
    else:
        run_stream(args, model, device)


if __name__ == "__main__":
    main()
