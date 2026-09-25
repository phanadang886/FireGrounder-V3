"""
Real-time Video Inference with FireBase-YOLO26.
Features:
- Single-Keypoint Fire Base Ground Contact Point tracking
- Temporal EMA (Exponential Moving Average) Smoothing to stabilize flame root jitter
- Support for webcam (device 0), RTSP stream, or video file (.mp4, .avi)
- Real-time FPS & Latency HUD overlay
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class TemporalPointFilter:
    """Exponential Moving Average (EMA) filter to prevent flame root jitter across frames."""
    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.smooth_pt = None

    def update(self, pt: tuple[float, float]) -> tuple[float, float]:
        if self.smooth_pt is None:
            self.smooth_pt = np.array(pt, dtype=float)
        else:
            self.smooth_pt = self.alpha * np.array(pt, dtype=float) + (1.0 - self.alpha) * self.smooth_pt
        return float(self.smooth_pt[0]), float(self.smooth_pt[1])

    def reset(self):
        self.smooth_pt = None


def parse_args():
    parser = argparse.ArgumentParser(description="FireBase-YOLO26 Real-Time Video Tracker")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or path to video file")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to FireBase-YOLO26 checkpoint weights")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=384, help="Inference resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--smooth", action="store_true", default=True, help="Enable temporal EMA smoothing")
    parser.add_argument("--output", type=str, default=None, help="Optional output video save path (.mp4)")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # Find weights
    candidate_weights = [
        Path(args.weights),
        script_dir / args.weights,
        script_dir / "best.pt",
        script_dir / "training_runs" / "firebase_yolo26n" / "weights" / "best.pt",
        script_dir.parent / "YOLO26_Pose" / "best.pt",
    ]
    ckpt_path = next((p for p in candidate_weights if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"❌ Không tìm thấy weights: {args.weights}!")

    print(f"🚀 Đang khởi tạo FireBase-YOLO26 từ: {ckpt_path}")
    model = YOLO(str(ckpt_path))

    # Initialize video capture
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Không thể mở nguồn video: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"📹 Nguồn video: {w}x{h} @ {fps_in:.1f} FPS")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

    pt_filter = TemporalPointFilter(alpha=0.65) if args.smooth else None
    frame_times = []

    print("\n🟢 Đang chạy... Nhấn phím 'q' trên cửa sổ để dừng lại.\n")

    try:
        while cap.isOpened():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            # Forward inference
            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )[0]

            has_fire = False
            if len(results.boxes) > 0 and results.keypoints is not None and len(results.keypoints.xy) > 0:
                for idx, (box, kpt) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), results.keypoints.xy.cpu().numpy())):
                    has_fire = True
                    bx1, by1, bx2, by2 = map(int, box[:4])
                    conf_score = float(results.boxes.conf[idx].item())

                    # Draw Fire Bounding Box (Amber Orange)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
                    cv2.putText(
                        frame,
                        f"FIRE {conf_score*100:.0f}%",
                        (bx1, max(20, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 140, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Fire Base Ground Contact Point
                    if len(kpt) > 0:
                        raw_kpt = (kpt[0][0], kpt[0][1])
                        kx, ky = pt_filter.update(raw_kpt) if pt_filter else raw_kpt
                        kx, ky = int(kx), int(ky)

                        # Draw Bold Fire Base Checkpoint: Black outline + Vivid Crimson Red + White core
                        cv2.circle(frame, (kx, ky), 7, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.circle(frame, (kx, ky), 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (kx, ky), 1, (255, 255, 255), -1, cv2.LINE_AA)

                        # Label with shadow
                        label = f"BASE ({kx},{ky})"
                        tx, ty = min(w - 120, kx + 10), max(20, ky - 6)
                        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                if pt_filter:
                    pt_filter.reset()

            # Latency and FPS Calculation
            dt = (time.perf_counter() - t0) * 1000
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_dt = np.mean(frame_times)
            cur_fps = 1000.0 / max(avg_dt, 1e-4)

            # HUD Display
            hud_color = (0, 0, 255) if has_fire else (0, 255, 0)
            hud_status = "ALERT: FIRE DETECTED" if has_fire else "NORMAL: NO FIRE"
            cv2.putText(frame, f"FireBase-YOLO26 | {cur_fps:.1f} FPS ({avg_dt:.1f} ms)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"FireBase-YOLO26 | {cur_fps:.1f} FPS ({avg_dt:.1f} ms)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, hud_status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, hud_status, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2, cv2.LINE_AA)

            if writer:
                writer.write(frame)

            cv2.imshow("FireBase-YOLO26 Real-Time Monitor (Q to Quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("🛑 Đã dừng theo dõi video.")


if __name__ == "__main__":
    main()
