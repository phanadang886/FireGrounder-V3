# -*- coding: utf-8 -*-
"""
pipeline.py — End-to-end: FireBase-YOLO26 → Depth → FireLocalizer3D
====================================================================

Nối 3 module lại thành một luồng chạy được, dùng cho cả ảnh, video và webcam.

────────────────────────────────────────────────────────────────────────────
Ba chế độ depth
────────────────────────────────────────────────────────────────────────────
  --depth relative   DepthART relative / Depth Anything  → depth_normalized, zone
                     (nhanh, chạy được CPU, KHÔNG có xyz_metric)

  --depth metric     DepthART metric indoor/outdoor      → xyz_metric (mét)
                     (cần --intrinsics; chậm hơn nhiều trên CPU vì TinyViM)

  --depth none       Bỏ qua depth, chỉ chạy detector (nhanh nhất, hữu ích để
                     kiểm tra YOLO)

────────────────────────────────────────────────────────────────────────────
Ví dụ
────────────────────────────────────────────────────────────────────────────
  # Ảnh đơn, relative depth
  python pipeline.py --image fire.jpg --out out/

  # Video + metric depth (đã calibrate)
  python pipeline.py --source video.mp4 --depth metric \
      --intrinsics intrinsics.json --domain indoor --out out/

  # Webcam, relative depth, có hiện depth map
  python pipeline.py --source 0 --show-depth --out out/

  # Bỏ qua YOLO (dùng keypoint cho sẵn) — tách lỗi khi debug depth
  python pipeline.py --image fire.jpg --keypoints 112,150 224,160

────────────────────────────────────────────────────────────────────────────
⚠️  Về FireBase-YOLO26 weights
────────────────────────────────────────────────────────────────────────────
Mặc định tìm tại:
  models/FireBase_YOLO26/training_runs/firebase_yolo26n/weights/best.pt

Nếu không có file này (git đã .gitignore *.pt), truyền --weights <path>.
Xem README.md phần "Cài đặt YOLO weights".
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from localizer_3d import CameraIntrinsics, FireLocalizer3D, FireLocation3D  # noqa: E402

REPO_ROOT = _HERE.parent.parent

DEFAULT_WEIGHTS = (
    REPO_ROOT / "models" / "FireBase_YOLO26" / "training_runs"
    / "firebase_yolo26n" / "weights" / "best.pt"
)

# Màu BGR
ZONE_COLOR = {"NEAR": (0, 0, 255), "MID": (0, 200, 255), "FAR": (0, 220, 0)}
FIRE_COLOR = (0, 140, 255)


# ─────────────────────────────────────────────────────────────────────────────
#  Detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Một vùng lửa phát hiện được."""
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    conf: float
    keypoint: Optional[tuple[float, float]]  # (kx, ky) — gốc lửa


class FireDetector:
    """
    Bọc FireBase-YOLO26. Import ultralytics lazily để phần depth/localizer
    vẫn chạy được khi chưa cài ultralytics.
    """

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        conf: float = 0.20,
        imgsz: int = 384,
        device: Optional[str] = None,
    ):
        self.weights = Path(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or ("cuda:0" if _cuda_ok() else "cpu")
        self._model = None
        self._pt_filter = None  # EMA filter

    # ── Load ──────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Không tìm thấy FireBase-YOLO26 weights:\n  {self.weights}\n\n"
                f"  File này bị .gitignore (*.pt) nên không có trong repo.\n"
                f"  → Tải/copy best.pt vào đúng chỗ trên, hoặc truyền --weights.\n"
                f"  → Hoặc chạy pipeline KHÔNG cần YOLO: --keypoints x,y  (xem --help)"
            )
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Chưa cài 'ultralytics' — cần cho FireBase-YOLO26.\n"
                "  → pip install ultralytics\n"
                "  → Hoặc bỏ qua YOLO: dùng --keypoints x,y để chạy phần depth."
            )
        self._model = YOLO(str(self.weights))
        self._ema_alpha = 0.65

    def reset_track(self) -> None:
        self._pt_filter = None

    # ── Detect ────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        results = self._model.predict(
            frame, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]

        out: list[Detection] = []
        boxes = getattr(results, "boxes", None)
        kps = getattr(results, "keypoints", None)
        if boxes is None or len(boxes) == 0:
            self.reset_track()
            return out

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kp_xy = None
        if kps is not None and len(kps.xy) > 0:
            kp_xy = kps.xy.cpu().numpy()

        for i, (bx, cf) in enumerate(zip(boxes_xyxy, confs)):
            kpt = None
            if kp_xy is not None and i < len(kp_xy) and len(kp_xy[i]) > 0:
                raw = (float(kp_xy[i][0][0]), float(kp_xy[i][0][1]))
                kpt = self._smooth(raw)
            out.append(Detection(
                bbox=(int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])),
                conf=float(cf),
                keypoint=kpt,
            ))
        return out

    def _smooth(self, pt: tuple[float, float]) -> tuple[float, float]:
        """
        EMA theo thời gian để chống rung điểm gốc lửa.
        Ngọn lửa run rất nhiều; điểm BASE nhấp nháy sẽ làm depth nhảy loạn.
        """
        if self._pt_filter is None:
            self._pt_filter = np.array(pt, dtype=float)
        else:
            a = self._ema_alpha
            self._pt_filter = a * np.array(pt, dtype=float) + (1 - a) * self._pt_filter
        return float(self._pt_filter[0]), float(self._pt_filter[1])


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    depth_mode: str = "relative"          # relative | metric | none
    prefer: str = "auto"
    domain: str = "indoor"                # metric domain
    encoder: str = "s"                    # metric encoder
    intrinsics: Optional[CameraIntrinsics] = None
    show_depth: bool = False
    near_thresh: float = 0.70
    far_thresh: float = 0.30
    extra: dict = field(default_factory=dict)


@dataclass
class FrameResult:
    """Kết quả xử lý 1 frame."""
    detections: list[Detection]
    locations: list[FireLocation3D]
    depth_map: Optional[np.ndarray]
    timings: dict           # ms cho từng giai đoạn
    annotated: Optional[np.ndarray] = None


class FirePipeline3D:
    """
    Pipeline: detect → depth → localize → annotate.

    Cố ý tách `DepthBackend` (relative) và `MetricDepthBackend` (mét) vì quy ước
    chiều depth NGƯỢC NHAU — xem docstring của metric_depth.py.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        detector: Optional[FireDetector] = None,
    ):
        self.cfg = cfg
        self.detector = detector
        self.localizer = FireLocalizer3D(
            near_thresh=cfg.near_thresh, far_thresh=cfg.far_thresh
        )
        self._depth_backend = None
        self._metric_backend = None
        self.depth_model_name = "disabled"

    # ── Lazy init ─────────────────────────────────────────────────────
    @property
    def depth_backend(self):
        if self._depth_backend is None and self.cfg.depth_mode == "relative":
            from depth_model import DepthBackend
            self._depth_backend = DepthBackend(
                prefer=self.cfg.prefer,
                depthart_encoder=self.cfg.extra.get("depthart_encoder"),
            )
            self._depth_backend.load()
            self.depth_model_name = self._depth_backend.model_name
        return self._depth_backend

    @property
    def metric_backend(self):
        if self._metric_backend is None and self.cfg.depth_mode == "metric":
            if self.cfg.intrinsics is None:
                raise ValueError(
                    "--depth metric yêu cầu --intrinsics <file.json>.\n"
                    "  → Tạo bằng: python calibrate_camera.py --images <bàn cờ> "
                    "--output intrinsics.json"
                )
            from metric_depth import MetricDepthBackend
            self._metric_backend = MetricDepthBackend(
                intrinsics=self.cfg.intrinsics,
                domain=self.cfg.domain,
                encoder=self.cfg.encoder,
            )
            self._metric_backend.load()
            self.depth_model_name = self._metric_backend.model_name
        return self._metric_backend

    def warmup(self) -> None:
        """Load model sẵn để không tính vào thời gian của frame đầu tiên."""
        if self.cfg.depth_mode == "relative":
            _ = self.depth_backend
        elif self.cfg.depth_mode == "metric":
            _ = self.metric_backend
        if self.detector is not None:
            self.detector.load()

    # ── Core ──────────────────────────────────────────────────────────
    def process(
        self,
        frame: np.ndarray,
        fixed_keypoints: Optional[list[tuple[int, int]]] = None,
    ) -> FrameResult:
        """
        Xử lý 1 frame BGR.

        Parameters
        ----------
        fixed_keypoints : nếu có, BỎ QUA detector và dùng các keypoint này.
                          Dùng để cô lập lỗi khi debug phần depth.
        """
        H, W = frame.shape[:2]
        timings: dict[str, float] = {}

        # 1) Detect
        t0 = time.perf_counter()
        if fixed_keypoints is not None:
            dets: list[Detection] = []
            for kx, ky in fixed_keypoints:
                dets.append(Detection(
                    bbox=(int(kx) - 1, int(ky) - 1, int(kx) + 1, int(ky) + 1),
                    conf=1.0, keypoint=(float(kx), float(ky)),
                ))
        elif self.detector is not None:
            dets = self.detector.detect(frame)
        else:
            dets = []
        timings["detect"] = (time.perf_counter() - t0) * 1000

        # 2) Depth
        t0 = time.perf_counter()
        depth_map = None
        is_metric = self.cfg.depth_mode == "metric"
        if self.cfg.depth_mode == "relative":
            depth_map = self.depth_backend.infer_bgr(frame)
        elif is_metric:
            depth_map = self.metric_backend.infer_bgr(frame)
        timings["depth"] = (time.perf_counter() - t0) * 1000

        # 3) Localize
        t0 = time.perf_counter()
        locations: list[FireLocation3D] = []
        if depth_map is not None:
            for d in dets:
                if d.keypoint is None:
                    continue
                kx, ky = d.keypoint
                try:
                    loc = self.localizer.localize(
                        depth_map, kx=int(kx), ky=int(ky),
                        img_w=W, img_h=H,
                        intrinsics=self.cfg.intrinsics,
                        model_name=self.depth_model_name,
                        metric_depth=is_metric,
                    )
                except ValueError as e:
                    # depth không hợp lệ tại điểm này (ví dụ 0 m) → bỏ qua,
                    # không làm sập cả frame.
                    print(f"  [warn] localize that bai: {e}")
                    continue
                locations.append(loc)
        timings["localize"] = (time.perf_counter() - t0) * 1000

        # 4) Annotate
        t0 = time.perf_counter()
        annotated = self.annotate(frame, dets, locations, depth_map)
        timings["annotate"] = (time.perf_counter() - t0) * 1000

        return FrameResult(
            detections=dets, locations=locations, depth_map=depth_map,
            timings=timings, annotated=annotated,
        )

    # ── Visualization ─────────────────────────────────────────────────
    def colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Depth → BGR để hiển thị. Sáng = gần, tối = xa."""
        if self.cfg.depth_mode == "metric":
            return self.metric_backend.colorize(depth)
        return self.depth_backend.colorize(depth)

    def annotate(
        self,
        frame: np.ndarray,
        dets: list[Detection],
        locations: list[FireLocation3D],
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]

        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), FIRE_COLOR, 2)
            cv2.putText(out, f"FIRE {d.conf * 100:.0f}%", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, FIRE_COLOR, 2, cv2.LINE_AA)

            loc = locations[i] if i < len(locations) else None
            if d.keypoint is None:
                continue
            kx, ky = int(d.keypoint[0]), int(d.keypoint[1])

            # Điểm BASE
            cv2.circle(out, (kx, ky), 7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(out, (kx, ky), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (kx, ky), 2, (255, 255, 255), -1, cv2.LINE_AA)

            if loc is None:
                continue
            col = ZONE_COLOR.get(loc.zone, (200, 200, 200))
            if loc.xyz_metric:
                label = (f"#{i + 1} {loc.zone}  Z={loc.depth_m:.2f}m")
                detail = (f"XYZ=({loc.xyz_metric[0]:+.2f},"
                          f"{loc.xyz_metric[1]:+.2f},{loc.xyz_metric[2]:.2f})m")
            else:
                label = f"#{i + 1} {loc.zone}  d_n={loc.depth_normalized:.2f}"
                detail = f"Z_n={loc.xyz_normalized[2]:.2f}"
            _put_tag(out, label, (kx, ky), col)
            _put_tag(out, detail, (kx, ky + 26), col)

        # Panel depth
        if self.cfg.show_depth and depth_map is not None:
            dv = self.colorize_depth(depth_map)
            dv = cv2.resize(dv, (W, H), interpolation=cv2.INTER_NEAREST)
            out = np.concatenate([out, dv], axis=1)

        # HUD
        has_fire = len(dets) > 0
        hud_col = (0, 0, 255) if has_fire else (0, 220, 0)
        status = "ALERT: FIRE DETECTED" if has_fire else "NORMAL: NO FIRE"
        mode = f"depth={self.cfg.depth_mode} {self.depth_model_name}"
        _put_hud(out, 12, 26, mode, (255, 255, 255))
        _put_hud(out, 12, 54, status, hud_col)
        return out

    def summary(self, r: FrameResult) -> dict:
        """Tóm tắt JSON-friendly cho 1 frame."""
        return {
            "depth_model": self.depth_model_name,
            "depth_mode": self.cfg.depth_mode,
            "timings_ms": {k: round(v, 2) for k, v in r.timings.items()},
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "conf": round(d.conf, 3),
                    "keypoint": (
                        [round(d.keypoint[0], 1), round(d.keypoint[1], 1)]
                        if d.keypoint else None
                    ),
                }
                for d in r.detections
            ],
            "locations": [loc.to_dict() for loc in r.locations],
        }


def _put_tag(img: np.ndarray, text: str, anchor: tuple[int, int], color) -> None:
    """Nhãn nhỏ có nền, đặt lệch khỏi điểm neo."""
    x, y = anchor
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tx, ty = min(max(4, x + 10), img.shape[1] - tw - 6), max(14, y - 8)
    cv2.rectangle(img, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _put_hud(img: np.ndarray, x: int, y: int, text: str, color) -> None:
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Sources
# ─────────────────────────────────────────────────────────────────────────────

def read_image(path: str | Path) -> np.ndarray:
    """Đọc ảnh, hỗ trợ đường dẫn Unicode trên Windows."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {p}")
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh (có thể sai định dạng): {p}")
    return img


def open_source(source: str):
    """Mở nguồn video. '0'/'1' = camera index, còn lại = đường dẫn."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    return cap


def parse_keypoints(spec: Optional[str]) -> Optional[list[tuple[int, int]]]:
    """--keypoints '112,150 224,160' → [(112,150), (224,160)]"""
    if not spec:
        return None
    pts = []
    for tok in spec.replace(";", " ").split():
        if "," not in tok:
            raise ValueError(f"Keypoint sai định dạng: {tok!r} (cần 'x,y')")
        x, y = tok.split(",", 1)
        pts.append((int(x), int(y)))
    return pts or None


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    p = argparse.ArgumentParser(
        description="FireBase-YOLO26 + Depth + FireLocalizer3D (end-to-end)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_argument_group("Nguồn vào (chọn 1)")
    src.add_argument("--image", help="đường dẫn 1 ảnh")
    src.add_argument("--source", default=None,
                     help="camera index (0, 1) hoặc file video")

    d = p.add_argument_group("Detector (FireBase-YOLO26)")
    d.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    d.add_argument("--conf", type=float, default=0.20)
    d.add_argument("--imgsz", type=int, default=384)
    d.add_argument("--device", default=None, help="'cpu' / '0' (cuda). Mặc định tự chọn")
    d.add_argument("--no-detector", action="store_true",
                   help="bỏ qua YOLO (dùng --keypoints)")
    d.add_argument("--keypoints", default=None,
                   help="keypoint cố định 'x,y x,y' — bỏ qua detector")

    dp = p.add_argument_group("Depth")
    dp.add_argument("--depth", choices=("relative", "metric", "none"),
                    default="relative")
    dp.add_argument("--prefer", default="auto",
                    choices=("auto", "depthart", "dav2", "dav1"),
                    help="backend relative (mặc định auto)")
    dp.add_argument("--domain", choices=("indoor", "outdoor"), default="indoor",
                    help="domain cho metric depth")
    dp.add_argument("--encoder", choices=("s", "b", "l"), default="s",
                    help="encoder cho metric depth")
    dp.add_argument("--intrinsics", default=None,
                    help="file JSON từ calibrate_camera.py (bắt buộc cho --depth metric)")
    dp.add_argument("--fov", type=float, default=90.0,
                    help="FOV ngang để ƯỚC LƯỢNG intrinsics nếu không có file")
    dp.add_argument("--show-depth", action="store_true", help="ghép depth map vào khung")

    o = p.add_argument_group("Xuất")
    o.add_argument("--out", default=None, help="thư mục lưu kết quả")
    o.add_argument("--save-json", action="store_true", help="lưu kết quả .json")
    o.add_argument("--max-frames", type=int, default=None,
                   help="dừng sau N frame (mặc định: toàn bộ video)")

    w = p.add_argument_group("Hiển thị")
    w.add_argument("--display", action="store_true",
                   help="mở cửa sổ xem trực tiếp (cần màn hình)")
    w.add_argument("--near-thresh", type=float, default=0.70)
    w.add_argument("--far-thresh", type=float, default=0.30)
    return p.parse_args()


def build_pipeline(args: argparse.Namespace) -> FirePipeline3D:
    intr = None
    if args.intrinsics:
        intr = CameraIntrinsics.load(args.intrinsics)
    elif args.depth == "metric":
        raise SystemExit(
            "LỖI: --depth metric cần --intrinsics <file.json>.\n"
            "  Nếu chưa calibrate, dùng ước lượng FOV bằng cách bỏ --depth metric,\n"
            "  hoặc tạo file: python calibrate_camera.py --from-fov 90 \\\n"
            "      --img-size 640 480 --output intrinsics.json"
        )

    detector = None
    if not args.no_detector and not args.keypoints:
        detector = FireDetector(
            weights=args.weights, conf=args.conf,
            imgsz=args.imgsz, device=args.device,
        )

    cfg = PipelineConfig(
        depth_mode=args.depth,
        prefer=args.prefer,
        domain=args.domain,
        encoder=args.encoder,
        intrinsics=intr,
        show_depth=args.show_depth,
        near_thresh=args.near_thresh,
        far_thresh=args.far_thresh,
    )
    return FirePipeline3D(cfg, detector)


def main() -> int:
    args = parse_args()

    if not args.image and not args.source:
        print("LỖI: cần --image hoặc --source")
        return 1

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    fixed_kpts = parse_keypoints(args.keypoints)
    if fixed_kpts:
        print(f"[info] Bo qua detector, dung {len(fixed_kpts)} keypoint: {fixed_kpts}")

    pipe = build_pipeline(args)

    print("[info] Đang nạp model...")
    t0 = time.perf_counter()
    try:
        pipe.warmup()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
        print(f"\nLỖI khi nạp model:\n{e}")
        return 1
    print(f"[ok] Nạp xong sau {(time.perf_counter() - t0):.1f}s")
    if pipe.depth_model_name != "disabled":
        print(f"[ok] Depth: {pipe.depth_model_name}")
    if args.depth == "metric" and pipe.cfg.intrinsics is not None:
        i = pipe.cfg.intrinsics
        warn = "" if i.is_calibrated else "  <-- UOC LUONG, chua calibrate that"
        print(f"[ok] Intrinsics: fx={i.fx:.1f} fy={i.fy:.1f} "
              f"cx={i.cx:.1f} cy={i.cy:.1f} @ {i.width}x{i.height} "
              f"({i.source}){warn}")

    all_records: list[dict] = []

    # ── Ảnh đơn ──────────────────────────────────────────────────────
    if args.image:
        frame = read_image(args.image)
        res = pipe.process(frame, fixed_keypoints=fixed_kpts)
        _print_frame_report(res, 0)

        if out_dir:
            img_out = out_dir / f"pred_{Path(args.image).stem}.jpg"
            cv2.imencode(str(img_out), res.annotated)[1].tofile(str(img_out))
            print(f"\n[out] Ảnh: {img_out}")
        if args.save_json:
            rec = pipe.summary(res)
            rec["source"] = args.image
            all_records.append(rec)
        if args.display:
            _show_forever(res.annotated)

    # ── Video / webcam ────────────────────────────────────────────────
    elif args.source:
        cap = open_source(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        is_cam = args.source.isdigit()
        print(f"[info] Nguồn: {W}x{H} @ {fps:.1f} FPS"
              + (f" | tổng {total} frame" if total else ""))

        writer = None
        if out_dir:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_vid = out_dir / (
                "pred_cam.mp4" if is_cam
                else f"pred_{Path(args.source).stem}.mp4"
            )
            writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (W, H))
            print(f"[out] Video: {out_vid}")

        n = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if args.max_frames and n >= args.max_frames:
                    break
                res = pipe.process(frame, fixed_keypoints=fixed_kpts)
                if n < 3 or n % 30 == 0:
                    _print_frame_report(res, n)
                if args.save_json:
                    all_records.append(pipe.summary(res))
                if writer is not None:
                    writer.write(res.annotated)
                if args.display:
                    cv2.imshow("FireGrounder 3D", res.annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                n += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if args.display:
                cv2.destroyAllWindows()
        print(f"\n[ok] Xử lý {n} frame")

    if args.save_json and out_dir:
        jf = out_dir / "results.json"
        jf.write_text(
            json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[out] JSON: {jf}")

    return 0


def _print_frame_report(res: FrameResult, idx: int) -> None:
    t = res.timings
    total = sum(t.values())
    parts = " ".join(f"{k}={v:.0f}" for k, v in t.items())
    print(f"\n[frame {idx}] {parts} | total={total:.0f}ms")
    if not res.detections:
        print("  (không phát hiện lửa)")
        return
    for i, (d, loc) in enumerate(zip(res.detections, res.locations)):
        kp = d.keypoint
        if loc.xyz_metric:
            x, y, z = loc.xyz_metric
            print(f"  #{i + 1} conf={d.conf:.2f} kp=({kp[0]:.0f},{kp[1]:.0f}) "
                  f"zone={loc.zone} XYZ=({x:+.2f},{y:+.2f},{z:.2f})m")
        else:
            print(f"  #{i + 1} conf={d.conf:.2f} kp=({kp[0]:.0f},{kp[1]:.0f}) "
                  f"zone={loc.zone} d_norm={loc.depth_normalized:.3f}")


def _show_forever(img: np.ndarray) -> None:
    cv2.imshow("FireGrounder 3D", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())