"""
`utils.py` — Hàm dùng chung cho simulator đánh giá depth + toán 2D→3D.

**Đây là Phase 1 của `SIM_CONTRACT.md` mục 11.** Tất cả script khác (`download_dataset.py`,
`sample_points.py`, `run_eval.py`, `eval_metrics.py`) import từ đây — **không viết lại**
các hàm này ở nơi khác.

⚠️ 3 ràng buộc bắt buộc (xem `SIM_CONTRACT.md` mục 3 và mục 11 "Trước khi code"):
1. **Cấm ghi vào ổ C.** Mọi dữ liệu/kết quả nằm dưới `D:\\replica_sim\\`.
   Kể cả file tạm trong self-test cũng phải tạo trên ổ D, KHÔNG dùng
   `tempfile.mkdtemp()` mặc định ( nó ghi vào `%TEMP%` trên ổ C ).
2. **Không import torch / model nặng ở cấp module.** `numpy` thì được (mọi hàm toán
   đều cần), nhưng `cv2`, `torch`, `depth_model`, `metric_depth` chỉ import **bên
   trong** hàm khi thật sự cần. Import module này phải tốn < 0.5 s.
3. **Quy ước trục 3D phải khớp `localizer_3d.py`:** X phải, Y **xuống dưới**, Z ra xa camera.

Xem thêm: `SIM_CONTRACT.md` (mục 5 format dữ liệu, mục 8 danh sách hàm),
`README.md` (cách chạy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

# ── Giới hạn thread OpenBLAS TRƯỚC khi import numpy ────────────────────
# Máy lab có 15.7 GB nhưng thường chỉ còn ~1 GB trống khi mở Chrome/Docker/IDE.
# OpenBLAS cấp buffer theo số luồng (8 ở máy này) và khi thiếu RAM nó **abort cứng
# tiến trình** với "OpenBLAS error: Memory allocation still failed after 10 retries"
# — abort ở tầng C nên `try/except` của Python không bắt được. Phải giới hạn
# TRƯỚC khi numpy nạp thư viện.
#
# ⚠️ Chỉ đặt OPENBLAS (numpy dùng). KHÔNG đặt OMP/MKL_NUM_THREADS — hai cái đó
# điều khiển đa luồng của torch, đặt = 1 sẽ làm Phase 4 chậm nhiều lần.
if not os.environ.get("OPENBLAS_NUM_THREADS"):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

try:
    import numpy as np
except Exception:  # noqa: BLE001 — bắt cả MemoryError lẫn lỗi native
    # Đến đây vẫn fail → máy thật sự không đủ RAM cho cả 1 luồng.
    raise RuntimeError(
        "Không import được numpy: RAM trống quá ít.\n"
        "  → Đóng Chrome / Docker / IDE rồi chạy lại (cần > 1 GB trống).\n"
        "  → Kiểm tra: Get-CimInstance Win32_OperatingSystem | "
        "Select-Object FreePhysicalMemory"
    ) from None

__all__ = [
    "DEFAULT_DATA_ROOT",
    "MILLIMETER_PNG_DEPTH_SCALE",
    "REPLICA_PNG_DEPTH_SCALE",
    "REPLICA_CAM",
    "ScaleShift",
    "check_disk_space",
    "load_depth_map",
    "parse_intrinsics_txt",
    "parse_pose_line",
    "pixel_to_cam",
    "cam_to_world",
    "add_pose_noise",
    "align_scale_shift",
    "load_module_backend",
    "replica_intrinsics",
]

# ── Hằng số dataset ─────────────────────────────────────────────────────

#: Depth PNG 16-bit lưu theo **milimét** → `depth_m = raw / 1000`. (NYUv2, nhiều bộ khác)
MILLIMETER_PNG_DEPTH_SCALE = 1000.0

#: ⚠️ Replica (bản NICE-SLAM) lưu depth PNG 16-bit với hệ số **6553.5**, KHÔNG phải 1000.
#: Nguồn: `cvg/nice-slam` → `configs/Replica/replica.yaml` (`png_depth_scale`) và
#: `src/utils/datasets.py`: ``depth_data.astype(np.float32) / self.png_depth_scale``.
#: Dùng nhầm 1000 ⇒ **sai mọi số đo 6.55 lần**. Kiểm tra ngược: 65535 / 6553.5 = 10.0 m
#: đúng bằng max depth của Replica.
REPLICA_PNG_DEPTH_SCALE = 6553.5

#: Thông số camera Replica — cũng lấy từ `configs/Replica/replica.yaml`.
#: **Dataset KHÔNG chứa file intrinsics**, nên phải lấy từ đây.
REPLICA_CAM = {
    "height": 680,
    "width": 1200,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 599.5,
    "cy": 339.5,
    "png_depth_scale": REPLICA_PNG_DEPTH_SCALE,
    # traj.txt lưu pose theo quy ước OpenGL (Y lên, Z ngược) → cần đảo cột 1,2
    # traj.txt ĐÃ Ở hệ OpenCV — KHÔNG đảo cột 1,2.
    # Xac minh bang mesh that (2026-10-09): khoang cach median tu diem back-project toi
    # be mat mesh la **0.005 m** khi KHONG dao, vs 0.666 m khi dao cot 1,2 va
    # 0.894 m khi dao ca 3 cot — ca 8 scene deu vay.
    # Ly do de nham: cvg/nice-slam co `c2w[:3,1] *= -1; c2w[:3,2] *= -1` trong
    # `src/utils/datasets.py`, nhung do la de chuyen he cho RENDERER OpenGL cua ho,
    # KHONG phai de doc file `traj.txt` da luu. Da sua sau khi do thuc te.
    "opengl_to_opencv": False,
}

# ── Hằng số ─────────────────────────────────────────────────────────────

#: Gốc dữ liệu của simulator. Override bằng `--data-root` hoặc biến môi trường
#: `SIM_DATA_ROOT`. KHÔNG đổi thành `C:\` — xem `SIM_CONTRACT.md` mục 3.
DEFAULT_DATA_ROOT = Path(os.environ.get("SIM_DATA_ROOT", r"D:\replica_sim"))


def replica_intrinsics(data_dir: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Intrinsics + hệ số depth của Replica, dạng dict.

    **Ưu tiên đọc `cam_params.json`** do chính dataset đóng gói
    (`Replica/cam_params.json`), là nguồn chân lý:

    ```json
    {"camera": {"w":1200, "h":680, "fx":600.0, "fy":600.0,
                "cx":599.5, "cy":339.5, "scale":6553.5}}
    ```

    Chỉ khi không tìm thấy file mới dùng hằng số `REPLICA_CAM`. Lý do: hard-code
    giá trị ngoài dataset là nguồn sai âm thầm — sai hệ số depth sẽ ra số sai
    nhiều lần mà không báo lỗi.

    ⚠️ **Không suy `width = 2·cx`**: `cx = 599.5` suy ra 1199, nhưng ảnh rộng **1200**
    (principal point lệch nửa pixel). Dùng sai sẽ lệch 1 px ở mép phải.

    Parameters
    ----------
    data_dir : str | Path | None
        Thư mục gốc chứa Replica (ví dụ `D:\\replica_sim\\data\\replica_pp`).
        Nếu `None` thì chỉ dùng hằng số, không đọc file.

    Returns
    -------
    dict
        ``{"fx","fy","cx","cy","width","height","source","png_depth_scale"}``

    Examples
    --------
    >>> replica_intrinsics()["png_depth_scale"]
    6553.5
    """
    if data_dir is not None:
        for cand in _find_cam_params(Path(data_dir)):
            raw = json.loads(cand.read_text(encoding="utf-8"))
            cam = raw.get("camera", raw)
            try:
                out = {
                    "fx": float(cam["fx"]), "fy": float(cam["fy"]),
                    "cx": float(cam["cx"]), "cy": float(cam["cy"]),
                    "width": int(cam["w"]), "height": int(cam["h"]),
                    "png_depth_scale": float(cam["scale"]),
                    "source": str(cand),
                }
            except KeyError as e:
                raise ValueError(
                    f"{cand} thiếu trường {e}. Cần: w, h, fx, fy, cx, cy, scale."
                ) from None
            if out["fx"] <= 0 or out["fy"] <= 0:
                raise ValueError(f"Tiêu cự không hợp lệ trong {cand}: {out}")
            return out

    return {
        "fx": REPLICA_CAM["fx"],
        "fy": REPLICA_CAM["fy"],
        "cx": REPLICA_CAM["cx"],
        "cy": REPLICA_CAM["cy"],
        "width": REPLICA_CAM["width"],
        "height": REPLICA_CAM["height"],
        "source": "replica_cam_params",
        "png_depth_scale": REPLICA_CAM["png_depth_scale"],
    }


def _find_cam_params(data_dir: Path) -> List[Path]:
    """Tìm `cam_params.json` trong cây thư mục (giới hạn 2 cấp để khỏi quét sâu)."""
    cands = [data_dir / "cam_params.json",
             data_dir / "Replica" / "cam_params.json"]
    for scene in sorted(data_dir.glob("*")):
        if scene.is_dir():
            cands.append(scene / "cam_params.json")
    return [c for c in cands if c.is_file()][:3]

_GIB = 1024.0 ** 3


class ScaleShift(NamedTuple):
    """Kết quả căn scale+shift cho depth relative.

    Áp dụng: ``aligned = scale * pred + shift``.

    Đây là hệ số **ít nhất bình phương** trên các pixel có GT hợp lệ — dùng để
    chuyển depth *affine-invariant* (DepthART relative, Depth Anything) về thang mét
    của depth GT. Model **metric** thì KHÔNG được căn (đã có đơn vị mét rồi).
    """

    scale: float
    shift: float


# ── 1. Ổ đĩa ────────────────────────────────────────────────────────────


def _free_gb(path: Union[str, Path]) -> float:
    """Số GB trống của ổ chứa `path`. Raise OSError nếu ổ không tồn tại."""
    import shutil

    p = Path(path)
    # Leo lên thư mục cha tồn tại gần nhất (thư mục có thể chưa được tạo)
    probe = p if p.exists() else p.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / _GIB


def check_disk_space(
    min_c_gb: float = 2.0,
    min_d_gb: float = 10.0,
    data_root: Union[str, Path] = DEFAULT_DATA_ROOT,
) -> Dict[str, float]:
    """Kiểm tra ổ đĩa trước khi làm I/O nặng (tải dataset, giải nén).

    Gọi hàm này ở đầu mọi script, trước bước tải/giải nén/ghi kết quả.

    Parameters
    ----------
    min_c_gb : float
        Ngưỡng GB trống tối thiểu của ổ C. Mặc định 2.0.
    min_d_gb : float
        Ngưỡng GB trống tối thiểu của ổ chứa `data_root`. Mặc định 10.0.
    data_root : str | Path
        Gốc dữ liệu simulator (mặc định `D:\\replica_sim`).

    Returns
    -------
    dict
        ``{"c_free_gb": float, "data_free_gb": float}`` — giá trị GB trống.

    Raises
    ------
    RuntimeError
        Nếu ổ C trống < `min_c_gb`, hoặc ổ chứa `data_root` trống < `min_d_gb`.

    Examples
    --------
    >>> check_disk_space()  # doctest: +SKIP
    {'c_free_gb': 0.8, 'data_free_gb': 33.1}
    """
    data_root = Path(data_root)
    data_free = _free_gb(data_root)

    info: Dict[str, float] = {"data_free_gb": round(data_free, 2)}
    c_free: Optional[float] = None
    if Path("C:/").exists() or Path("C:\\").exists():
        c_free = _free_gb("C:\\")
        info["c_free_gb"] = round(c_free, 2)

    problems = []
    if c_free is not None and c_free < min_c_gb:
        problems.append(
            f"Ổ C chỉ còn {c_free:.1f} GB (cần tối thiểu {min_c_gb:.0f} GB).\n"
            f"    → Dọn ổ C (Start → Disk Cleanup), hoặc giải phóng bằng cách xoá file tạm.\n"
            f"    → TUYỆT ĐỐI KHÔNG để dataset/simulation ghi vào ổ C."
        )
    if data_free < min_d_gb:
        problems.append(
            f"Ổ chứa {data_root} chỉ còn {data_free:.1f} GB "
            f"(cần tối thiểu {min_d_gb:.0f} GB).\n"
            f"    → Xoá bớt thứ khác trên ổ đó, hoặc đổi --data-root sang ổ trống nhiều hơn."
        )
    if problems:
        raise RuntimeError(
            "KHÔNG ĐỦ CHỖ TRÊN ĐĨA:\n  - " + "\n  - ".join(problems)
        )
    return info


# ── 2. Depth map ────────────────────────────────────────────────────────


def load_depth_map(
    path: Union[str, Path],
    scale: Optional[float] = None,
) -> np.ndarray:
    """Đọc depth map và chuẩn hoá về **mét (float32)**, pixel không hợp lệ → `NaN`.

    | Định dạng            | Quy ước         | `scale` cần?              |
    |----------------------|-----------------|---------------------------|
    | `.npy`               | **mét**         | không (bỏ qua)            |
    | PNG/TIF `float32/64` | **mét**         | không (bỏ qua)            |
    | PNG/TIF `uint16`     | **số nguyên**  | **BẮT BUỘC** — xem bảng dưới |
    | PNG/TIF `uint8`      | không hợp lệ   | raise — 8 bit không đủ chính xác |

    **Hệ số `scale` của từng dataset (sai là ra số ảo):**

    | Dataset                          | `scale`   | Ghi chú |
    |----------------------------------|-----------|---------|
    | **Replica (NICE-SLAM)**          | **6553.5** | ⚠️ KHÔNG phải 1000 |
    | NYUv2, và PNG milimét thông thường| **1000.0** | `MILLIMETER_PNG_DEPTH_SCALE` |

    Vì sao **không tự đoán**: đọc Replica bằng 1000 sẽ cho depth sai **6.55 lần**, mọi
    metric sau đó đều vô nghĩa mà không báo lỗi. Nên hàm này **bắt buộc** khai báo
    `scale` khi gặp depth nguyên, thay vì đoán sai cho bạn.

    Lý do phải chuẩn hoá về mét: toàn bộ simulator so sánh theo đơn vị mét, còn Replica
    lưu theo hệ số 6553.5 còn NYUv2 theo milimét.

    Parameters
    ----------
    path : str | Path
        Đường dẫn file depth (`.npy`, `.png`, `.tif`, `.tiff`).
    scale : float | None
        Hệ số chia cho depth nguyên. Bắt buộc với PNG/TIF nguyên; bỏ qua với `.npy`
        và PNG float. Dùng `REPLICA_PNG_DEPTH_SCALE` hoặc `MILLIMETER_PNG_DEPTH_SCALE`.

    Returns
    -------
    np.ndarray
        Mảng `float32` đơn vị mét, shape `(H, W)`. Pixel không đo được = `NaN`.

    Raises
    ------
    FileNotFoundError, ValueError

    Examples
    --------
    >>> d = load_depth_map("D:/.../results/depth000000.png",
    ...                    scale=REPLICA_PNG_DEPTH_SCALE)
    >>> d.shape, d.dtype
    ((680, 1200), dtype('float32'))
    >>> d.max() <= 10.01      # Replica cắt ở 10 m
    True
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy depth map: {p}")

    if p.suffix.lower() == ".npy":
        raw = np.load(p)
    else:
        import cv2  # lazy: chỉ cần khi đọc ảnh

        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(
                f"Không đọc được ảnh depth: {p}\n"
                f"    → File có thể hỏng, hoặc thiếu codec (thử mở bằng PIL để xác nhận)."
            )

    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[..., 0]  # depth 1 kênh; bỏ kênh alpha nếu có

    kind = raw.dtype.kind
    if kind == "u":
        if raw.dtype.itemsize == 1:  # uint8
            raise ValueError(
                f"Depth map 8-bit không dùng được: {p}\n"
                f"    → uint8 chỉ chứa 0-255, không đủ độ chính xác để tính sai số mét.\n"
                f"    → Dataset này có lẽ lưu depth dạng PNG 16-bit hoặc NPY."
            )
        if scale is None:
            raise ValueError(
                f"Depth map nguyên ({p}) BẮT BUỘC phải khai báo `scale`.\n"
                f"    → Replica (NICE-SLAM): scale=REPLICA_PNG_DEPTH_SCALE "
                f"({REPLICA_PNG_DEPTH_SCALE}) — KHÔNG phải 1000.\n"
                f"    → PNG milimét (NYUv2, v.v.): "
                f"scale=MILLIMETER_PNG_DEPTH_SCALE ({MILLIMETER_PNG_DEPTH_SCALE}).\n"
                f"    → Không đoán giúp: đọc sai hệ số cho số sai 6.55 lần mà không báo lỗi."
            )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale phải là số hữu hạn > 0 — nhận {scale!r}")
        out = raw.astype(np.float32) / float(scale)
    elif kind == "f":
        out = raw.astype(np.float32)  # đã là mét
    elif kind == "i":
        if scale is None:
            raise ValueError(
                f"Depth map số nguyên có dấu ({p}) BẮT BUỘC phải khai báo `scale`.\n"
                f"    → Replica: REPLICA_PNG_DEPTH_SCALE | milimét: "
                f"MILLIMETER_PNG_DEPTH_SCALE"
            )
        out = raw.astype(np.float32) / float(scale)
    else:
        raise ValueError(f"Kiểu dtype không hỗ trợ: {raw.dtype} ({p})")

    out[~np.isfinite(out)] = np.nan
    out[out <= 0.0] = np.nan
    return out


# ── 3. Intrinsics ───────────────────────────────────────────────────────


def parse_intrinsics_txt(path: Union[str, Path]) -> Dict[str, Any]:
    """Đọc `intrinsics.txt` → dict thông số nội tại camera.

    Chấp nhận 3 dạng nội dung (bỏ qua dòng trống và dòng comment `#`):

    | Nội dung dòng      | Ví dụ                    | Quy ước                      |
    |--------------------|--------------------------|------------------------------|
    | 4 số               | `600 600 320 240`       | `fx fy cx cy`                |
    | 6 số               | `600 600 320 240 640 480`| `fx fy cx cy W H`            |
    | 9 số               | 3×3 K theo hàng         | `K[0,0] K[1,1] K[0,2] K[1,2]`|

    Nếu file không ghi `width`/`height` thì suy ra bằng `2·cx` và `2·cy`
    (đúng với camera lấy nét ở tâm ảnh — giả định chuẩn của các dataset này).

    Returns
    -------
    dict
        ``{"fx": float, "fy": float, "cx": float, "cy": float,
           "width": int, "height": int, "source": str}``

        `source` để `CameraIntrinsics.from_dict()` dùng đánh dấu `is_calibrated`.

    Raises
    ------
    FileNotFoundError, ValueError

    Examples
    --------
    >>> parse_intrinsics_txt("intrinsics.txt")["width"]
    640
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy intrinsics: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"File intrinsics rỗng: {p}")

    vals = np.array([float(x) for x in lines[0].replace(",", " ").split()], dtype=float)

    if vals.size == 4:
        fx, fy, cx, cy = vals
        width = int(round(2 * cx))
        height = int(round(2 * cy))
    elif vals.size == 6:
        fx, fy, cx, cy, width, height = vals
        width, height = int(round(width)), int(round(height))
    elif vals.size == 9:
        K = vals.reshape(3, 3)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        width = int(round(2 * cx))
        height = int(round(2 * cy))
    else:
        raise ValueError(
            f"File intrinsics phải có 4, 6 hoặc 9 số — thấy {vals.size} trong {p}.\n"
            f"    → Dòng đầu: {lines[0][:80]}"
        )

    if fx <= 0 or fy <= 0:
        raise ValueError(f"Tiêu cự không hợp lệ (fx={fx}, fy={fy}) trong {p}")

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": width,
        "height": height,
        "source": str(p),
    }


# ── 4. Pose ─────────────────────────────────────────────────────────────


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → ma trận xoay 3×3. Chuẩn hoá quaternion trước."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion có độ lớn bằng 0 — không tạo được ma trận xoay.")
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def parse_pose_line(
    line: str,
    opengl_to_opencv: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse 1 dòng `traj.txt` / `poses.txt` → `(R, t)`.

    Hỗ trợ 2 dạng (Replica preprocessed có cả hai biến thể):

    | Số phần tử | Định dạng                     | Nguồn                    |
    |------------|-------------------------------|--------------------------|
    | 16         | ma trận 4×4 row-major         | **Replica `traj.txt` (NICE-SLAM)** |
    | 7          | `tx ty tz qx qy qz qw`        | iMAP / một số bản khác   |

    Với dạng 16, `R = M[:3, :3]`, `t = M[:3, 3]`.

    ⚠️ **Thứ tự quaternion là `x, y, z, w`** — không phải `w, x, y, z`. Đảo thứ tự
    vẫn ra ma trận xoay hợp lệ nhưng **xoay sai hướng**, rất khó phát hiện bằng mắt.

    ⚠️ **Quy ước trục — phải biết dataset nào dùng cái nào.**
    Dự án này (khớp `localizer_3d.py`) dùng **OpenCV**: X phải, Y **xuống**, Z **xa**.
    Nhưng `traj.txt` của Replica lưu theo **OpenGL/Blender**: X phải, Y **lên**, Z **ngược**.
    Dùng thẳng sẽ ra toạ độ 3D sai ở trục Y và Z.

    NICE-SLAM xử lý việc này bằng cách **đảo cột 1 và cột 2** của phần 3×4:

    ```python
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    ```

    Vì vậy với Replica **bắt buộc** truyền `opengl_to_opencv=True`.

    Parameters
    ----------
    line : str
        Nội dung 1 dòng file pose.
    opengl_to_opencv : bool
        `True` nếu nguồn lưu pose theo quy ước OpenGL (Replica `traj.txt`).
        `False` (mặc định) = nguồn đã ở quy ước OpenCV.

    Returns
    -------
    (np.ndarray, np.ndarray)
        `R` hình `(3, 3)` float, `t` hình `(3,)` float (mét).

    Raises
    ------
    ValueError

    Examples
    --------
    >>> R, t = parse_pose_line("1 2 3 0 0 0 1")   # quaternion w=1 → R = I
    >>> np.allclose(R, np.eye(3)), t
    (True, array([1., 2., 3.]))

    >>> # Replica: đảo cột 1,2 để về hệ OpenCV
    >>> line = "1 0 0 0  0 1 0 0  0 0 1 0  9 8 7 1"
    >>> R_raw, _ = parse_pose_line(line)
    >>> R_cv, _ = parse_pose_line(line, opengl_to_opencv=True)
    >>> np.allclose(np.diag(R_cv), [1, -1, -1])   # cột Y và Z bị đảo dấu
    True
    """
    s = str(line).strip()
    if not s or s.startswith("#"):
        raise ValueError(f"Dòng pose rỗng hoặc là comment: {line!r}")

    vals = np.array([float(x) for x in s.replace(",", " ").split()], dtype=float)

    if vals.size == 7:
        t = vals[0:3].copy()
        R = _quat_to_R(vals[3:7])
    elif vals.size == 16:
        M = vals.reshape(4, 4)
        R = M[:3, :3].copy()
        t = M[:3, 3].copy()
    else:
        raise ValueError(
            f"Dòng pose phải có 7 số (tx ty tz qx qy qz qw) hoặc 16 số (4×4) — "
            f"thấy {vals.size}.\n    → Dòng: {s[:80]}"
        )

    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        raise ValueError(f"Pose chứa giá trị NaN/inf: {s[:80]}")

    if opengl_to_opencv:
        # Nhân cột 1,2 của R bằng -1 == nhân phải bởi diag(1,-1,-1,1)
        # (cột 0 và 3 nằm ngoài khối 3×3; t là cột 3 nên KHÔNG bị đổi)
        R = R.copy()
        R[:, 1] *= -1.0
        R[:, 2] *= -1.0

    return R, t


# ── 5. Toán 2D → 3D ─────────────────────────────────────────────────────


def _intr_get(intr: Union[Dict[str, Any], Any], key: str) -> float:
    """Lấy 1 trường từ intrinsics (dict hoặc object có `.fx`, `.cx`, ...)."""
    if isinstance(intr, dict):
        if key not in intr:
            raise KeyError(f"intrinsics thiếu trường {key!r}. Cần: fx, fy, cx, cy.")
        return float(intr[key])
    val = getattr(intr, key, None)
    if val is None:
        raise AttributeError(
            f"intrinsics không có thuộc tính {key!r} "
            f"(kiểu {type(intr).__name__}). Cần: fx, fy, cx, cy."
        )
    return float(val)


def pixel_to_cam(
    pixel: Sequence[float],
    depth_m: float,
    intr: Union[Dict[str, Any], Any],
) -> Tuple[float, float, float]:
    """Back-project `(u, v)` + depth → tọa độ 3D hệ camera.

    Công thức pinhole, **khớp đúng** `localizer_3d.CameraIntrinsics.pixel_to_metric_xyz`:

    ```
    X = (u - cx) / fx · Z      Y = (v - cy) / fy · Z      Z = depth_m
    ```

    Quy ước trục: **X phải, Y xuống dưới, Z ra xa camera** — giống hệ tọa độ ảnh.

    Parameters
    ----------
    pixel : (u, v)
        Tọa độ pixel (float được, không cần làm tròn).
    depth_m : float
        Khoảng cách theo trục quang, **đơn vị mét**. Dùng `load_depth_map()` để
        chắc chắn đơn vị đúng.
    intr : dict | CameraIntrinsics
        Nhận được cả dict (như trong sample JSON mục 5.2 của contract) lẫn object
        `CameraIntrinsics` — để script không phải tự chuyển đổi.

    Returns
    -------
    (float, float, float)
        `(X, Y, Z)` đơn vị mét.

    Raises
    ------
    ValueError
        Nếu `depth_m` không phải số hữu hạn hoặc ≤ 0.

    Examples
    --------
    >>> intr = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    >>> xyz = pixel_to_cam((320, 240), 2.0, intr)      # pixel chính giữa → X=Y=0
    >>> xyz
    (0.0, 0.0, 2.0)
    """
    u, v = float(pixel[0]), float(pixel[1])
    fx, fy = _intr_get(intr, "fx"), _intr_get(intr, "fy")
    cx, cy = _intr_get(intr, "cx"), _intr_get(intr, "cy")

    z = float(depth_m)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(
            f"depth_m không hợp lệ: {depth_m!r}. Phải là số hữu hạn và > 0 (mét)."
        )

    return ((u - cx) / fx * z, (v - cy) / fy * z, z)


def cam_to_world(
    xyz_cam: Sequence[float],
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Đổi tọa độ hệ camera → hệ thế giới: ``xyz_world = R @ xyz_cam + t``.

    Parameters
    ----------
    xyz_cam : (X, Y, Z)
        Tọa độ hệ camera (mét) — output của `pixel_to_cam`.
    R : (3, 3) array
        Ma trận xoay từ `parse_pose_line`.
    t : (3,) array
        Vị trí camera trong hệ thế giới (mét).

    Returns
    -------
    np.ndarray
        Hình `(3,)` float — tọa độ hệ thế giới (mét).

    Examples
    --------
    >>> cam_to_world((0.0, 0.0, 2.0), np.eye(3), np.array([1.0, 2.0, 3.0]))
    array([1., 2., 5.])
    """
    xyz = np.asarray(xyz_cam, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    return R @ xyz + t


# ── 6. Nhiễu pose ───────────────────────────────────────────────────────


def _rodrigues(w: np.ndarray) -> np.ndarray:
    """Vector xoay (rad) → ma trận xoay 3×3 (Rodrigues), không cần scipy/cv2."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(w, dtype=float) / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=float
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def add_pose_noise(
    R: np.ndarray,
    t: np.ndarray,
    seed: Optional[int] = None,
    sigma_trans_m: float = 0.02,
    sigma_rot_deg: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Thêm nhiễu ngẫu nhiên vào pose camera để mô phỏng sai số thực tế.

    Mặc định theo `SIM_CONTRACT.md` mục 11 Phase 4: **σ dịch 2 cm, σ xoay 0.5°**
    mỗi trục (độc lập, phân phối Gauss). Nhiễu xoay áp dụng **trong hệ camera**
    (`R_noisy = R · exp(ω)`) — mô phỏng camera bị đặt lệch, đúng như lúc lắp đặt thật.

    Vì sao cần case nhiễu: pose chuẩn cho MAE đẹp (~0.13 m) nhưng **không phải thực tế**.
    Chỉ báo cáo số pose chuẩn là tự đánh giá thuận lợi. Số để báo cáo là case nhiễu.

    Parameters
    ----------
    R, t : array
        Pose gốc từ `parse_pose_line`.
    seed : int | None
        Cho phép tái lập đúng kết quả. `None` = ngẫu nhiên thật (không reproduce được).
    sigma_trans_m : float
        σ nhiễu dịch chuyển (mét) mỗi trục.
    sigma_rot_deg : float
        σ nhiễu xoay (độ) mỗi trục.

    Returns
    -------
    (np.ndarray, np.ndarray)
        `(R_noisy, t_noisy)`.

    Examples
    --------
    >>> R1, t1 = add_pose_noise(np.eye(3), np.zeros(3), seed=42)
    >>> R2, t2 = add_pose_noise(np.eye(3), np.zeros(3), seed=42)
    >>> np.allclose(t1, t2)      # cùng seed → cùng kết quả
    True
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    rng = np.random.default_rng(seed)

    t_noisy = t + rng.normal(0.0, sigma_trans_m, size=3)
    omega = np.deg2rad(rng.normal(0.0, sigma_rot_deg, size=3))
    R_noisy = R @ _rodrigues(omega)
    return R_noisy, t_noisy


# ── 7. Căn scale + shift cho depth relative ─────────────────────────────


def align_scale_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> ScaleShift:
    """Tính hệ số căn scale+shift ít nhất bình phương cho depth **relative**.

    Model relative (DepthART relative, Depth Anything V1/V2) cho depth *affine-invariant*:
    `d_metric = s · d_rel + b`. Không căn thì sai số tính ra vô nghĩa.

    ⚠️ **Model metric KHÔNG được căn** — nó đã trả về mét, căn lại sẽ hỏng kết quả.
    Xem `SIM_CONTRACT.md` mục 5.3.

    Áp dụng cho **cả bản đồ** (không phải chỉ pixel đã dùng để fit):
    ``aligned = scale * depth_pred + shift``.

    Parameters
    ----------
    pred, gt : array
        Hai bản đồ depth cùng shape. `pred` là relative, `gt` là GT tính bằng mét.
    valid_mask : array bool | None
        Mặt nạ pixel hợp lệ. Mặc định = `NaN/inf` bị loại ở cả 2 map và `gt > 0`.

    Returns
    -------
    ScaleShift
        NamedTuple `(scale, shift)`.

    Raises
    ------
    ValueError
        Nếu không còn pixel hợp lệ nào, hoặc `pred` không đổi trên các pixel đó
        (variance = 0 → không xác định được `scale`).

    Examples
    --------
    >>> gt = np.array([[1.0, 2.0, 3.0, 4.0]])
    >>> rel = (gt - 0.5) / 2.0              # depth_metric = 2 * rel + 0.5
    >>> ss = align_scale_shift(rel, gt)
    >>> np.allclose([ss.scale, ss.shift], [2.0, 0.5])
    True
    """
    pred = np.asarray(pred, dtype=float).reshape(-1)
    gt = np.asarray(gt, dtype=float).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"pred và gt phải cùng shape — {pred.shape} vs {gt.shape}")

    if valid_mask is None:
        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0.0)
    else:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1) & np.isfinite(pred)
        valid &= np.isfinite(gt) & (gt > 0.0)

    n = int(valid.sum())
    if n < 2:
        raise ValueError(
            f"Chỉ còn {n} pixel hợp lệ để căn scale+shift — cần tối thiểu 2.\n"
            f"    → Kiểm tra depth GT và depth dự đoán có NaN ở diện tích lớn không."
        )

    p, g = pred[valid], gt[valid]
    var_p = float(np.var(p))
    if var_p < 1e-12:
        raise ValueError(
            "depth dự đoán không đổi trên các pixel hợp lệ (var ≈ 0) — "
            "không xác định được hệ số scale.\n"
            "    → Model có thể trả về bản đồ hằng số; kiểm tra lại backend."
        )

    scale = float(np.cov(p, g, bias=True)[0, 1] / var_p)
    shift = float(np.mean(g) - scale * np.mean(p))
    return ScaleShift(scale=scale, shift=shift)


# ── 8. Nạp backend depth ────────────────────────────────────────────────


def load_module_backend(
    name: str,
    intrinsics: Optional[Any] = None,
    load: bool = False,
    **kwargs: Any,
) -> Any:
    """Nạp backend depth từ module chính (chỉ **import**, không sửa).

    Import `depth_model` / `metric_depth` từ thư mục cha `simulation/` bằng cách
    thêm nó vào `sys.path` — để script chạy được mà không cần cài package.

    Parameters
    ----------
    name : str
        | Giá trị | Trả về | Ghi chú |
        |---|---|---|
        | `"relative"`, `"depthart"`, `"mnv4s"`, `"depthart-mnv4s"` | `DepthBackend` | depth **relative**, không có đơn vị mét |
        | `"metric"`, `"depthart-metric"` | `MetricDepthBackend` | depth **mét**; **bắt buộc** `intrinsics` |

        Rút gọn như `"mnv4s"` sẽ tự truyền `depthart_encoder="mnv4s"`.
    intrinsics : CameraIntrinsics | None
        Bắt buộc với backend metric (nó là **input** của model, không phải hậu xử lý).
    load : bool
        `True` = gọi `.load()` ngay (tải checkpoint, vài giây). Mặc định `False` để
        test chữ ký hàm không mất thời gian.
    **kwargs
        Chuyển tiếp cho constructor (`depthart_encoder`, `domain`, `device`, ...).

    Returns
    -------
    object
        Instance backend (chưa load nếu `load=False`).

    Raises
    ------
    ValueError — tên lạ, thiếu `intrinsics` khi chọn metric.
    ImportError — không import được module chính.

    Examples
    --------
    >>> b = load_module_backend("mnv4s")        # chưa load, rất nhanh
    >>> b.model_name
    'unloaded'
    """
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    key = str(name).strip().lower().replace("_", "-")
    relative_aliases = {"relative", "depthart", "mnv4s", "depthart-mnv4s", "mnv4m", "s"}
    metric_aliases = {"metric", "depthart-metric", "metric-indoor"}

    # Tên encoder chuẩn của DepthART. Rút gọn của người dùng (`mnv4s`) phải được ánh xạ
    # sang tên đúng, vì backend chỉ nhận đúng danh sách này và raise nếu sai.
    encoder_map = {
        "mnv4s": "MNV4-S", "mnv4-s": "MNV4-S", "depthart-mnv4s": "MNV4-S",
        "mnv4m": "MNV4-M", "mnv4-m": "MNV4-M",
        "mnv4m-slim-spf": "MNV4-M-SLIM-SPF",
        "s": "S", "tinyvim-s": "S",
        "b": "B", "tinyvim-b": "B",
        "l": "L", "tinyvim-l": "L",
    }

    if key in relative_aliases:
        from depth_model import DepthBackend  # lazy: chỉ khi thật sự cần

        if key in ("mnv4s", "mnv4-s", "depthart-mnv4s", "mnv4m"):
            kwargs.setdefault("depthart_encoder", encoder_map[key])
        # Chuẩn hoá cả khi caller truyền `depthart_encoder='mnv4s'` trực tiếp —
        # backend chỉ nhận đúng tên trong `encoder_map`, sai tên sẽ raise.
        enc = kwargs.get("depthart_encoder")
        if enc is not None:
            canon = encoder_map.get(str(enc).strip().lower().replace("_", "-"))
            if canon is None:
                raise ValueError(
                    f"Encoder không hợp lệ: {enc!r}.\n"
                    f"    → Chấp nhận: {sorted(set(encoder_map.values()))} "
                    f"(hoặc để backend tự chọn)."
                )
            kwargs["depthart_encoder"] = canon
        backend = DepthBackend(prefer="depthart", **kwargs)
    elif key in metric_aliases:
        from metric_depth import MetricDepthBackend  # lazy

        if intrinsics is None:
            raise ValueError(
                "Backend metric BẮT BUỘC cần CameraIntrinsics — nó là input của model.\n"
                "  → Nạp từ calibration: CameraIntrinsics.load('intrinsics.json')\n"
                "  → Trong simulator: parse_intrinsics_txt(...) → CameraIntrinsics.from_dict(...)\n"
                "  ⚠️ Dùng intrinsics ước lượng (from_fov) thì sai số mét sai ngay từ đầu."
            )
        encoder = str(kwargs.get("encoder", "s")).upper()
        canon = {"MNV4S": "MNV4-S", "MNV4M": "MNV4-M", "S": "S", "B": "B", "L": "L"}.get(
            encoder.replace("-", ""), encoder)
        if canon not in ("S", "B", "L"):
            raise ValueError(
                f"DepthART metric chỉ hỗ trợ encoder TinyViM S/B/L, không có MobileNetV4.\n"
                f"    → Nhận: {encoder!r} (sau khi chuẩn hoá: {canon!r}). "
                f"Dùng encoder='s' cho nhẹ nhất."
            )
        kwargs["encoder"] = canon
        backend = MetricDepthBackend(intrinsics=intrinsics, **kwargs)
    else:
        raise ValueError(
            f"Tên backend không hợp lệ: {name!r}.\n"
            f"  → Relative: 'relative', 'depthart', 'mnv4s', 'depthart-mnv4s'\n"
            f"  → Metric:   'metric', 'depthart-metric' (cần intrinsics)"
        )

    if load:
        backend.load()
    return backend


# ── 9. CLI: --help / --selftest ─────────────────────────────────────────

_FUNCTION_DOCS: Tuple[Tuple[str, str, str], ...] = (
    ("check_disk_space", "min_c_gb=2.0, min_d_gb=10.0, data_root",
     "Kiểm tra ổ trống, raise lỗi kèm cách sửa. Gọi trước mọi I/O nặng."),
    ("load_depth_map", "path, scale=None",
     "PNG16/NPY → mét (float32), pixel lỗi = NaN. scale BẮT BUỘC cho depth nguyên: "
     "Replica=6553.5, milimét=1000.0."),
    ("parse_intrinsics_txt", "path",
     "intrinsics.txt → dict {fx, fy, cx, cy, width, height, source}. Nhận 4/6/9 số."),
    ("replica_intrinsics", "()",
     "Intrinsics Replica (1200×680, f=600) — dataset KHÔNG có file intrinsics."),
    ("parse_pose_line", "line, opengl_to_opencv=False",
     "traj.txt/poses.txt → (R, t). Nhận 7 số (quat x,y,z,w) hoặc 16 số (4×4). "
     "Replica traj.txt BẮT BUỘC opengl_to_opencv=True."),
    ("pixel_to_cam", "pixel, depth_m, intr",
     "(u,v) + depth mét → (X,Y,Z) hệ camera. X phải, Y xuống, Z xa."),
    ("cam_to_world", "xyz_cam, R, t",
     "Hệ camera → hệ thế giới: R @ xyz + t."),
    ("add_pose_noise", "R, t, seed, sigma_trans_m=0.02, sigma_rot_deg=0.5",
     "Nhiễu pose (mặc định 2 cm / 0.5°). Truyền seed để tái lập kết quả."),
    ("align_scale_shift", "pred, gt, valid_mask=None",
     "Căn scale+shift cho depth RELATIVE → ScaleShift(scale, shift). Metric thì ĐỪNG gọi."),
    ("load_module_backend", "name, intrinsics=None, load=False, **kw",
     "Nạp DepthBackend (relative) / MetricDepthBackend (metric). Import lazy."),
)


def _build_parser() -> argparse.ArgumentParser:
    epilog_lines = ["Hàm trong module `utils.py`:", ""]
    for fn, sig, desc in _FUNCTION_DOCS:
        epilog_lines.append(f"  {fn}({sig})")
        epilog_lines.append(f"      {desc}")
    epilog_lines += [
        "",
        "Ví dụ:",
        "  python utils.py --selftest          # test toán, KHÔNG cần model",
        "  python utils.py --check-disk         # chỉ kiểm tra ổ đĩa",
        "",
        "Xem `SIM_CONTRACT.md` mục 8 để biết hàm này dùng ở đâu.",
    ]
    return argparse.ArgumentParser(
        prog="utils.py",
        description=(
            "Hàm dùng chung cho simulator đánh giá depth + toán 2D→3D. "
            "Script khác import từ đây, không viết lại."
        ),
        epilog="\n".join(epilog_lines),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _selftest(data_root: Path) -> int:
    """Test toán round-trip, KHÔNG cần model. Trả về số lỗi (0 = pass)."""
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 72)
    print("SELFTEST utils.py — không cần model, không cần dataset")
    print("=" * 72)

    # -- 0. Ổ đĩa — báo cáo TÌNH TRẠNG MÁY, không tính vào FAIL.
    #    Đây là điều kiện môi trường, utils.py không sửa được. Nếu thiếu chỗ thì
    #    đây chính là cảnh báo sớm cần thấy, không phải lỗi code.
    print("\n[0] Tình trạng ổ đĩa (INFO — không tính vào kết quả PASS/FAIL)")
    try:
        info = check_disk_space(data_root=data_root)
        print(f"  [ OK ] Ổ đĩa đủ chỗ: {json.dumps(info, ensure_ascii=False)}")
    except RuntimeError as e:
        msg = str(e).replace("\n", "\n         ")
        print("  [CẢNH BÁO] Ổ đĩa KHÔNG đủ — phase sau sẽ không chạy được:")
        print(f"         {msg}")
        print("         → Đây là tình trạng máy, không phải lỗi utils.py.")
        print("         → Ổ C < 2 GB là rủi ro Windows bị treo: dọn ổ C trước khi tải gì.")

    # -- 1. Round-trip: điểm 3D -> pixel -> depth -> 3D
    print("\n[1] Round-trip pixel_to_cam (tọa độ 3D -> pixel -> 3D)")
    intr = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480}
    for target in [(0.0, 0.0, 2.0), (0.5, -0.3, 3.5), (-0.8, 0.6, 1.5)]:
        X, Y, Z = target
        u = intr["fx"] * X / Z + intr["cx"]
        v = intr["fy"] * Y / Z + intr["cy"]
        back = pixel_to_cam((u, v), Z, intr)
        err = max(abs(back[i] - target[i]) for i in range(3))
        check(f"pixel_to_cam round-trip {target}", err < 1e-9,
              f"sai số {err:.2e} m")

    # -- 2. pixel chính giữa -> X=Y=0
    print("\n[2] Pixel chính giữa")
    mid = pixel_to_cam((intr["cx"], intr["cy"]), 2.0, intr)
    check("center pixel → X=Y=0", abs(mid[0]) < 1e-12 and abs(mid[1]) < 1e-12,
          f"({mid[0]:.3f}, {mid[1]:.3f}, {mid[2]:.3f})")

    # -- 3. Round-trip qua hệ thế giới
    print("\n[3] Round-trip hệ thế giới (pixel_to_cam → cam_to_world)")
    R0, t0 = parse_pose_line("1.5 0.5 -2.0 0 0 0 1")   # quat w=1 → R = I
    check("parse_pose_line quat(w=1) → R=I", np.allclose(R0, np.eye(3)))
    check("parse_pose_line giữ t", np.allclose(t0, [1.5, 0.5, -2.0]))
    xyz_cam = pixel_to_cam((417.0, 133.0), 2.8, intr)
    xyz_world = cam_to_world(xyz_cam, R0, t0)
    xyz_back = R0.T @ (xyz_world - t0)
    err = float(np.max(np.abs(xyz_back - np.asarray(xyz_cam))))
    check("cam_to_world rồi quay ngược lại", err < 1e-12, f"sai số {err:.2e} m")

    # -- 4. parse_pose_line dạng 4x4 + chuyển quy ước OpenGL -> OpenCV
    print("\n[4] parse_pose_line dạng 16 số + quy ước trục")
    M = np.eye(4); M[:3, 3] = [1.0, 2.0, 3.0]
    R16, t16 = parse_pose_line(" ".join(f"{v:.6f}" for v in M.reshape(-1)))
    check("4×4 → (R, t)", np.allclose(R16, np.eye(3)) and np.allclose(t16, [1, 2, 3]))
    R_cv, t_cv = parse_pose_line(" ".join(f"{v:.6f}" for v in M.reshape(-1)),
                                 opengl_to_opencv=True)
    check("OpenGL→OpenCV đảo cột 1,2", np.allclose(np.diag(R_cv), [1, -1, -1]),
          f"diag={np.diag(R_cv)}")
    check("OpenGL→OpenCV KHÔNG đổi t", np.allclose(t_cv, [1, 2, 3]))
    check("ma trận xoay vẫn chuẩn hoá (RᵀR=I)",
          np.allclose(R_cv.T @ R_cv, np.eye(3)))
    check("default (OpenCV) giữ nguyên R", np.allclose(R16, np.eye(3)))

    # -- 5. Nhiễu pose: reproduce + đúng cỡ σ
    print("\n[5] add_pose_noise")
    _, ta = add_pose_noise(np.eye(3), np.zeros(3), seed=42)
    _, tb = add_pose_noise(np.eye(3), np.zeros(3), seed=42)
    check("cùng seed → kết quả giống nhau", np.allclose(ta, tb), f"t={ta}")
    _, tc = add_pose_noise(np.eye(3), np.zeros(3), seed=7)
    check("seed khác → kết quả khác", not np.allclose(ta, tc))
    samples = np.array([add_pose_noise(np.eye(3), np.zeros(3), seed=s)[1][0]
                        for s in range(4000)])
    est = float(np.std(samples))
    check("σ dịch chuyển ≈ 2 cm", abs(est - 0.02) < 0.002, f"σ ước lượng = {est*100:.2f} cm")

    # -- 6. Căn scale+shift
    print("\n[6] align_scale_shift")
    rng = np.random.default_rng(0)
    rel = rng.random((40, 60)) * 10 + 1.0
    true_s, true_b = 2.5, -0.35
    gt_map = true_s * rel + true_b
    gt_map[0, 0] = np.nan                                  # pixel lỗi phải bị loại
    ss = align_scale_shift(rel, gt_map)
    check("khôi phục đúng scale", abs(ss.scale - true_s) < 1e-9, f"scale={ss.scale:.9f}")
    check("khôi phục đúng shift", abs(ss.shift - true_b) < 1e-9, f"shift={ss.shift:.9f}")
    check("bỏ qua pixel NaN",
          float(np.nanmax(np.abs(ss.scale * rel + ss.shift - gt_map))) < 1e-9)
    try:
        align_scale_shift(np.full((5, 5), 3.0), np.arange(25, dtype=float).reshape(5, 5))
        check("báo lỗi khi pred không đổi", False, "không raise")
    except ValueError:
        check("báo lỗi khi pred không đổi", True, "raise ValueError đúng")

    # -- 7. load_depth_map trên ổ D (KHÔNG ghi tạm ra ổ C)
    print("\n[7] load_depth_map (file tạm đặt trên ổ data, không đụng ổ C)")
    import tempfile

    tmp_parent = Path(data_root)
    try:
        tmp_parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        check("tạo thư mục tạm", False, f"{e}")
        tmp_parent = None
    if tmp_parent is not None:
        # dir=tmp_parent: tempfile KHÔNG mặc định ghi vào %TEMP% (ổ C)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as td:
            import cv2

            td = Path(td)
            d_mm = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 500)  # mm
            cv2.imwrite(str(td / "d.png"), d_mm)
            d_npy = (np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0)
            np.save(td / "d.npy", d_npy)
            cv2.imwrite(str(td / "d8.png"), (d_mm / 100).astype(np.uint8))

            got = load_depth_map(td / "d.png", scale=MILLIMETER_PNG_DEPTH_SCALE)
            # 0 mm = "không đo được" → phải ra NaN, nên mảng kỳ vọng cũng phải NaN ở ô đó
            expected = np.where(d_mm == 0, np.nan, d_mm / 1000.0)
            check("PNG uint16 + scale=1000 → mét",
                  np.allclose(got, expected, equal_nan=True),
                  f"max={np.nanmax(got):.4f} m")
            check("pixel 0 → NaN (không đo được)", bool(np.isnan(got[0, 0])),
                  f"got[0,0]={got[0, 0]}")
            check("NPY → giữ nguyên mét", np.allclose(load_depth_map(td / "d.npy"), d_npy))
            check("dtype float32", got.dtype == np.float32, str(got.dtype))

            # Hệ số Replica: cùng file, khác scale → kết quả khác 6.5535 lần
            got_rep = load_depth_map(td / "d.png", scale=REPLICA_PNG_DEPTH_SCALE)
            check("scale Replica 6553.5 ≠ scale 1000",
                  not np.allclose(np.nan_to_num(got_rep), np.nan_to_num(got)),
                  f"6553.5→{np.nanmax(got_rep):.4f} m, 1000→{np.nanmax(got):.4f} m")
            try:
                load_depth_map(td / "d.png")  # thiếu scale
                check("bắt buộc khai báo scale", False, "không raise")
            except ValueError as e:
                check("bắt buộc khai báo scale", "6553.5" in str(e),
                      "raise ValueError nhắc đúng hệ số Replica")
            try:
                load_depth_map(td / "d8.png", scale=1000.0)
                check("từ chối PNG 8-bit", False, "không raise")
            except ValueError:
                check("từ chối PNG 8-bit", True, "raise ValueError đúng")

    # -- 7b. Intrinsics Replica (dataset không có file intrinsics)
    print("\n[7b] replica_intrinsics()")
    ri = replica_intrinsics()
    check("Replica 1200×680", (ri["width"], ri["height"]) == (1200, 680),
          f"{ri['width']}×{ri['height']}")
    check("Replica fx=fy=600, cx=599.5, cy=339.5",
          (ri["fx"], ri["fy"], ri["cx"], ri["cy"]) == (600.0, 600.0, 599.5, 339.5))
    check("ghi kèm png_depth_scale=6553.5", ri["png_depth_scale"] == 6553.5)
    check("không suy width=2*cx (sẽ ra 1199)",
          ri["width"] != int(round(2 * ri["cx"])),
          f"2*cx={2 * ri['cx']:.0f} nhưng width={ri['width']}")

    # -- 8. Ràng buộc: import utils không được kéo torch/cv2/model theo.
    #    Đo bằng PROCESS MỚI — nếu đo trong process này thì chính selftest đã
    #    import cv2 ở bước [7] và kết quả luôn sai.
    print("\n[8] Ràng buộc import (mục 11: không import nặng ở cấp module)")
    import subprocess
    import time

    probe = (
        "import sys, time; t=time.perf_counter(); "
        "import utils; dt=(time.perf_counter()-t)*1000; "
        "heavy=[m for m in ('torch','cv2','depth_model','metric_depth') if m in sys.modules]; "
        "print(f'{dt:.0f} {heavy}')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=120,
        )
        ms_txt, heavy_txt = r.stdout.strip().split(" ", 1)
        ms = float(ms_txt)
        heavy = eval(heavy_txt)  # noqa: S307 — output do chính subprocess in ra
        check("không có torch/cv2/model ở cấp module", not heavy,
              f"đang có: {heavy}" if heavy else "sạch")
        check("import utils < 500 ms", ms < 500, f"{ms:.0f} ms")
    except Exception as e:  # noqa: BLE001
        check("probe import subprocess", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    if fails:
        print(f"KẾT QUẢ: {len(fails)}/{len(fails)} MỤC FAIL → {fails}")
        return 1
    print("KẾT QUẢ: TẤT CẢ PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point cho CLI: ``--help``, ``--selftest``, ``--check-disk``."""
    # Bọc stdout UTF-8 để không gặp UnicodeEncodeError trên Windows (cp1252)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    parser = _build_parser()
    parser.add_argument(
        "--selftest", action="store_true",
        help="Chạy test toán round-trip (không cần model, không cần dataset).",
    )
    parser.add_argument(
        "--check-disk", action="store_true",
        help="Chỉ kiểm tra ổ đĩa rồi thoát.",
    )
    parser.add_argument(
        "--data-root", default=str(DEFAULT_DATA_ROOT),
        help=f"Gốc dữ liệu simulator (mặc định: {DEFAULT_DATA_ROOT}).",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest(Path(args.data_root))
    if args.check_disk:
        info = check_disk_space(data_root=Path(args.data_root))
        print("OK — " + json.dumps(info, ensure_ascii=False))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
