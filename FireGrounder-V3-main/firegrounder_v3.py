"""
FireGrounder V3: MobileNetV4 + FPN-lite heatmap head.

The V2 checkpoint and model are intentionally left untouched. V3 keeps the
classification task on a pooled deep feature, but predicts the fire-base
location from a two-level spatial feature pyramid:

    f4  (stride 4, 64x64 for a 256px input)  +  f8 (stride 8, 32x32)
                         -> 64x64 heatmap -> soft-argmax -> (x, y)

The public forward API returns [p_fire, x_norm, y_norm] by default. Training
can request a dictionary containing the heatmap logits as well.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("Install timm first: pip install timm") from exc


BACKBONE_NAME = "mobilenetv4_conv_medium"
IMG_SIZE = 256
HEATMAP_SIZE = 64
FIRE_THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


val_transform_v3 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class SoftArgmax2D(nn.Module):
    """Convert a heatmap into normalized (x, y) coordinates."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Expected heatmap logits with shape (B, 1, H, W)")

        batch, _, height, width = logits.shape
        probabilities = torch.softmax(
            logits.flatten(1) / self.temperature,
            dim=1,
        ).view(batch, height, width)

        ys = torch.linspace(0.0, 1.0, height, device=logits.device)
        xs = torch.linspace(0.0, 1.0, width, device=logits.device)
        x = (probabilities.sum(dim=1) * xs.view(1, width)).sum(dim=1)
        y = (probabilities.sum(dim=2) * ys.view(1, height)).sum(dim=1)
        return torch.stack((x, y), dim=1)


class FireGrounderV3(nn.Module):
    """MobileNetV4 with separate classification and spatial heatmap heads."""

    model_version = "v3_fpn_heatmap"

    def __init__(
        self,
        backbone_name: str = BACKBONE_NAME,
        pretrained: bool = True,
        img_size: int = IMG_SIZE,
        heatmap_size: int = HEATMAP_SIZE,
        hidden_channels: int = 96,
        softargmax_temperature: float = 0.07,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = int(img_size)
        self.heatmap_size = int(heatmap_size)
        self.hidden_channels = int(hidden_channels)
        self.softargmax_temperature = float(softargmax_temperature)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 4),
        )
        channels = list(self.backbone.feature_info.channels())
        if len(channels) != 3:
            raise RuntimeError(
                "Expected three feature levels from out_indices=(1, 2, 4), "
                f"got channels={channels}"
            )
        c4, c8, c32 = channels

        self.lat4 = nn.Conv2d(c4, hidden_channels, kernel_size=1)
        self.lat8 = nn.Conv2d(c8, hidden_channels, kernel_size=1)
        self.neck = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self.cls_head = nn.Sequential(
            nn.Linear(c32, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )
        self.softargmax = SoftArgmax2D(softargmax_temperature)

        total = sum(parameter.numel() for parameter in self.parameters())
        print(
            f"[FireGrounderV3] backbone={backbone_name} "
            f"channels={channels} heatmap={heatmap_size} params={total:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        features = self.backbone(x)
        f4, f8, f32 = features

        f4_lateral = self.lat4(f4)
        f8_lateral = self.lat8(f8)
        f8_lateral = F.interpolate(
            f8_lateral,
            size=f4_lateral.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.neck(f4_lateral + f8_lateral)
        heatmap_logits = self.heatmap_head(fused)
        if heatmap_logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            heatmap_logits = F.interpolate(
                heatmap_logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )

        cls_feature = F.adaptive_avg_pool2d(f32, output_size=1).flatten(1)
        p_fire = torch.sigmoid(self.cls_head(cls_feature))
        xy = self.softargmax(heatmap_logits)
        prediction = torch.cat((p_fire, xy), dim=1)

        if return_dict:
            return {
                "pred": prediction,
                "p_fire": p_fire[:, 0],
                "xy": xy,
                "heatmap_logits": heatmap_logits,
            }
        return prediction


class FocalLoss(nn.Module):
    """Binary focal loss operating on probabilities in [0, 1]."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(prediction, target, reduction="none")
        pt = torch.where(target > 0.5, prediction, 1.0 - prediction)
        alpha_t = torch.where(
            target > 0.5,
            torch.as_tensor(self.alpha, device=target.device, dtype=target.dtype),
            torch.as_tensor(1.0 - self.alpha, device=target.device, dtype=target.dtype),
        )
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


def make_gaussian_target(
    xy: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Create a differentiable Gaussian target with shape (B, 1, H, W)."""

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (B, 2)")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    device = xy.device
    dtype = xy.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xy = xy.clamp(0.0, 1.0)
    cx = xy[:, 0].view(-1, 1, 1) * (width - 1)
    cy = xy[:, 1].view(-1, 1, 1) * (height - 1)
    exponent = ((xx - cx).pow(2) + (yy - cy).pow(2)) / (2.0 * sigma * sigma)
    return torch.exp(-exponent).unsqueeze(1)


class FireLossV3(nn.Module):
    """Classification focal loss plus heatmap and optional coordinate loss."""

    def __init__(
        self,
        w_cls: float = 1.5,
        w_heatmap: float = 5.0,
        w_coord: float = 1.0,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        gaussian_sigma: float = 2.0,
        heatmap_positive_weight: float = 4.0,
    ):
        super().__init__()
        self.w_cls = float(w_cls)
        self.w_heatmap = float(w_heatmap)
        self.w_coord = float(w_coord)
        self.gaussian_sigma = float(gaussian_sigma)
        self.heatmap_positive_weight = float(heatmap_positive_weight)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(outputs, dict):
            raise TypeError("FireLossV3 expects model(..., return_dict=True)")
        if targets.ndim != 2 or targets.shape[1] < 3:
            raise ValueError("targets must have shape (B, 3): [has_fire, x, y]")

        p_fire = outputs["p_fire"]
        xy_pred = outputs["xy"]
        heatmap_logits = outputs["heatmap_logits"]
        has_fire = targets[:, 0]
        xy_true = targets[:, 1:3]
        fire_mask = has_fire > 0.5

        cls_loss = self.focal(p_fire, has_fire)
        target_heatmap = make_gaussian_target(
            xy_true,
            height=heatmap_logits.shape[-2],
            width=heatmap_logits.shape[-1],
            sigma=self.gaussian_sigma,
        )
        target_heatmap = target_heatmap * fire_mask.to(heatmap_logits.dtype).view(-1, 1, 1, 1)
        heatmap_probability = torch.sigmoid(heatmap_logits)
        heatmap_weights = 1.0 + self.heatmap_positive_weight * target_heatmap
        heatmap_loss = (
            (heatmap_probability - target_heatmap).pow(2) * heatmap_weights
        ).sum() / heatmap_weights.sum().clamp_min(1.0)

        if fire_mask.any():
            coord_loss = F.smooth_l1_loss(xy_pred[fire_mask], xy_true[fire_mask])
        else:
            coord_loss = heatmap_logits.sum() * 0.0

        total = (
            self.w_cls * cls_loss
            + self.w_heatmap * heatmap_loss
            + self.w_coord * coord_loss
        )
        parts = {
            "cls": cls_loss,
            "heatmap": heatmap_loss,
            "coord": coord_loss,
            "total": total,
        }
        return total, parts


def _load_checkpoint_file(path: str, device: str | torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_v3(path: str, device: str | torch.device = "cpu") -> FireGrounderV3:
    """Load a V3 checkpoint and validate its model version/state dict."""

    checkpoint = _load_checkpoint_file(path, device)
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    version = str(checkpoint.get("model_version", ""))
    looks_like_v3 = any(key.startswith("lat4.") for key in state.keys())
    if version and version != FireGrounderV3.model_version:
        raise ValueError(f"Checkpoint is {version}, expected {FireGrounderV3.model_version}")
    if not looks_like_v3:
        raise ValueError(
            "The checkpoint is not a FireGrounder V3 checkpoint. "
            "Use inference.py for the existing V1/V2 checkpoints."
        )

    config = checkpoint.get("config", {})
    model = FireGrounderV3(
        backbone_name=checkpoint.get("backbone", config.get("backbone", BACKBONE_NAME)),
        pretrained=False,
        img_size=int(checkpoint.get("img_size", config.get("img_size", IMG_SIZE))),
        heatmap_size=int(
            checkpoint.get("heatmap_size", config.get("heatmap_size", HEATMAP_SIZE))
        ),
        hidden_channels=int(
            checkpoint.get("hidden_channels", config.get("hidden_channels", 96))
        ),
        softargmax_temperature=float(
            checkpoint.get(
                "softargmax_temperature",
                config.get("softargmax_temperature", 0.07),
            )
        ),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(
        f"[INFO] Loaded {FireGrounderV3.model_version}: {path} "
        f"| device={device}"
    )
    return model


if __name__ == "__main__":
    print("Running FireGrounder V3 smoke test")
    model = FireGrounderV3(pretrained=False).eval()
    batch = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        outputs = model(batch, return_dict=True)
    assert outputs["pred"].shape == (2, 3)
    assert outputs["heatmap_logits"].shape == (2, 1, HEATMAP_SIZE, HEATMAP_SIZE)
    assert torch.isfinite(outputs["pred"]).all()
    assert ((outputs["pred"][:, 1:] >= 0.0) & (outputs["pred"][:, 1:] <= 1.0)).all()
    print("Smoke test passed")
