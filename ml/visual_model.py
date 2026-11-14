"""Shared pieces of the experimental visual pattern-type model (training in ml/train_visual.py, serving in backend/visual.py).

Architectures are torchvision ImageNet-pretrained backbones with a new multi-label head. Inputs are letterboxed
(padded, never cropped) RGB screenshots. Evidence: Grad-CAM on the last convolutional block; the box around the largest
connected region of the top 10% of the heatmap is mapped back to original image pixels.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ARCHS = {
    "mobilenet_v3_large": "IMAGENET1K_V2",
    "efficientnet_b0": "IMAGENET1K_V1",
    "convnext_tiny": "IMAGENET1K_V1",
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build(arch: str, n_labels: int, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tvm
    weights = getattr(tvm, {"mobilenet_v3_large": "MobileNet_V3_Large_Weights", "efficientnet_b0": "EfficientNet_B0_Weights",
                            "convnext_tiny": "ConvNeXt_Tiny_Weights"}[arch])[ARCHS[arch]] if pretrained else None
    m = getattr(tvm, arch)(weights=weights)
    last = m.classifier[-1]
    m.classifier[-1] = nn.Linear(last.in_features, n_labels)
    return m


def head_params(m: nn.Module) -> list:
    return list(m.classifier[-1].parameters())


def cam_layer(m: nn.Module) -> nn.Module:
    return m.features[-1]


def to_tensor(batch_uint8: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> normalised float tensor (N,3,H,W)."""
    x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def calibrated(logits: np.ndarray, temps: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / temps))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One temperature for one label, minimising binary NLL on validation (grid, then refine)."""
    def nll(t):
        p = np.clip(1.0 / (1.0 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    grid = np.exp(np.linspace(np.log(0.2), np.log(10), 200))
    best = grid[int(np.argmin([nll(t) for t in grid]))]
    fine = np.linspace(best * 0.9, best * 1.1, 41)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def gradcam(model: nn.Module, x: torch.Tensor, label_idx: int) -> np.ndarray:
    """Grad-CAM heatmap (H,W) in [0,1] for one image tensor (1,3,H,W) and one label."""
    acts, grads = {}, {}
    layer = cam_layer(model)
    h1 = layer.register_forward_hook(lambda _m, _i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        out = model(x)
        out[0, label_idx].backward()
        a, g = acts["a"][0], grads["g"][0]
        w = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * a).sum(0)).detach().float().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    if cam.max() > 0:
        cam = cam / cam.max()
    from PIL import Image
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((x.shape[3], x.shape[2]), Image.Resampling.BILINEAR),
                      dtype=np.float32) / 255.0


def evidence_box(cam: np.ndarray, top: float = 0.10) -> tuple[int, int, int, int] | None:
    """Box (x, y, w, h) in heatmap pixels around the largest connected component of the top `top` fraction."""
    from scipy import ndimage
    if cam.max() <= 0:
        return None
    thr = np.quantile(cam, 1 - top)
    lab, n = ndimage.label(cam >= thr)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(cam), lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == k)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def to_original(box, img_w: int, img_h: int, size: int) -> dict:
    """Map a letterboxed (size x size) box back to original image pixels, clipped to the image."""
    s = size / max(img_w, img_h)
    px, py = (size - round(img_w * s)) / 2, (size - round(img_h * s)) / 2
    x, y, w, h = box
    x0 = max(0, min(img_w, round((x - px) / s)))
    y0 = max(0, min(img_h, round((y - py) / s)))
    x1 = max(0, min(img_w, round((x + w - px) / s)))
    y1 = max(0, min(img_h, round((y + h - py) / s)))
    return {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)}
