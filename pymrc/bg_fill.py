"""
Background hole-filling algorithms for the BG layer.

BG should remain smooth/low-frequency; visual realism is not the goal.
Avoid introducing texture or edge detail that hurts compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import cv2
import numpy as np

from .image_utils import ensure_odd


@dataclass(frozen=True)
class BgFillSpec:
    key: str
    label: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]


_LOCAL_MEAN_KERNEL = 31
_DIFFUSION_ITERS = 40
_DIFFUSION_SIGMA = 3.0
_INPAINT_RADIUS = 3


def _fg_mask_bool(fg_mask: np.ndarray) -> np.ndarray:
    return fg_mask.astype(np.uint8) > 0


def fill_bg_flat_mean(bg_image: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes with a single flat mean color from known BG pixels.
    """
    fg = _fg_mask_bool(fg_mask)
    if not fg.any():
        return bg_image.copy()

    bg_pixels = bg_image[~fg]
    if bg_pixels.size == 0:
        return bg_image.copy()

    mean = np.round(bg_pixels.mean(axis=0)).astype(np.uint8)
    out = bg_image.copy()
    out[fg] = mean
    return out


def fill_bg_local_mean(bg_image: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Fill holes with a local mean over nearby BG pixels (smooth, low-frequency).
    """
    fg = _fg_mask_bool(fg_mask)
    if not fg.any():
        return bg_image.copy()

    bg_pixels = bg_image[~fg]
    if bg_pixels.size == 0:
        return bg_image.copy()

    global_mean = bg_pixels.mean(axis=0).astype(np.float32)
    kernel = ensure_odd(_LOCAL_MEAN_KERNEL, 3)
    bg_mask = (~fg).astype(np.float32)
    img_f = bg_image.astype(np.float32)

    weighted = img_f * bg_mask[..., None]
    sum_bgr = cv2.boxFilter(weighted, -1, (kernel, kernel), normalize=False)
    count = cv2.boxFilter(bg_mask, -1, (kernel, kernel), normalize=False)
    zero_mask = count < 0.5
    count = np.maximum(count, 1.0)

    mean = sum_bgr / count[..., None]
    if zero_mask.any():
        mean[zero_mask] = global_mean
    out = bg_image.copy()
    out[fg] = np.clip(mean[fg], 0, 255).astype(np.uint8)
    return out


def fill_bg_diffusion(bg_image: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Diffusion fill: iterative blurring into holes for smooth backgrounds.
    """
    fg = _fg_mask_bool(fg_mask)
    if not fg.any():
        return bg_image.copy()

    out = fill_bg_flat_mean(bg_image, fg_mask)
    for _ in range(_DIFFUSION_ITERS):
        blurred = cv2.GaussianBlur(out, (0, 0), _DIFFUSION_SIGMA)
        out[fg] = blurred[fg]
    return out


def fill_bg_inpaint_telea(bg_image: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """
    Telea inpaint for holes (NOT recommended for production MRC pipelines).
    """
    fg = _fg_mask_bool(fg_mask)
    if not fg.any():
        return bg_image.copy()

    # Inpainting can introduce texture; avoid for production MRC.
    mask = fg.astype(np.uint8) * 255
    return cv2.inpaint(bg_image, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


BG_FILL_SPECS: List[BgFillSpec] = [
    BgFillSpec("FLAT_MEAN", "Flat mean (default)", fill_bg_flat_mean),
    BgFillSpec("LOCAL_MEAN", "Local mean", fill_bg_local_mean),
    BgFillSpec("DIFFUSION", "Diffusion", fill_bg_diffusion),
    BgFillSpec("INPAINT_TELEA", "Inpaint (Telea) - not for production MRC", fill_bg_inpaint_telea),
]

BG_FILL_BY_KEY: Dict[str, BgFillSpec] = {spec.key: spec for spec in BG_FILL_SPECS}
