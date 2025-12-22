"""Image utility helpers for mask processing and Qt conversions."""

import numpy as np
from PySide6.QtGui import QImage


def ensure_odd(n: int, minv: int = 3) -> int:
    n = max(int(n), minv)
    return n if (n % 2 == 1) else (n + 1)


def clamp_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def qimage_from_gray(gray: np.ndarray) -> QImage:
    h, w = gray.shape
    return QImage(gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8)


def qimage_from_bgr(bgr: np.ndarray) -> QImage:
    h, w, _ = bgr.shape
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888)


def to_black_fg(mask_white_fg: np.ndarray) -> np.ndarray:
    """Convert mask where WHITE=FG to mask where BLACK=FG."""
    return (255 - mask_white_fg).astype(np.uint8)


def make_bg_image(color_bgr: np.ndarray, mask_black_fg: np.ndarray) -> np.ndarray:
    """
    BG bitmap:
      - keep original where mask is WHITE (background)
      - fill FG areas with white
    """
    bg = np.full_like(color_bgr, 255)
    bg[mask_black_fg != 0] = color_bgr[mask_black_fg != 0]
    return bg
