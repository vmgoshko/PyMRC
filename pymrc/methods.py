"""Mask generation methods and registry."""

from typing import Any, Dict, List

import cv2
import numpy as np

from .image_utils import ensure_odd
from .specs import MethodSpec, ParamSpec


# =========================================================
# Methods (ALL kept, unchanged)
# All methods return: uint8 mask WHITE=FG candidates, BLACK=BG
# =========================================================

def method_fixed_threshold(gray, p):
    # dark -> FG
    _, m = cv2.threshold(gray, int(p["threshold"]), 255, cv2.THRESH_BINARY_INV)
    return m


def method_otsu_dark(gray, _p):
    inv = 255 - gray
    _, m = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def method_adaptive_mean(gray, p):
    win = ensure_odd(int(p["window"]), 3)
    C = int(p["C"])
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win, C
    )


def method_adaptive_gaussian(gray, p):
    win = ensure_odd(int(p["window"]), 3)
    C = int(p["C"])
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, win, C
    )


def method_sauvola(gray, p):
    win = ensure_odd(int(p["window"]), 3)
    k = float(p["k"])
    R = float(p["R"])
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    th = mean * (1 + k * ((std / max(R, 1e-6)) - 1))
    return (f < th).astype(np.uint8) * 255  # dark -> FG


def auto_tune_sauvola(gray: np.ndarray) -> tuple[int, float]:
    """
    Heuristic auto-tuning for document scans (compression-oriented).
    Conservative text detection is preferred over clean background.
    """
    h, w = gray.shape[:2]
    base = int(round(min(h, w) / 30.0))
    base = max(21, min(81, base))
    window = ensure_odd(base, 21)
    if window > 81:
        window = 81

    std = float(gray.std())
    k = 0.25 + 0.20 * max(0.0, min(1.0, std / 64.0))
    # Slight positive bias to avoid losing thin strokes.
    k = min(0.45, k + 0.02)
    return window, k


def method_niblack(gray, p):
    win = ensure_odd(int(p["window"]), 3)
    k = float(p["k"])
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    th = mean + k * std
    return (f < th).astype(np.uint8) * 255  # dark -> FG


def method_canny(gray, p):
    low = int(p["low"])
    high = int(p["high"])
    return cv2.Canny(gray, low, high)


def detect_text_mask(gray: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """
    Text detector for MRC: conservative binarization is preferred.
    Uses Sauvola (manual/auto) or Adaptive Mean based on params.
    """
    text_mode = int(p.get("text_mode", 0))
    if text_mode == 1:
        win = ensure_odd(int(p.get("mean_window", 31)), 3)
        C = int(p.get("mean_C", 10))
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, win, C
        )

    if int(p.get("sauvola_auto", 0)) == 1:
        win, k = auto_tune_sauvola(gray)
    else:
        win = ensure_odd(int(p.get("window", 31)), 3)
        k = float(p.get("k", 0.2))
    R = float(p.get("R", 128.0))
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    th = mean * (1 + k * ((std / max(R, 1e-6)) - 1))
    return (f < th).astype(np.uint8) * 255  # dark -> FG


def detect_lineart_mask(gray: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    """
    Line-art detector for thin graphics and frames using classic CV.
    """
    low = int(p.get("line_low", 50))
    high = int(p.get("line_high", 150))
    edges = cv2.Canny(gray, low, high)

    k = ensure_odd(int(p.get("line_kernel", 3)), 1)
    if k > 1:
        horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
        edges = cv2.dilate(edges, horiz, iterations=1)
        edges = cv2.dilate(edges, vert, iterations=1)

    dilate_k = ensure_odd(int(p.get("line_dilate", 3)), 1)
    if dilate_k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        edges = cv2.dilate(edges, se, iterations=1)

    return (edges > 0).astype(np.uint8) * 255


def combine_fg_masks(text_mask: np.ndarray,
                     line_mask: np.ndarray,
                     p: Dict[str, Any]) -> np.ndarray:
    """
    Combine detectors for a stable FG mask (MRC compression oriented).
    """
    combined = cv2.bitwise_or(text_mask, line_mask)
    close_k = ensure_odd(int(p.get("close_kernel", 3)), 1)
    if close_k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, se, iterations=1)
    open_k = ensure_odd(int(p.get("open_kernel", 1)), 1)
    if open_k > 1:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, se, iterations=1)
    return combined


def method_text_lineart(gray, p):
    """
    Combined text + line-art foreground detection for MRC compression.
    Uses separate detectors and merges them conservatively.
    """
    text_mask = detect_text_mask(gray, p)
    line_mask = detect_lineart_mask(gray, p)
    return combine_fg_masks(text_mask, line_mask, p)


# =========================================================
# Method registry (ALL kept)
# =========================================================

METHODS: List[MethodSpec] = [
    MethodSpec("Fixed threshold (dark → mask)", [
        ParamSpec("threshold", "Threshold", "int", 0, 255, 160, 1)
    ], method_fixed_threshold),

    MethodSpec("Otsu (dark → mask)", [], method_otsu_dark),

    MethodSpec("Adaptive Mean (dark → mask)", [
        ParamSpec("window", "Window size (odd)", "int", 3, 101, 31, 2),
        ParamSpec("C", "C (bias)", "int", -50, 50, 10, 1)
    ], method_adaptive_mean),

    MethodSpec("Adaptive Gaussian (dark → mask)", [
        ParamSpec("window", "Window size (odd)", "int", 3, 101, 31, 2),
        ParamSpec("C", "C (bias)", "int", -50, 50, 10, 1)
    ], method_adaptive_gaussian),

    MethodSpec("Sauvola (dark → mask)", [
        ParamSpec("window", "Window size (odd)", "int", 3, 101, 31, 2),
        ParamSpec("k", "k", "float", -1.0, 1.0, 0.2, 0.01, 100),
        ParamSpec("R", "R (dynamic range)", "float", 1.0, 255.0, 128.0, 1.0, 10),
    ], method_sauvola),

    MethodSpec("Niblack (dark → mask)", [
        ParamSpec("window", "Window size (odd)", "int", 3, 101, 31, 2),
        ParamSpec("k", "k", "float", -1.0, 1.0, -0.2, 0.01, 100),
    ], method_niblack),

    MethodSpec("Canny edges", [
        ParamSpec("low", "Low threshold", "int", 0, 255, 40, 1),
        ParamSpec("high", "High threshold", "int", 0, 255, 120, 1),
    ], method_canny),

    MethodSpec("Text + Lineart (MRC)", [
        ParamSpec("text_mode", "Text mode (0=Sauvola,1=AdaptiveMean)", "int", 0, 1, 0, 1),
        ParamSpec("sauvola_auto", "Sauvola auto (1=auto)", "int", 0, 1, 1, 1),
        ParamSpec("window", "Sauvola window (odd)", "int", 3, 101, 31, 2),
        ParamSpec("k", "Sauvola k", "float", 0.05, 0.5, 0.25, 0.01, 100),
        ParamSpec("R", "Sauvola R", "float", 1.0, 255.0, 128.0, 1.0, 10),
        ParamSpec("mean_window", "Adaptive mean window (odd)", "int", 3, 101, 31, 2),
        ParamSpec("mean_C", "Adaptive mean C", "int", -50, 50, 10, 1),
        ParamSpec("line_low", "Lineart Canny low", "int", 0, 255, 50, 1),
        ParamSpec("line_high", "Lineart Canny high", "int", 0, 255, 150, 1),
        ParamSpec("line_kernel", "Line emphasis kernel", "int", 1, 31, 3, 2),
        ParamSpec("line_dilate", "Line dilate", "int", 1, 31, 3, 2),
        ParamSpec("close_kernel", "Combine close kernel", "int", 1, 15, 3, 2),
        ParamSpec("open_kernel", "Combine open kernel", "int", 1, 15, 1, 2),
    ], method_text_lineart),
]
