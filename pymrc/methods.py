"""Mask generation methods and registry."""

from typing import Any, Dict, List

import cv2
import numpy as np

from .image_utils import clamp_u8, ensure_odd
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


def method_niblack(gray, p):
    win = ensure_odd(int(p["window"]), 3)
    k = float(p["k"])
    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    th = mean + k * std
    return (f < th).astype(np.uint8) * 255  # dark -> FG


def method_sobel_edges(gray, p):
    k = ensure_odd(int(p["kernel"]), 3)
    t = int(p["threshold"])
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(sx, sy)
    mag = clamp_u8(mag / (mag.max() + 1e-6) * 255.0)
    _, m = cv2.threshold(mag, t, 255, cv2.THRESH_BINARY)
    return m


def method_laplacian(gray, p):
    t = int(p["threshold"])
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap = clamp_u8(lap / (lap.max() + 1e-6) * 255.0)
    _, m = cv2.threshold(lap, t, 255, cv2.THRESH_BINARY)
    return m


def method_dog(gray, p):
    s1 = float(p["sigma1"])
    s2 = float(p["sigma2"])
    t = int(p["threshold"])
    g1 = cv2.GaussianBlur(gray, (0, 0), s1)
    g2 = cv2.GaussianBlur(gray, (0, 0), s2)
    dog = clamp_u8(cv2.absdiff(g1, g2))
    _, m = cv2.threshold(dog, t, 255, cv2.THRESH_BINARY)
    return m


def method_canny(gray, p):
    low = int(p["low"])
    high = int(p["high"])
    return cv2.Canny(gray, low, high)


def method_morph_gradient(gray, p):
    k = ensure_odd(int(p["kernel"]), 3)
    t = int(p["threshold"])
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se)
    _, m = cv2.threshold(grad, t, 255, cv2.THRESH_BINARY)
    return m


def method_top_percent_energy(gray, p):
    pct = float(p["percent"])
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap = lap / (lap.max() + 1e-6) * 255.0
    lap = clamp_u8(lap)
    cut = np.percentile(lap, 100.0 - pct)
    return (lap >= cut).astype(np.uint8) * 255


def method_low_variance_filled(gray, p):
    """
    Low-variance filled geometry detector.
    Produces FILLED uniform regions (not just borders).
    """
    win = ensure_odd(int(p["window"]), 3)
    std_th = float(p["stddev"])

    f = gray.astype(np.float32)
    mean = cv2.boxFilter(f, cv2.CV_32F, (win, win))
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (win, win))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))

    mask = (std <= std_th).astype(np.uint8) * 255  # uniform -> FG

    # Fill holes inside uniform components (binary hole filling)
    inv = cv2.bitwise_not(mask)
    flood = inv.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def method_mser_text(gray, p):
    delta = int(p["delta"])
    min_area = int(p["min_area"])
    max_area = int(p["max_area"])
    max_variation = float(p["max_variation"])
    min_diversity = float(p["min_diversity"])
    invert = int(p["invert"])

    img = 255 - gray if invert else gray
    mser = cv2.MSER_create(delta, min_area, max_area, max_variation, min_diversity)
    regions, _ = mser.detectRegions(img)

    mask = np.zeros_like(gray, dtype=np.uint8)
    if not regions:
        return mask

    for region in regions:
        poly = region.reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 255)
    return mask


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

    MethodSpec("Sobel edges → threshold", [
        ParamSpec("kernel", "Sobel kernel (odd)", "int", 3, 31, 3, 2),
        ParamSpec("threshold", "Edge threshold", "int", 0, 255, 40, 1),
    ], method_sobel_edges),

    MethodSpec("Laplacian magnitude → threshold", [
        ParamSpec("threshold", "Response threshold", "int", 0, 255, 35, 1),
    ], method_laplacian),

    MethodSpec("DoG → threshold", [
        ParamSpec("sigma1", "Sigma 1", "float", 0.3, 10.0, 1.0, 0.05, 100),
        ParamSpec("sigma2", "Sigma 2", "float", 0.3, 20.0, 2.0, 0.05, 100),
        ParamSpec("threshold", "Response threshold", "int", 0, 255, 20, 1),
    ], method_dog),

    MethodSpec("Canny edges", [
        ParamSpec("low", "Low threshold", "int", 0, 255, 40, 1),
        ParamSpec("high", "High threshold", "int", 0, 255, 120, 1),
    ], method_canny),

    MethodSpec("Morph gradient → threshold", [
        ParamSpec("kernel", "Kernel size (odd)", "int", 3, 31, 3, 2),
        ParamSpec("threshold", "Response threshold", "int", 0, 255, 30, 1),
    ], method_morph_gradient),

    MethodSpec("Top-percent energy (Laplacian)", [
        ParamSpec("percent", "Keep top (%)", "float", 0.1, 50.0, 5.0, 0.1, 10),
    ], method_top_percent_energy),

    MethodSpec("Low-variance filled geometry", [
        ParamSpec("window", "Window size (odd)", "int", 3, 101, 31, 2),
        ParamSpec("stddev", "StdDev threshold", "float", 0.5, 50.0, 6.0, 0.1, 10),
    ], method_low_variance_filled),

    MethodSpec("MSER text regions", [
        ParamSpec("delta", "Delta", "int", 1, 20, 5, 1),
        ParamSpec("min_area", "Min area", "int", 10, 10000, 60, 10),
        ParamSpec("max_area", "Max area", "int", 100, 200000, 8000, 100),
        ParamSpec("max_variation", "Max variation", "float", 0.1, 1.0, 0.25, 0.01, 100),
        ParamSpec("min_diversity", "Min diversity", "float", 0.0, 1.0, 0.2, 0.01, 100),
        ParamSpec("invert", "Invert (1=detect bright)", "int", 0, 1, 0, 1),
    ], method_mser_text),
]
