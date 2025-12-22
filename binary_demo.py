# mrc_text_mask_gui.py
# Method-centric MRC candidate mask explorer + FG preview (+ BG preview)
#
# Output mask convention:
#   BLACK (0)  = goes to text/FG mask
#   WHITE (255)= background
#
# FG preview:
#   FG image = original pixels where mask is BLACK, elsewhere white
#
# BG preview:
#   BG image = original pixels where mask is WHITE, FG areas filled with white
#
# Added:
#   1) Save BG (JPEG, MRC-style aggressive)
#   2) Save Mask (JBIG2) via Ghostscript (Windows)
#
# Install:
#   python -m pip install PySide6 opencv-python numpy
#
# Ghostscript:
#   Needs gswin64c.exe installed. Update GS_EXE path below to your install.
#
# Run:
#   python mrc_text_mask_gui.py

import sys
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
from PIL import Image


import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QCheckBox, QFileDialog,
    QPushButton, QMessageBox, QScrollArea, QSpinBox, QDoubleSpinBox, QSlider
)

# =========================================================
# Ghostscript path (Windows, console version)
# =========================================================

GS_EXE = r"C:\Program Files\gs\gs10.06.0\bin\gswin64c.exe"

# =========================================================
# Utils
# =========================================================

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

# =========================================================
# Saving helpers
# =========================================================

def save_bg_jpeg(bg_bgr: np.ndarray, out_path: str, quality: int = 40):
    # MRC-style: BG can be aggressively compressed because sharp stuff is in FG
    cv2.imwrite(
        out_path,
        bg_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    )

def save_fg_jpeg(fg_bgr: np.ndarray, out_path: str, quality: int = 40):
    # MRC-style: BG can be aggressively compressed because sharp stuff is in FG
    cv2.imwrite(
        out_path,
        fg_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    )

def save_mask_jbig2_via_gs(mask_black_fg: np.ndarray, out_path: str):
    """
    Save binary mask as JBIG2 using Ghostscript (Windows).
    Uses PBM (P4) intermediate.
    BLACK (0) = FG => '1' bit in PBM (black).
    """
    if not os.path.exists(GS_EXE):
        raise RuntimeError(
            "Ghostscript not found. Update GS_EXE path to gswin64c.exe.\n"
            f"Current: {GS_EXE}"
        )

    bilevel = (mask_black_fg == 0).astype(np.uint8)  # 1 for FG/black

    with tempfile.TemporaryDirectory() as tmp:
        pbm_path = os.path.join(tmp, "mask.pbm")

        h, w = bilevel.shape
        with open(pbm_path, "wb") as f:
            f.write(f"P4\n{w} {h}\n".encode("ascii"))
            packed = np.packbits(bilevel, axis=1)
            f.write(packed.tobytes())

        cmd = [
            GS_EXE,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=jbig2",
            f"-sOutputFile={out_path}",
            pbm_path
        ]
        subprocess.check_call(cmd)
        
def save_mask_tiff_g4(mask_black_fg: np.ndarray, out_path: str):
    """
    Save binary mask as TIFF CCITT Group 4 (1-bit, lossless).
    BLACK (0) = FG
    """
    # Convert to 1-bit image: FG=black(0), BG=white(255)
    bilevel = (mask_black_fg == 0).astype(np.uint8) * 255

    img = Image.fromarray(bilevel, mode="L").convert("1")

    img.save(
        out_path,
        format="TIFF",
        compression="group4"
    )


# =========================================================
# Param / Method specs
# =========================================================

@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str          # "int" | "float"
    minv: float
    maxv: float
    default: float
    step: float = 1.0
    scale: int = 100   # float slider scale (value = int/scale)

@dataclass
class MethodSpec:
    name: str
    params: List[ParamSpec]
    fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]  # returns WHITE=FG mask

class LabeledSlider(QWidget):
    """Slider + SpinBox synced. Always shows slider + box. Label is provided by QFormLayout."""
    def __init__(self, spec: ParamSpec, on_change):
        super().__init__()
        self.spec = spec
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1)

        if spec.kind == "int":
            self.spin = QSpinBox()
            self.spin.setRange(int(spec.minv), int(spec.maxv))
            self.spin.setSingleStep(int(spec.step))
            self.spin.setValue(int(spec.default))

            self.slider.setRange(int(spec.minv), int(spec.maxv))
            self.slider.setValue(int(spec.default))

            self.slider.valueChanged.connect(self.spin.setValue)
            self.spin.valueChanged.connect(self.slider.setValue)
            self.spin.valueChanged.connect(lambda _: on_change())
        else:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(4)
            self.spin.setRange(float(spec.minv), float(spec.maxv))
            self.spin.setSingleStep(float(spec.step))
            self.spin.setValue(float(spec.default))

            smin = int(round(spec.minv * spec.scale))
            smax = int(round(spec.maxv * spec.scale))
            sval = int(round(spec.default * spec.scale))
            self.slider.setRange(smin, smax)
            self.slider.setValue(sval)

            self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / spec.scale))
            self.spin.valueChanged.connect(lambda v: self.slider.setValue(int(round(v * spec.scale))))
            self.spin.valueChanged.connect(lambda _: on_change())

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 4)
        lay.addWidget(self.spin, 1)

    def value(self):
        return self.spin.value()

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
]

# =========================================================
# GUI
# =========================================================

class MRCTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRC explorer — Mask + FG + BG (black=FG)")

        # images
        self.gray: Optional[np.ndarray] = None
        self.color: Optional[np.ndarray] = None

        # outputs
        self.mask_black_fg: Optional[np.ndarray] = None
        self.fg_bgr: Optional[np.ndarray] = None
        self.bg_bgr: Optional[np.ndarray] = None

        # rendered pixmaps cache (avoid recompute on resize)
        self._mask_pix: Optional[QPixmap] = None
        self._fg_pix: Optional[QPixmap] = None
        self._bg_pix: Optional[QPixmap] = None

        # controls
        self.method_combo = QComboBox()
        for m in METHODS:
            self.method_combo.addItem(m.name)

        self.params_group = QGroupBox("Method parameters")
        self.params_form = QFormLayout(self.params_group)
        self.param_widgets: Dict[str, LabeledSlider] = {}

        self.refine_group = QGroupBox("Refinement (optional)")
        rf = QFormLayout(self.refine_group)
        self.cb_open = QCheckBox("Morph open (remove noise)")
        self.cb_close = QCheckBox("Morph close (connect strokes)")
        self.cb_dilate = QCheckBox("Dilate strokes")
        self.refine_kernel = QSpinBox()
        self.refine_kernel.setRange(1, 31)
        self.refine_kernel.setValue(3)
        rf.addRow(self.cb_open)
        rf.addRow(self.cb_close)
        rf.addRow(self.cb_dilate)
        rf.addRow("Kernel size (odd)", self.refine_kernel)

        self.bg_group = QGroupBox("BG save settings (MRC-style)")
        bgf = QFormLayout(self.bg_group)
        self.bg_quality = QSpinBox()
        self.bg_quality.setRange(1, 100)
        self.bg_quality.setValue(40)
        bgf.addRow("JPEG quality", self.bg_quality)
        
        self.fg_group = QGroupBox("FG save settings (MRC-style)")
        bgf = QFormLayout(self.fg_group)
        self.fg_quality = QSpinBox()
        self.fg_quality.setRange(1, 100)
        self.fg_quality.setValue(40)
        bgf.addRow("JPEG quality", self.fg_quality)
            
        self.btn_open = QPushButton("Open image…")
        self.btn_save_mask = QPushButton("Save mask (PNG/BMP)…")
        self.btn_save_mask_jbig2 = QPushButton("Save mask (JBIG2)…")
        self.btn_save_mask_g4 = QPushButton("Save mask (TIFF G4)…")
        self.btn_save_fg = QPushButton("Save FG (PNG/BMP)…")
        self.btn_save_fg_jpeg = QPushButton("Save FG (JPEG)…")
        self.btn_save_bg = QPushButton("Save BG (JPEG)…")

        self.btn_save_mask.setEnabled(False)
        self.btn_save_mask_jbig2.setEnabled(False)
        self.btn_save_fg.setEnabled(False)
        self.btn_save_fg_jpeg.setEnabled(False)
        self.btn_save_bg.setEnabled(False)
        self.btn_save_mask_g4.setEnabled(False)

        # =================================================
        # FG color unification
        # =================================================
        self.fg_color_group = QGroupBox("FG color mode")
        fgf = QFormLayout(self.fg_color_group)

        self.cb_fg_unified = QCheckBox("Use unified FG color")
        self.cb_fg_unified.setChecked(False)

        self.fg_color_mode = QComboBox()
        self.fg_color_mode.addItems([
            "Median FG color",
            "Mean FG color",
            "Forced black",
            "Block colors"
        ])


        fgf.addRow(self.cb_fg_unified)
        fgf.addRow("Color mode", self.fg_color_mode)
        
        # views
        self.mask_view = QLabel("Mask (black = FG)")
        self.fg_view = QLabel("FG (original pixels under mask)")
        self.bg_view = QLabel("BG (original pixels outside mask)")
        for v in (self.mask_view, self.fg_view, self.bg_view):
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(520, 640)

        # layout left controls in scroll
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        top_group = QGroupBox("Candidate mask method")
        tf = QFormLayout(top_group)
        tf.addRow("Method", self.method_combo)

        cl.addWidget(self.btn_open)
        cl.addWidget(top_group)
        cl.addWidget(self.params_group)
        cl.addWidget(self.refine_group)
        cl.addWidget(self.bg_group)
        cl.addWidget(self.fg_group)
        cl.addWidget(self.fg_color_group)
        cl.addWidget(self.btn_save_mask)
        cl.addWidget(self.btn_save_mask_jbig2)
        cl.addWidget(self.btn_save_fg)
        cl.addWidget(self.btn_save_fg_jpeg)
        cl.addWidget(self.btn_save_bg)
        cl.addWidget(self.btn_save_mask_g4)
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(ctrl)

        views = QHBoxLayout()
        views.addWidget(self.mask_view)
        views.addWidget(self.fg_view)
        views.addWidget(self.bg_view)

        main = QHBoxLayout(self)
        main.addWidget(scroll, 1)
        main.addLayout(views, 4)

        # signals
        self.btn_open.clicked.connect(self.open_image)
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_save_mask_jbig2.clicked.connect(self.save_mask_jbig2)
        self.btn_save_fg.clicked.connect(self.save_fg)
        self.btn_save_fg_jpeg.clicked.connect(self.save_fg_jpeg)
        self.btn_save_bg.clicked.connect(self.save_bg)
        self.btn_save_mask_g4.clicked.connect(self.save_mask_g4)

        self.method_combo.currentIndexChanged.connect(self.rebuild_params)
        self.cb_open.stateChanged.connect(self.update_results)
        self.cb_close.stateChanged.connect(self.update_results)
        self.cb_dilate.stateChanged.connect(self.update_results)
        self.refine_kernel.valueChanged.connect(self.update_results)
        self.bg_quality.valueChanged.connect(self.update_results)
        self.fg_quality.valueChanged.connect(self.update_results)
        self.cb_fg_unified.stateChanged.connect(self.update_results)
        self.fg_color_mode.currentIndexChanged.connect(self.update_results)


        self.rebuild_params()

    def current_method(self) -> MethodSpec:
        return METHODS[self.method_combo.currentIndex()]

    def rebuild_params(self):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.param_widgets.clear()

        m = self.current_method()
        if not m.params:
            self.params_form.addRow(QLabel("No parameters for this method."))
        else:
            for ps in m.params:
                w = LabeledSlider(ps, self.update_results)
                self.param_widgets[ps.key] = w
                self.params_form.addRow(ps.label, w)

        self.update_results()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        color = cv2.imread(path, cv2.IMREAD_COLOR)
        if color is None:
            QMessageBox.critical(self, "Error", "Cannot load image.")
            return

        self.color = color
        self.gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        self.btn_save_mask.setEnabled(True)
        self.btn_save_mask_jbig2.setEnabled(True)
        self.btn_save_fg.setEnabled(True)
        self.btn_save_fg_jpeg.setEnabled(True)
        self.btn_save_bg.setEnabled(True)
        self.btn_save_mask_g4.setEnabled(True)

        self.update_results()

    def gather_params(self) -> Dict[str, Any]:
        return {k: w.value() for k, w in self.param_widgets.items()}

    def refine_white_fg(self, mask_white_fg: np.ndarray) -> np.ndarray:
        k = ensure_odd(int(self.refine_kernel.value()), 1)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        m = mask_white_fg
        if self.cb_open.isChecked():
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
        if self.cb_close.isChecked():
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
        if self.cb_dilate.isChecked():
            m = cv2.dilate(m, se, iterations=1)
        return m

    def update_results(self):
        if self.gray is None or self.color is None:
            return

        try:
            params = self.gather_params()
            method = self.current_method()

            mask_white = method.fn(self.gray, params)
            mask_white = (mask_white > 0).astype(np.uint8) * 255
            mask_white = self.refine_white_fg(mask_white)

            self.mask_black_fg = to_black_fg(mask_white)

            # FG bitmap
            if self.cb_fg_unified.isChecked():
                mode = self.fg_color_mode.currentText()

                if mode == "Block colors":
                    self.fg_bgr = self.make_fg_block_colored(
                        self.color,
                        self.mask_black_fg
                    )
                else:
                    self.fg_bgr = self.make_fg_unified(
                        self.color,
                        self.mask_black_fg,
                        mode
                    )
            else:
                # ORIGINAL FG: keep original pixels under mask
                fg = np.full_like(self.color, 255)
                fg[self.mask_black_fg == 0] = self.color[self.mask_black_fg == 0]
                self.fg_bgr = fg

            # BG bitmap: keep original pixels where mask is WHITE, FG filled with white
            self.bg_bgr = make_bg_image(self.color, self.mask_black_fg)

            # render pixmaps (scaled later in resize)
            if self.mask_black_fg is not None:
                self._mask_pix = QPixmap.fromImage(
                    qimage_from_gray(self.mask_black_fg)
                )

            if self.fg_bgr is not None:
                self._fg_pix = QPixmap.fromImage(
                    qimage_from_bgr(self.fg_bgr)
                )

            if self.bg_bgr is not None:
                self._bg_pix = QPixmap.fromImage(
                    qimage_from_bgr(self.bg_bgr)
                )


            self._rescale_views()

        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))

    def _rescale_views(self):
        if self._mask_pix is not None:
            self.mask_view.setPixmap(self._mask_pix.scaled(
                self.mask_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        if self._fg_pix is not None:
            self.fg_view.setPixmap(self._fg_pix.scaled(
                self.fg_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
        if self._bg_pix is not None:
            self.bg_view.setPixmap(self._bg_pix.scaled(
                self.bg_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale_views()

    def save_mask(self):
        if self.mask_black_fg is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask", "mask.png", "PNG (*.png);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, self.mask_black_fg)

    def save_mask_jbig2(self):
        if self.mask_black_fg is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask (JBIG2)", "mask.jb2", "JBIG2 (*.jb2)"
        )
        if path:
            save_mask_jbig2_via_gs(self.mask_black_fg, path)

    def save_fg(self):
        if self.fg_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FG", "fg.png", "PNG (*.png);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, self.fg_bgr)

    def save_bg(self):
        if self.bg_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save BG (JPEG)", "bg.jpg", "JPEG (*.jpg *.jpeg)"
        )
        if path:
            save_bg_jpeg(self.bg_bgr, path, quality=int(self.bg_quality.value()))

    def save_fg_jpeg(self):
        if self.fg_bgr is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save BG (JPEG)", "fg.jpg", "JPEG (*.jpg *.jpeg)"
        )
        if path:
            save_fg_jpeg(self.fg_bgr, path, quality=int(self.fg_quality.value()))
            
    def save_mask_g4(self):
        if self.mask_black_fg is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save mask (TIFF G4)",
            "mask.tif",
            "TIFF (*.tif *.tiff)"
        )
        if path:
            save_mask_tiff_g4(self.mask_black_fg, path)


    def make_fg_unified(self,
                        color_bgr: np.ndarray,
                        mask_black_fg: np.ndarray,
                        mode: str) -> np.ndarray:
        """
        Create FG image with unified color.
        FG pixels are BLACK (0) in mask_black_fg.
        """
        fg = np.full_like(color_bgr, 255)

        fg_pixels = color_bgr[mask_black_fg == 0]
        if fg_pixels.size == 0:
            return fg

        if mode == "Forced black":
            fg_color = np.array([0, 0, 0], dtype=np.uint8)

        elif mode == "Mean FG color":
            fg_color = fg_pixels.mean(axis=0).astype(np.uint8)

        else:  # Median FG color
            fg_color = np.median(fg_pixels, axis=0).astype(np.uint8)

        fg[mask_black_fg == 0] = fg_color
        return fg

    def make_fg_block_colored(self,
                            color_bgr: np.ndarray,
                            mask_black_fg: np.ndarray,
                            min_area: int = 20) -> np.ndarray:
        """
        Create FG where each connected text component
        is filled with its own representative color.
        """
        fg = np.full_like(color_bgr, 255)

        fg_mask = (mask_black_fg == 0).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )

        for i in range(1, num):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            comp_mask = (labels == i)

            pixels = color_bgr[comp_mask]
            if pixels.size == 0:
                continue

            # robust for scanned text
            color = np.median(pixels, axis=0).astype(np.uint8)

            fg[comp_mask] = color

        return fg

def main():
    app = QApplication(sys.argv)
    w = MRCTool()
    w.resize(2000, 900)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
