"""Qt GUI for exploring mask candidates and saving results."""

import sys
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from .image_utils import make_bg_image, qimage_from_bgr, qimage_from_gray, to_black_fg, ensure_odd
from .methods import METHODS
from .save_utils import (
    save_bg_jpeg,
    save_fg_jpeg,
    save_mask_jbig2_via_gs,
    save_mask_tiff_g4,
)
from .specs import MethodSpec
from .widgets import LabeledSlider


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

    def resizeEvent(self, event):  # noqa: N802 (Qt signature)
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
