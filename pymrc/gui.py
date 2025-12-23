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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
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
        self.recon_bgr: Optional[np.ndarray] = None

        # rendered pixmaps cache (avoid recompute on resize)
        self._mask_pix: Optional[QPixmap] = None
        self._fg_pix: Optional[QPixmap] = None
        self._bg_pix: Optional[QPixmap] = None
        self._recon_pix: Optional[QPixmap] = None
        self._src_pix: Optional[QPixmap] = None

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
        self.btn_reconstruct = QPushButton("Reconstruct image")

        self.btn_save_mask.setEnabled(False)
        self.btn_save_mask_jbig2.setEnabled(False)
        self.btn_save_fg.setEnabled(False)
        self.btn_save_fg_jpeg.setEnabled(False)
        self.btn_save_bg.setEnabled(False)
        self.btn_save_mask_g4.setEnabled(False)
        self.btn_reconstruct.setEnabled(False)

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

        # =================================================
        # Panel toggles + zoom
        # =================================================
        self.panel_group = QGroupBox("Panels")
        pf = QFormLayout(self.panel_group)
        self.cb_show_src = QCheckBox("Source")
        self.cb_show_mask = QCheckBox("Mask")
        self.cb_show_fg = QCheckBox("FG")
        self.cb_show_bg = QCheckBox("BG")
        self.cb_show_recon = QCheckBox("Reconstructed")
        for cb in (
            self.cb_show_src,
            self.cb_show_mask,
            self.cb_show_fg,
            self.cb_show_bg,
            self.cb_show_recon,
        ):
            cb.setChecked(True)
        pf.addRow(self.cb_show_src)
        pf.addRow(self.cb_show_mask)
        pf.addRow(self.cb_show_fg)
        pf.addRow(self.cb_show_bg)
        pf.addRow(self.cb_show_recon)

        self.zoom_group = QGroupBox("Zoom")
        zf = QFormLayout(self.zoom_group)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(5, 800)
        self.zoom_slider.setValue(75)
        self.zoom_value = QLabel("75%")
        zf.addRow(self.zoom_slider)
        zf.addRow("Value", self.zoom_value)

        # views
        self.src_view = QLabel()
        self.mask_view = QLabel()
        self.fg_view = QLabel()
        self.bg_view = QLabel()
        self.recon_view = QLabel()
        for v in (self.src_view, self.mask_view, self.fg_view, self.bg_view, self.recon_view):
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(280, 360)

        def make_view_panel(title: str, view: QLabel) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            scroll.setWidget(view)
            layout.addWidget(scroll, 1)
            return panel

        self.src_panel = make_view_panel("Source", self.src_view)
        self.mask_panel = make_view_panel("Mask (black = FG)", self.mask_view)
        self.fg_panel = make_view_panel("FG (original pixels under mask)", self.fg_view)
        self.bg_panel = make_view_panel("BG (original pixels outside mask)", self.bg_view)
        self.recon_panel = make_view_panel("Reconstructed", self.recon_view)

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
        cl.addWidget(self.panel_group)
        cl.addWidget(self.zoom_group)
        cl.addWidget(self.btn_save_mask)
        cl.addWidget(self.btn_save_mask_jbig2)
        cl.addWidget(self.btn_save_fg)
        cl.addWidget(self.btn_save_fg_jpeg)
        cl.addWidget(self.btn_save_bg)
        cl.addWidget(self.btn_save_mask_g4)
        cl.addWidget(self.btn_reconstruct)
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(ctrl)

        views = QGridLayout()
        views.addWidget(self.src_panel, 0, 0)
        views.addWidget(self.mask_panel, 0, 1)
        views.addWidget(self.fg_panel, 0, 2)
        views.addWidget(self.bg_panel, 1, 0)
        views.addWidget(self.recon_panel, 1, 1)
        views.setRowStretch(0, 1)
        views.setRowStretch(1, 1)
        views.setColumnStretch(0, 1)
        views.setColumnStretch(1, 1)
        views.setColumnStretch(2, 1)

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
        self.btn_reconstruct.clicked.connect(self.reconstruct_image)

        self.method_combo.currentIndexChanged.connect(self.rebuild_params)
        self.cb_open.stateChanged.connect(self.update_results)
        self.cb_close.stateChanged.connect(self.update_results)
        self.cb_dilate.stateChanged.connect(self.update_results)
        self.refine_kernel.valueChanged.connect(self.update_results)
        self.bg_quality.valueChanged.connect(self.update_results)
        self.fg_quality.valueChanged.connect(self.update_results)
        self.cb_fg_unified.stateChanged.connect(self.update_results)
        self.fg_color_mode.currentIndexChanged.connect(self.update_results)
        self.cb_show_src.stateChanged.connect(self.update_results)
        self.cb_show_mask.stateChanged.connect(self.update_results)
        self.cb_show_fg.stateChanged.connect(self.update_results)
        self.cb_show_bg.stateChanged.connect(self.update_results)
        self.cb_show_recon.stateChanged.connect(self.update_results)
        self.zoom_slider.valueChanged.connect(self._on_zoom_change)

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
        self.recon_bgr = None
        self._recon_pix = None
        self.recon_view.clear()

        self.btn_save_mask.setEnabled(True)
        self.btn_save_mask_jbig2.setEnabled(True)
        self.btn_save_fg.setEnabled(True)
        self.btn_save_fg_jpeg.setEnabled(True)
        self.btn_save_bg.setEnabled(True)
        self.btn_save_mask_g4.setEnabled(True)
        self.btn_reconstruct.setEnabled(True)

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
        self._apply_panel_visibility()
        if self.gray is None or self.color is None:
            return

        try:
            params = self.gather_params()
            method = self.current_method()

            show_src = self.cb_show_src.isChecked()
            show_mask = self.cb_show_mask.isChecked()
            show_fg = self.cb_show_fg.isChecked()
            show_bg = self.cb_show_bg.isChecked()
            show_recon = self.cb_show_recon.isChecked()

            if show_src:
                self._src_pix = QPixmap.fromImage(qimage_from_bgr(self.color))
            else:
                self._src_pix = None
                self.src_view.clear()

            self.recon_bgr = None
            self._recon_pix = None
            self.recon_view.clear()

            need_mask = show_mask or show_fg or show_bg or show_recon
            need_fg = show_fg or show_recon
            need_bg = show_bg or show_recon

            if need_mask:
                mask_white = method.fn(self.gray, params)
                mask_white = (mask_white > 0).astype(np.uint8) * 255
                mask_white = self.refine_white_fg(mask_white)
                self.mask_black_fg = to_black_fg(mask_white)
            else:
                self.mask_black_fg = None
                self._mask_pix = None
                self.mask_view.clear()

            if need_fg:
                if self.mask_black_fg is None:
                    return
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
            else:
                self.fg_bgr = None
                self._fg_pix = None
                self.fg_view.clear()

            if need_bg:
                if self.mask_black_fg is None:
                    return
                # BG bitmap: keep original pixels where mask is WHITE, FG filled with white
                self.bg_bgr = make_bg_image(self.color, self.mask_black_fg)
            else:
                self.bg_bgr = None
                self._bg_pix = None
                self.bg_view.clear()

            # render pixmaps (scaled later in resize)
            if show_mask and self.mask_black_fg is not None:
                self._mask_pix = QPixmap.fromImage(
                    qimage_from_gray(self.mask_black_fg)
                )
            elif not show_mask:
                self._mask_pix = None
                self.mask_view.clear()

            if show_fg and self.fg_bgr is not None:
                self._fg_pix = QPixmap.fromImage(
                    qimage_from_bgr(self.fg_bgr)
                )
            elif not show_fg:
                self._fg_pix = None
                self.fg_view.clear()

            if show_bg and self.bg_bgr is not None:
                self._bg_pix = QPixmap.fromImage(
                    qimage_from_bgr(self.bg_bgr)
                )
            elif not show_bg:
                self._bg_pix = None
                self.bg_view.clear()

            self._rescale_views()

        except Exception as e:
            QMessageBox.critical(self, "Processing error", str(e))

    def _rescale_views(self):
        def scale_pix(pix: QPixmap) -> QPixmap:
            factor = self._zoom_factor()
            width = max(1, int(pix.width() * factor))
            height = max(1, int(pix.height() * factor))
            return pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self._src_pix is not None:
            scaled = scale_pix(self._src_pix)
            self.src_view.setPixmap(scaled)
            self.src_view.resize(scaled.size())
        if self._mask_pix is not None:
            scaled = scale_pix(self._mask_pix)
            self.mask_view.setPixmap(scaled)
            self.mask_view.resize(scaled.size())
        if self._fg_pix is not None:
            scaled = scale_pix(self._fg_pix)
            self.fg_view.setPixmap(scaled)
            self.fg_view.resize(scaled.size())
        if self._bg_pix is not None:
            scaled = scale_pix(self._bg_pix)
            self.bg_view.setPixmap(scaled)
            self.bg_view.resize(scaled.size())
        if self._recon_pix is not None:
            scaled = scale_pix(self._recon_pix)
            self.recon_view.setPixmap(scaled)
            self.recon_view.resize(scaled.size())

    def resizeEvent(self, event):  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        self._rescale_views()

    def _apply_panel_visibility(self):
        self.src_panel.setVisible(self.cb_show_src.isChecked())
        self.mask_panel.setVisible(self.cb_show_mask.isChecked())
        self.fg_panel.setVisible(self.cb_show_fg.isChecked())
        self.bg_panel.setVisible(self.cb_show_bg.isChecked())
        self.recon_panel.setVisible(self.cb_show_recon.isChecked())
        self.btn_reconstruct.setEnabled(
            self.cb_show_recon.isChecked() and self.color is not None
        )

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

    def _zoom_factor(self) -> float:
        return float(self.zoom_slider.value()) / 100.0

    def _on_zoom_change(self):
        self.zoom_value.setText(f"{self.zoom_slider.value()}%")
        self._rescale_views()

    def reconstruct_image(self):
        if not self.cb_show_recon.isChecked():
            return
        if self.mask_black_fg is None or self.fg_bgr is None or self.bg_bgr is None:
            QMessageBox.warning(self, "Reconstruct", "Open an image and compute FG/BG first.")
            return

        if self.bg_bgr.shape != self.fg_bgr.shape:
            QMessageBox.warning(self, "Reconstruct", "FG and BG sizes differ.")
            return
        if self.bg_bgr.shape[:2] != self.mask_black_fg.shape:
            QMessageBox.warning(self, "Reconstruct", "Mask size differs from FG/BG.")
            return

        # Invert to match convention: WHITE (255) = FG
        mask_white_fg = (255 - self.mask_black_fg).astype(np.uint8)
        fg_pixels = (mask_white_fg == 255)

        recon = self.bg_bgr.copy()
        recon[fg_pixels] = self.fg_bgr[fg_pixels]
        self.recon_bgr = recon
        self._recon_pix = QPixmap.fromImage(qimage_from_bgr(self.recon_bgr))
        self._rescale_views()

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
        is represented by a solid colored rectangle.
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

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            fg[y:y + h, x:x + w] = color

        return fg


def main():
    app = QApplication(sys.argv)
    w = MRCTool()
    w.resize(1800, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
