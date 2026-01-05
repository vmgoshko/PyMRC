"""Qt GUI for exploring mask candidates and saving results."""

import sys
import os
import tempfile
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PIL import Image

from .bg_fill import BG_FILL_BY_KEY, BG_FILL_SPECS, BgFillSpec
from .image_utils import (
    ensure_odd,
    make_bg_image,
    qimage_from_bgr,
    qimage_from_gray,
    to_black_fg,
)
from .methods import METHODS
from .save_utils import (
    save_mask_jbig2_via_gs,
    save_mask_tiff_g4,
    save_mask_tiff_g4_white_fg,
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
        self.mask_black_fg_raw: Optional[np.ndarray] = None
        self.fg_bgr_raw: Optional[np.ndarray] = None
        self.bg_bgr_raw: Optional[np.ndarray] = None

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

        self.save_group = QGroupBox("Layer compression (MRC-style)")
        sgf = QFormLayout(self.save_group)

        self.fg_format_combo = QComboBox()
        self.fg_format_combo.addItem("JPEG", "JPEG")
        self.fg_format_combo.addItem("PNG", "PNG")
        self.fg_jpeg_label = QLabel("FG JPEG quality")
        self.fg_jpeg_quality = QSpinBox()
        self.fg_jpeg_quality.setRange(1, 100)
        self.fg_jpeg_quality.setValue(40)
        self.fg_png_label = QLabel("FG PNG level")
        self.fg_png_level = QSpinBox()
        self.fg_png_level.setRange(0, 9)
        self.fg_png_level.setValue(3)
        self.fg_size_label = QLabel("—")

        self.bg_format_combo = QComboBox()
        self.bg_format_combo.addItem("JPEG", "JPEG")
        self.bg_format_combo.addItem("PNG", "PNG")
        self.bg_jpeg_label = QLabel("BG JPEG quality")
        self.bg_jpeg_quality = QSpinBox()
        self.bg_jpeg_quality.setRange(1, 100)
        self.bg_jpeg_quality.setValue(40)
        self.bg_png_label = QLabel("BG PNG level")
        self.bg_png_level = QSpinBox()
        self.bg_png_level.setRange(0, 9)
        self.bg_png_level.setValue(3)
        self.bg_size_label = QLabel("—")

        self.mask_format_combo = QComboBox()
        self.mask_format_combo.addItem("TIFF G4", "TIFF_G4")
        self.mask_size_label = QLabel("—")

        for combo in (self.fg_format_combo, self.bg_format_combo, self.mask_format_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)

        sgf.addRow("FG format", self.fg_format_combo)
        sgf.addRow(self.fg_jpeg_label, self.fg_jpeg_quality)
        sgf.addRow(self.fg_png_label, self.fg_png_level)
        sgf.addRow("FG size", self.fg_size_label)

        sgf.addRow("BG format", self.bg_format_combo)
        sgf.addRow(self.bg_jpeg_label, self.bg_jpeg_quality)
        sgf.addRow(self.bg_png_label, self.bg_png_level)
        sgf.addRow("BG size", self.bg_size_label)

        sgf.addRow("Mask format", self.mask_format_combo)
        sgf.addRow("Mask size", self.mask_size_label)

        self.bg_fill_group = QGroupBox("BG hole filling")
        bgf = QFormLayout(self.bg_fill_group)
        self.bg_fill_combo = QComboBox()
        self.bg_fill_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.bg_fill_combo.setMinimumContentsLength(12)
        for spec in BG_FILL_SPECS:
            self.bg_fill_combo.addItem(spec.label, spec.key)
        bgf.addRow("Fill method", self.bg_fill_combo)
        self.mask_post_group = QGroupBox("Mask post-processing")
        mpf = QFormLayout(self.mask_post_group)
        self.mask_dilate_size = QSpinBox()
        self.mask_dilate_size.setRange(1, 51)
        self.mask_dilate_size.setValue(3)
        self.mask_dilate_iter = QSpinBox()
        self.mask_dilate_iter.setRange(1, 10)
        self.mask_dilate_iter.setValue(1)
        mpf.addRow("Kernel size", self.mask_dilate_size)
        mpf.addRow("Iterations", self.mask_dilate_iter)


        self.btn_open = QPushButton("Open image…")
        self.btn_save_mask = QPushButton("Save mask")
        self.btn_save_mask_jbig2 = QPushButton("Save mask (JBIG2)…")
        self.btn_save_mask_g4 = QPushButton("Save mask (TIFF G4)…")
        self.btn_save_fg = QPushButton("Save FG")
        self.btn_save_fg_jpeg = QPushButton("Save FG (JPEG)…")
        self.btn_save_bg = QPushButton("Save BG")
        self.btn_reconstruct = QPushButton("Reconstruct image")
        self.btn_save_recon = QPushButton("Save reconstructed (PNG).")

        self.btn_save_mask.setEnabled(False)
        self.btn_save_mask_jbig2.setEnabled(False)
        self.btn_save_fg.setEnabled(False)
        self.btn_save_fg_jpeg.setEnabled(False)
        self.btn_save_bg.setEnabled(False)
        self.btn_save_mask_g4.setEnabled(False)
        self.btn_reconstruct.setEnabled(False)
        self.btn_save_recon.setEnabled(False)
        self.btn_save_mask_jbig2.setVisible(False)
        self.btn_save_fg_jpeg.setVisible(False)
        self.btn_save_mask_g4.setVisible(False)

        # =================================================
        # FG color unification
        # =================================================
        self.fg_color_group = QGroupBox("FG color mode")
        fgf = QFormLayout(self.fg_color_group)

        self.fg_color_mode = QComboBox()
        self.fg_color_mode.addItems([
            "Median FG color",
            "Mean FG color",
            "Forced black",
            "Block colors",
            "Posterized blocks"
        ])

        self.fg_block_size = QSpinBox()
        self.fg_block_size.setRange(2, 128)
        self.fg_block_size.setValue(8)
        self.fg_block_levels = QSpinBox()
        self.fg_block_levels.setRange(2, 32)
        self.fg_block_levels.setValue(4)

        fgf.addRow("Color mode", self.fg_color_mode)
        fgf.addRow("Block size", self.fg_block_size)
        fgf.addRow("Luma levels", self.fg_block_levels)

        # =================================================
        # Panels (order + visibility) + zoom
        # =================================================
        self.zoom_widget = QWidget()
        zf = QFormLayout(self.zoom_widget)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(5, 800)
        self.zoom_slider.setValue(75)
        self.zoom_value = QLabel("75%")
        zf.addRow(self.zoom_slider)
        zf.addRow("Value", self.zoom_value)

        self.order_group = QGroupBox("Panels")
        of = QFormLayout(self.order_group)
        self.panel_order = QListWidget()
        for name in ("Source", "Mask", "FG", "BG", "Reconstructed"):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.panel_order.addItem(item)
        self.btn_order_up = QPushButton("Move up")
        self.btn_order_down = QPushButton("Move down")
        of.addRow(self.panel_order)
        of.addRow(self.btn_order_up)
        of.addRow(self.btn_order_down)
        of.addRow("Zoom", self.zoom_widget)

        # views
        self.src_view = QLabel()
        self.mask_view = QLabel()
        self.fg_view = QLabel()
        self.bg_view = QLabel()
        self.recon_view = QLabel()
        for v in (self.src_view, self.mask_view, self.fg_view, self.bg_view, self.recon_view):
            v.setAlignment(Qt.AlignCenter)
            v.setMinimumSize(280, 360)

        self._scroll_areas: list[QScrollArea] = []
        self._scrollbar_map = {}
        self._scroll_pos = {"x": 0, "y": 0}
        self._syncing_scroll = False

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
            self._scroll_areas.append(scroll)
            return panel

        self.src_panel = make_view_panel("Source", self.src_view)
        self.mask_panel = make_view_panel("Mask (black = FG)", self.mask_view)
        self.fg_panel = make_view_panel("FG (original pixels under mask)", self.fg_view)
        self.bg_panel = make_view_panel("BG (original pixels outside mask)", self.bg_view)
        self.recon_panel = make_view_panel("Reconstructed", self.recon_view)

        self.panel_by_name = {
            "Source": self.src_panel,
            "Mask": self.mask_panel,
            "FG": self.fg_panel,
            "BG": self.bg_panel,
            "Reconstructed": self.recon_panel,
        }
        # layout left controls in scroll
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(10)

        self.top_group = QGroupBox("Candidate mask method")
        tf = QFormLayout(self.top_group)
        tf.addRow("Method", self.method_combo)

        cl.addWidget(self.btn_open)
        cl.addWidget(self.top_group)
        cl.addWidget(self.params_group)
        cl.addWidget(self.fg_color_group)
        cl.addWidget(self.order_group)
        cl.addWidget(self.mask_post_group)
        cl.addWidget(self.bg_fill_group)
        cl.addWidget(self.save_group)
        cl.addWidget(self.btn_save_mask)
        cl.addWidget(self.btn_save_fg)
        cl.addWidget(self.btn_save_bg)
        cl.addWidget(self.btn_reconstruct)
        cl.addWidget(self.btn_save_recon)
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(ctrl)

        self.views_container = QWidget()
        self.views_layout = QGridLayout(self.views_container)
        self.views_layout.setContentsMargins(0, 0, 0, 0)
        self._rebuild_views_layout()

        main = QHBoxLayout(self)
        main.addWidget(scroll, 1)
        main.addWidget(self.views_container, 4)

        # signals
        self.btn_open.clicked.connect(self.open_image)
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_save_mask_jbig2.clicked.connect(self.save_mask_jbig2)
        self.btn_save_fg.clicked.connect(self.save_fg)
        self.btn_save_fg_jpeg.clicked.connect(self.save_fg_jpeg)
        self.btn_save_bg.clicked.connect(self.save_bg)
        self.btn_save_mask_g4.clicked.connect(self.save_mask_g4)
        self.btn_reconstruct.clicked.connect(self.reconstruct_image)
        self.btn_save_recon.clicked.connect(self.save_reconstructed)

        self.method_combo.currentIndexChanged.connect(self.rebuild_params)
        self.fg_format_combo.currentIndexChanged.connect(self._update_format_controls)
        self.bg_format_combo.currentIndexChanged.connect(self._update_format_controls)
        self.mask_format_combo.currentIndexChanged.connect(self.update_results)
        self.fg_jpeg_quality.valueChanged.connect(self.update_results)
        self.fg_png_level.valueChanged.connect(self.update_results)
        self.bg_jpeg_quality.valueChanged.connect(self.update_results)
        self.bg_png_level.valueChanged.connect(self.update_results)
        self.fg_color_group.toggled.connect(self.update_results)
        self.fg_color_mode.currentIndexChanged.connect(self.update_results)
        self.fg_block_size.valueChanged.connect(self.update_results)
        self.fg_block_levels.valueChanged.connect(self.update_results)
        self.mask_post_group.toggled.connect(self.update_results)
        self.mask_dilate_size.valueChanged.connect(self.update_results)
        self.mask_dilate_iter.valueChanged.connect(self.update_results)
        self.bg_fill_group.toggled.connect(self._on_bg_fill_toggle)
        self.bg_fill_combo.currentIndexChanged.connect(self.update_results)
        self.panel_order.itemChanged.connect(self.update_results)
        self.zoom_slider.valueChanged.connect(self._on_zoom_change)
        self.btn_order_up.clicked.connect(self._move_panel_up)
        self.btn_order_down.clicked.connect(self._move_panel_down)
        self._wire_scroll_sync()

        self.fg_color_group.setChecked(False)
        self.mask_post_group.setChecked(False)
        self.bg_fill_group.setChecked(True)
        self.save_group.setChecked(True)
        self.save_group.toggled.connect(self.update_results)

        for group in (
            self.top_group,
            self.params_group,
            self.fg_color_group,
            self.order_group,
            self.mask_post_group,
            self.bg_fill_group,
            self.save_group,
        ):
            self._make_group_collapsible(group)

        self._update_format_controls()
        self._on_bg_fill_toggle()
        self.rebuild_params()

    def _make_group_collapsible(self, group: QGroupBox) -> None:
        group.setCheckable(True)
        layout = group.layout()
        if layout is None:
            return

        def set_items_visible(visible: bool) -> None:
            if isinstance(layout, QFormLayout):
                for row in range(layout.rowCount()):
                    for role in (
                        QFormLayout.LabelRole,
                        QFormLayout.FieldRole,
                        QFormLayout.SpanningRole,
                    ):
                        item = layout.itemAt(row, role)
                        if item is None:
                            continue
                        widget = item.widget()
                        if widget is not None:
                            widget.setVisible(visible)
            else:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is None:
                        continue
                    widget = item.widget()
                    if widget is not None:
                        widget.setVisible(visible)

        set_items_visible(group.isChecked())
        group.toggled.connect(set_items_visible)

    def _set_row_visible(self, label: QLabel, widget: QWidget, visible: bool) -> None:
        label.setVisible(visible)
        widget.setVisible(visible)

    def _update_format_controls(self) -> None:
        fg_is_jpeg = self.fg_format_combo.currentData() == "JPEG"
        bg_is_jpeg = self.bg_format_combo.currentData() == "JPEG"
        self._set_row_visible(self.fg_jpeg_label, self.fg_jpeg_quality, fg_is_jpeg)
        self._set_row_visible(self.fg_png_label, self.fg_png_level, not fg_is_jpeg)
        self._set_row_visible(self.bg_jpeg_label, self.bg_jpeg_quality, bg_is_jpeg)
        self._set_row_visible(self.bg_png_label, self.bg_png_level, not bg_is_jpeg)
        self.update_results()

    def _wire_scroll_sync(self) -> None:
        for scroll in self._scroll_areas:
            vbar = scroll.verticalScrollBar()
            hbar = scroll.horizontalScrollBar()
            self._scrollbar_map[vbar] = (scroll, True)
            self._scrollbar_map[hbar] = (scroll, False)
            vbar.valueChanged.connect(self._on_scrollbar_changed)
            hbar.valueChanged.connect(self._on_scrollbar_changed)

    def _on_scrollbar_changed(self, value: int) -> None:
        if self._syncing_scroll:
            return
        sender = self.sender()
        info = self._scrollbar_map.get(sender)
        if info is None:
            return
        source, vertical = info
        self._sync_scroll(source, value, vertical=vertical)

    def _sync_scroll(self, source: QScrollArea, value: int, *, vertical: bool) -> None:
        key = "y" if vertical else "x"
        self._scroll_pos[key] = int(value)
        self._apply_scroll_pos_axis(exclude=source, vertical=vertical)

    def _apply_scroll_pos_axis(self, *, exclude: Optional[QScrollArea], vertical: bool) -> None:
        key = "y" if vertical else "x"
        target = self._scroll_pos[key]
        self._syncing_scroll = True
        try:
            for scroll in self._scroll_areas:
                if scroll is exclude:
                    continue
                bar = scroll.verticalScrollBar() if vertical else scroll.horizontalScrollBar()
                value = max(bar.minimum(), min(bar.maximum(), target))
                if bar.value() == value:
                    continue
                bar.setValue(value)
        finally:
            self._syncing_scroll = False

    def current_method(self) -> MethodSpec:
        return METHODS[self.method_combo.currentIndex()]

    def current_bg_fill(self) -> BgFillSpec:
        key = self.bg_fill_combo.currentData()
        return BG_FILL_BY_KEY.get(key, BG_FILL_SPECS[0])

    def _on_bg_fill_toggle(self) -> None:
        enabled = self.bg_fill_group.isChecked()
        self.bg_fill_combo.setEnabled(enabled)
        self.update_results()

    def _format_size_text(self, size_bytes: Optional[int]) -> str:
        if size_bytes is None:
            return "-"
        return f"{size_bytes / 1024.0:.1f} KB"

    def _encode_decode_color(self,
                             bgr: np.ndarray,
                             fmt: str,
                             jpeg_quality: int,
                             png_level: int) -> tuple[Optional[np.ndarray], Optional[int]]:
        with tempfile.TemporaryDirectory() as tmp:
            if fmt == "JPEG":
                ext = ".jpg"
                params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            else:
                ext = ".png"
                params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_level)]
            path = os.path.join(tmp, f"layer{ext}")
            ok = cv2.imwrite(path, bgr, params)
            if not ok:
                return None, None
            size = os.path.getsize(path)
            decoded = cv2.imread(path, cv2.IMREAD_COLOR)
            return decoded, size

    def _encode_decode_mask(self, mask_black_fg: np.ndarray) -> tuple[Optional[np.ndarray], Optional[int]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mask.tif")
            # TIFF expects inverted convention: black=BG, white=FG.
            mask_white_fg = (255 - mask_black_fg).astype(np.uint8)
            save_mask_tiff_g4_white_fg(mask_white_fg, path)
            size = os.path.getsize(path)
            img = Image.open(path).convert("L")
            arr = np.array(img)
            mask_white_fg = (arr > 127).astype(np.uint8) * 255
            mask_black_fg_rt = (255 - mask_white_fg).astype(np.uint8)
            return mask_black_fg_rt, size

    def _apply_runtime_compression(self) -> None:
        if not self.save_group.isChecked():
            self.fg_bgr = self.fg_bgr_raw
            self.bg_bgr = self.bg_bgr_raw
            self.mask_black_fg = self.mask_black_fg_raw
            self.fg_size_label.setText("-")
            self.bg_size_label.setText("-")
            self.mask_size_label.setText("-")
            return

        fg_size = None
        bg_size = None
        mask_size = None

        if self.fg_bgr_raw is not None:
            fmt = self.fg_format_combo.currentData()
            decoded, fg_size = self._encode_decode_color(
                self.fg_bgr_raw,
                fmt,
                int(self.fg_jpeg_quality.value()),
                int(self.fg_png_level.value()),
            )
            if self.save_group.isChecked() and decoded is not None:
                self.fg_bgr = decoded
            else:
                self.fg_bgr = self.fg_bgr_raw
        else:
            self.fg_bgr = None

        if self.bg_bgr_raw is not None:
            fmt = self.bg_format_combo.currentData()
            decoded, bg_size = self._encode_decode_color(
                self.bg_bgr_raw,
                fmt,
                int(self.bg_jpeg_quality.value()),
                int(self.bg_png_level.value()),
            )
            if self.save_group.isChecked() and decoded is not None:
                self.bg_bgr = decoded
            else:
                self.bg_bgr = self.bg_bgr_raw
        else:
            self.bg_bgr = None

        if self.mask_black_fg_raw is not None:
            decoded, mask_size = self._encode_decode_mask(self.mask_black_fg_raw)
            if self.save_group.isChecked() and decoded is not None:
                self.mask_black_fg = decoded
            else:
                self.mask_black_fg = self.mask_black_fg_raw
        else:
            self.mask_black_fg = None

        self.fg_size_label.setText(self._format_size_text(fg_size))
        self.bg_size_label.setText(self._format_size_text(bg_size))
        self.mask_size_label.setText(self._format_size_text(mask_size))

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
        self.btn_save_fg.setEnabled(True)
        self.btn_save_bg.setEnabled(True)
        self.btn_reconstruct.setEnabled(True)
        self.btn_save_recon.setEnabled(True)

        self.update_results()

    def gather_params(self) -> Dict[str, Any]:
        return {k: w.value() for k, w in self.param_widgets.items()}

    def update_results(self):
        self._apply_panel_visibility()
        if self.gray is None or self.color is None:
            self.fg_size_label.setText("-")
            self.bg_size_label.setText("-")
            self.mask_size_label.setText("-")
            return

        try:
            params = self.gather_params()
            method = self.current_method()

            show_src = self._panel_is_enabled("Source")
            show_mask = self._panel_is_enabled("Mask")
            show_fg = self._panel_is_enabled("FG")
            show_bg = self._panel_is_enabled("BG")
            show_recon = self._panel_is_enabled("Reconstructed")

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

            mask_black_fg_raw = None
            if need_mask:
                mask_white = method.fn(self.gray, params)
                mask_white = (mask_white > 0).astype(np.uint8) * 255
                if self.mask_post_group.isChecked():
                    k = ensure_odd(int(self.mask_dilate_size.value()), 1)
                    it = int(self.mask_dilate_iter.value())
                    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                    mask_white = cv2.dilate(mask_white, se, iterations=it)
                mask_black_fg_raw = to_black_fg(mask_white)
            else:
                self.mask_black_fg = None
                self.mask_black_fg_raw = None
                self._mask_pix = None
                self.mask_view.clear()

            fg_raw = None
            if need_fg:
                if mask_black_fg_raw is None:
                    return
                if self.fg_color_group.isChecked():
                    mode = self.fg_color_mode.currentText()

                    if mode == "Block colors":
                        fg_raw = self.make_fg_block_colored(
                            self.color,
                            mask_black_fg_raw
                        )
                    elif mode == "Posterized blocks":
                        fg_raw = self.make_fg_posterized_blocks(
                            self.color,
                            mask_black_fg_raw,
                            block=int(self.fg_block_size.value()),
                            levels=int(self.fg_block_levels.value())
                        )
                    else:
                        fg_raw = self.make_fg_unified(
                            self.color,
                            mask_black_fg_raw,
                            mode
                        )
                else:
                    # FG must preserve edges; do not blur, inpaint, or smooth.
                    # ORIGINAL FG: keep original pixels under mask
                    fg = np.full_like(self.color, 255)
                    fg[mask_black_fg_raw == 0] = self.color[mask_black_fg_raw == 0]
                    fg_raw = fg
            else:
                self.fg_bgr = None
                self.fg_bgr_raw = None
                self._fg_pix = None
                self.fg_view.clear()

            bg_raw = None
            if need_bg:
                if mask_black_fg_raw is None:
                    return
                # BG bitmap: keep original pixels where mask is WHITE, fill FG holes
                if self.bg_fill_group.isChecked():
                    # BG should remain smooth/low-frequency for compression efficiency.
                    fg_mask = (mask_black_fg_raw == 0).astype(np.uint8) * 255
                    fill = self.current_bg_fill()
                    bg_raw = fill.fn(self.color, fg_mask)
                else:
                    bg_raw = make_bg_image(self.color, mask_black_fg_raw)
            else:
                self.bg_bgr = None
                self.bg_bgr_raw = None
                self._bg_pix = None
                self.bg_view.clear()

            if need_mask:
                self.mask_black_fg_raw = mask_black_fg_raw
            if need_fg:
                self.fg_bgr_raw = fg_raw
            if need_bg:
                self.bg_bgr_raw = bg_raw
            if need_mask or need_fg or need_bg:
                self._apply_runtime_compression()

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
            self.src_view.setFixedSize(scaled.size())
        if self._mask_pix is not None:
            scaled = scale_pix(self._mask_pix)
            self.mask_view.setPixmap(scaled)
            self.mask_view.setFixedSize(scaled.size())
        if self._fg_pix is not None:
            scaled = scale_pix(self._fg_pix)
            self.fg_view.setPixmap(scaled)
            self.fg_view.setFixedSize(scaled.size())
        if self._bg_pix is not None:
            scaled = scale_pix(self._bg_pix)
            self.bg_view.setPixmap(scaled)
            self.bg_view.setFixedSize(scaled.size())
        if self._recon_pix is not None:
            scaled = scale_pix(self._recon_pix)
            self.recon_view.setPixmap(scaled)
            self.recon_view.setFixedSize(scaled.size())
        self._apply_scroll_pos_axis(exclude=None, vertical=True)
        self._apply_scroll_pos_axis(exclude=None, vertical=False)

    def resizeEvent(self, event):  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        self._rescale_views()

    def _apply_panel_visibility(self):
        self.src_panel.setVisible(self._panel_is_enabled("Source"))
        self.mask_panel.setVisible(self._panel_is_enabled("Mask"))
        self.fg_panel.setVisible(self._panel_is_enabled("FG"))
        self.bg_panel.setVisible(self._panel_is_enabled("BG"))
        self.recon_panel.setVisible(self._panel_is_enabled("Reconstructed"))
        self.btn_reconstruct.setEnabled(
            self._panel_is_enabled("Reconstructed") and self.color is not None
        )
        self._rebuild_views_layout()

    def _panel_is_enabled(self, name: str) -> bool:
        for i in range(self.panel_order.count()):
            item = self.panel_order.item(i)
            if item is not None and item.text() == name:
                return item.checkState() == Qt.Checked
        return False

    def _panel_order_names(self) -> list[str]:
        return [self.panel_order.item(i).text() for i in range(self.panel_order.count())]

    def _ordered_visible_panels(self) -> list[QWidget]:
        panels: list[QWidget] = []
        for name in self._panel_order_names():
            panel = self.panel_by_name.get(name)
            if panel is None:
                continue
            if self._panel_is_enabled(name):
                panels.append(panel)
        return panels

    def _rebuild_views_layout(self):
        while self.views_layout.count():
            item = self.views_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self.views_layout.removeWidget(widget)

        panels = self._ordered_visible_panels()
        count = len(panels)
        if count == 0:
            return

        cols = 1 if count == 1 else 2
        row = 0
        col = 0
        for panel in panels:
            self.views_layout.addWidget(panel, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1

        for r in range(row + 1):
            self.views_layout.setRowStretch(r, 1)
        for c in range(cols):
            self.views_layout.setColumnStretch(c, 1)

    def _move_panel_up(self):
        row = self.panel_order.currentRow()
        if row <= 0:
            return
        item = self.panel_order.takeItem(row)
        self.panel_order.insertItem(row - 1, item)
        self.panel_order.setCurrentRow(row - 1)
        self._rebuild_views_layout()
        self._rescale_views()

    def _move_panel_down(self):
        row = self.panel_order.currentRow()
        if row < 0 or row >= self.panel_order.count() - 1:
            return
        item = self.panel_order.takeItem(row)
        self.panel_order.insertItem(row + 1, item)
        self.panel_order.setCurrentRow(row + 1)
        self._rebuild_views_layout()
        self._rescale_views()

    def save_mask(self):
        src = self.mask_black_fg_raw if self.mask_black_fg_raw is not None else self.mask_black_fg
        if src is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask (TIFF G4)", "mask.tif", "TIFF (*.tif *.tiff)"
        )
        if path:
            inverted = (255 - src).astype(np.uint8)
            save_mask_tiff_g4_white_fg(inverted, path)

    def save_mask_jbig2(self):
        src = self.mask_black_fg_raw if self.mask_black_fg_raw is not None else self.mask_black_fg
        if src is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask (JBIG2)", "mask.jb2", "JBIG2 (*.jb2)"
        )
        if path:
            save_mask_jbig2_via_gs(src, path)

    def save_fg(self):
        src = self.fg_bgr_raw if self.fg_bgr_raw is not None else self.fg_bgr
        if src is None:
            return
        fmt = self.fg_format_combo.currentData()
        if fmt == "JPEG":
            default_name = "fg.jpg"
            flt = "JPEG (*.jpg *.jpeg)"
            params = [
                cv2.IMWRITE_JPEG_QUALITY, int(self.fg_jpeg_quality.value()),
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            ]
        else:
            default_name = "fg.png"
            flt = "PNG (*.png)"
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.fg_png_level.value())]
        path, _ = QFileDialog.getSaveFileName(self, "Save FG", default_name, flt)
        if path:
            cv2.imwrite(path, src, params)

    def save_bg(self):
        src = self.bg_bgr_raw if self.bg_bgr_raw is not None else self.bg_bgr
        if src is None:
            return
        fmt = self.bg_format_combo.currentData()
        if fmt == "JPEG":
            default_name = "bg.jpg"
            flt = "JPEG (*.jpg *.jpeg)"
            params = [
                cv2.IMWRITE_JPEG_QUALITY, int(self.bg_jpeg_quality.value()),
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            ]
        else:
            default_name = "bg.png"
            flt = "PNG (*.png)"
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.bg_png_level.value())]
        path, _ = QFileDialog.getSaveFileName(self, "Save BG", default_name, flt)
        if path:
            cv2.imwrite(path, src, params)

    def save_fg_jpeg(self):
        src = self.fg_bgr_raw if self.fg_bgr_raw is not None else self.fg_bgr
        if src is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save FG (JPEG)", "fg.jpg", "JPEG (*.jpg *.jpeg)"
        )
        if path:
            cv2.imwrite(
                path,
                src,
                [
                    cv2.IMWRITE_JPEG_QUALITY, int(self.fg_jpeg_quality.value()),
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                ],
            )

    def save_mask_g4(self):
        src = self.mask_black_fg_raw if self.mask_black_fg_raw is not None else self.mask_black_fg
        if src is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save mask (TIFF G4)",
            "mask.tif",
            "TIFF (*.tif *.tiff)"
        )
        if path:
            inverted = (255 - src).astype(np.uint8)
            save_mask_tiff_g4_white_fg(inverted, path)

    def save_reconstructed(self):
        if self.recon_bgr is None:
            QMessageBox.warning(self, "Save reconstructed", "Reconstruct image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save reconstructed (PNG)", "reconstructed.png", "PNG (*.png)"
        )
        if path:
            cv2.imwrite(path, self.recon_bgr)

    def _zoom_factor(self) -> float:
        return float(self.zoom_slider.value()) / 100.0

    def _on_zoom_change(self):
        self.zoom_value.setText(f"{self.zoom_slider.value()}%")
        self._rescale_views()

    def reconstruct_image(self):
        if not self._panel_is_enabled("Reconstructed"):
            return
        if self.mask_black_fg is None or self.fg_bgr is None or self.bg_bgr is None:
            QMessageBox.warning(self, "Reconstruct", "Open an image and compute FG/BG first.")
            return

        if self.bg_bgr.shape != self.fg_bgr.shape:
            QMessageBox.warning(self, "Reconstruct", "FG and BG sizes differ.")
            return
        # Mask must align pixel-perfectly with FG/BG.
        if self.bg_bgr.shape[:2] != self.mask_black_fg.shape:
            QMessageBox.warning(self, "Reconstruct", "Mask size differs from FG/BG.")
            return

        # Reconstruction uses BLACK (0) = FG, WHITE (255) = BG.
        fg_pixels = (self.mask_black_fg == 0)

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
                            min_area: int = 10) -> np.ndarray:
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

    def make_fg_posterized_blocks(self,
                                  color_bgr: np.ndarray,
                                  mask_black_fg: np.ndarray,
                                  block: int = 8,
                                  levels: int = 4) -> np.ndarray:
        """
        Create FG from block rectangles with quantized luma (posterized).
        """
        fg = np.full_like(color_bgr, 255)
        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            return fg

        block = max(2, int(block))
        levels = max(2, int(levels))
        step = 255.0 / float(levels - 1)

        ycrcb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2YCrCb)
        h, w = fg_mask.shape

        for y in range(0, h, block):
            y2 = min(h, y + block)
            for x in range(0, w, block):
                x2 = min(w, x + block)
                block_mask = fg_mask[y:y2, x:x2]
                if not block_mask.any():
                    continue

                pixels = ycrcb[y:y2, x:x2][block_mask]
                mean = pixels.mean(axis=0)
                qy = int(round(mean[0] / step) * step)
                qy = int(max(0, min(255, qy)))

                qycrcb = np.array(
                    [[[qy, mean[1], mean[2]]]],
                    dtype=np.float32
                )
                qycrcb_u8 = np.clip(qycrcb, 0, 255).astype(np.uint8)
                q_bgr = cv2.cvtColor(qycrcb_u8, cv2.COLOR_YCrCb2BGR)[0, 0]

                fg_block = fg[y:y2, x:x2]
                fg_block[block_mask] = q_bgr

        return fg

def main():
    app = QApplication(sys.argv)
    w = MRCTool()
    w.resize(1800, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
