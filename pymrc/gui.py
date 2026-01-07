"""Qt GUI for exploring mask candidates and saving results."""

import sys
import os
import tempfile
import time
import subprocess
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QToolButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
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
from .methods import METHODS, auto_tune_sauvola, method_sauvola
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
        self.setWindowTitle("MRC explorer - Mask + FG + BG (black=FG)")
        self._trace_enabled = True

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
        self._method_param_info: Dict[str, Dict[str, str]] = {}
        self.method_label = self._make_label_with_info(
            "Method",
            "Выбор алгоритма построения маски FG/BG."
        )

        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self.update_results)

        self.cb_sauvola_auto = QCheckBox("Auto")
        self.cb_sauvola_auto.setChecked(False)
        self.cb_sauvola_auto.stateChanged.connect(self.schedule_update)
        self.lbl_sauvola_auto = QLabel("-")

        self.params_group = QGroupBox("Method parameters")
        self.params_form = QFormLayout(self.params_group)
        self.param_widgets: Dict[str, LabeledSlider] = {}

        self.save_group = QGroupBox("Layer compression (MRC-style)")
        sgf = QFormLayout(self.save_group)

        self.fg_format_combo = QComboBox()
        self.fg_format_combo.addItem("JPEG", "JPEG")
        self.fg_format_combo.addItem("PNG", "PNG")
        self.fg_format_combo.addItem("JPEG2000", "JP2")
        self.fg_format_combo.addItem("JPEG2000 (Kakadu)", "JP2_KDU")
        self.fg_jpeg_label = QLabel("FG JPEG quality")
        self.fg_jpeg_quality = QSpinBox()
        self.fg_jpeg_quality.setRange(1, 100)
        self.fg_jpeg_quality.setValue(40)
        self.fg_png_label = QLabel("FG PNG level")
        self.fg_png_level = QSpinBox()
        self.fg_png_level.setRange(0, 9)
        self.fg_png_level.setValue(3)
        self.fg_jp2_label = QLabel("FG JP2 compression (0-100)")
        self.fg_jp2_quality = QSpinBox()
        self.fg_jp2_quality.setRange(0, 100)
        self.fg_jp2_quality.setValue(60)
        self.fg_size_label = QLabel("-")

        self.bg_format_combo = QComboBox()
        self.bg_format_combo.addItem("JPEG", "JPEG")
        self.bg_format_combo.addItem("PNG", "PNG")
        self.bg_format_combo.addItem("JPEG2000", "JP2")
        self.bg_jpeg_label = QLabel("BG JPEG quality")
        self.bg_jpeg_quality = QSpinBox()
        self.bg_jpeg_quality.setRange(1, 100)
        self.bg_jpeg_quality.setValue(40)
        self.bg_png_label = QLabel("BG PNG level")
        self.bg_png_level = QSpinBox()
        self.bg_png_level.setRange(0, 9)
        self.bg_png_level.setValue(3)
        self.bg_jp2_label = QLabel("BG JP2 compression (0-100)")
        self.bg_jp2_quality = QSpinBox()
        self.bg_jp2_quality.setRange(0, 100)
        self.bg_jp2_quality.setValue(70)
        self.bg_size_label = QLabel("-")

        self.mask_format_combo = QComboBox()
        self.mask_format_combo.addItem("TIFF G4", "TIFF_G4")
        self.mask_size_label = QLabel("—")

        for combo in (self.fg_format_combo, self.bg_format_combo, self.mask_format_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)

        sgf.addRow("FG format", self.fg_format_combo)
        sgf.addRow(self.fg_jpeg_label, self.fg_jpeg_quality)
        sgf.addRow(self.fg_png_label, self.fg_png_level)
        sgf.addRow(self.fg_jp2_label, self.fg_jp2_quality)
        sgf.addRow("FG size", self.fg_size_label)

        sgf.addRow("BG format", self.bg_format_combo)
        sgf.addRow(self.bg_jpeg_label, self.bg_jpeg_quality)
        sgf.addRow(self.bg_png_label, self.bg_png_level)
        sgf.addRow(self.bg_jp2_label, self.bg_jp2_quality)
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
        self.bg_fill_label = self._make_label_with_info(
            "Fill method",
            "Выбор способа заполнения дыр в BG; влияет на гладкость и артефакты фона."
        )
        bgf.addRow(self.bg_fill_label, self.bg_fill_combo)
        self.mask_post_group = QGroupBox("Mask post-processing")
        mpf = QFormLayout(self.mask_post_group)
        self.mask_dilate_size = QSpinBox()
        self.mask_dilate_size.setRange(1, 51)
        self.mask_dilate_size.setValue(3)
        self.mask_dilate_iter = QSpinBox()
        self.mask_dilate_iter.setRange(1, 10)
        self.mask_dilate_iter.setValue(1)
        self.mask_dilate_size_label = self._make_label_with_info(
            "Kernel size",
            "Размер ядра дилатации маски; больше расширяет FG."
        )
        self.mask_dilate_iter_label = self._make_label_with_info(
            "Iterations",
            "Количество итераций дилатации; усиливает эффект расширения."
        )
        mpf.addRow(self.mask_dilate_size_label, self.mask_dilate_size)
        mpf.addRow(self.mask_dilate_iter_label, self.mask_dilate_iter)

        self.preprocess_group = QGroupBox("Mask preprocessing")
        ppf = QFormLayout(self.preprocess_group)
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItem("None", "NONE")
        self.preprocess_combo.addItem("Gaussian blur", "GAUSSIAN")
        self.preprocess_combo.addItem("Median blur", "MEDIAN")
        self.preprocess_combo.addItem("Bilateral filter", "BILATERAL")
        self.preprocess_combo.addItem("Box blur", "BOX")
        self.preprocess_kernel = QSpinBox()
        self.preprocess_kernel.setRange(1, 51)
        self.preprocess_kernel.setValue(3)
        self.preprocess_sigma = QSpinBox()
        self.preprocess_sigma.setRange(0, 100)
        self.preprocess_sigma.setValue(0)
        self.preprocess_sigma_color = QSpinBox()
        self.preprocess_sigma_color.setRange(0, 200)
        self.preprocess_sigma_color.setValue(50)
        self.preprocess_sigma_space = QSpinBox()
        self.preprocess_sigma_space.setRange(0, 200)
        self.preprocess_sigma_space.setValue(50)
        self.preprocess_filter_label = self._make_label_with_info(
            "Filter",
            "Тип предварительного фильтра для маски; влияет на шум и сохранность краев."
        )
        self.preprocess_kernel_label = self._make_label_with_info(
            "Kernel size",
            "Размер окна фильтра; больше сглаживает, но размывает детали."
        )
        self.preprocess_sigma_label = self._make_label_with_info(
            "Sigma",
            "Сигма Гаусса для Gaussian blur; больше — сильнее размытие."
        )
        self.preprocess_sigma_color_label = self._make_label_with_info(
            "Sigma color",
            "Билатеральный фильтр: чувствительность к цветовым различиям."
        )
        self.preprocess_sigma_space_label = self._make_label_with_info(
            "Sigma space",
            "Билатеральный фильтр: радиус/влияние по расстоянию."
        )
        ppf.addRow(self.preprocess_filter_label, self.preprocess_combo)
        ppf.addRow(self.preprocess_kernel_label, self.preprocess_kernel)
        ppf.addRow(self.preprocess_sigma_label, self.preprocess_sigma)
        ppf.addRow(self.preprocess_sigma_color_label, self.preprocess_sigma_color)
        ppf.addRow(self.preprocess_sigma_space_label, self.preprocess_sigma_space)

        self.btn_open = QPushButton("Open image…")
        self.btn_save_mask = QPushButton("Save mask")
        self.btn_save_mask_jbig2 = QPushButton("Save mask (JBIG2)…")
        self.btn_save_mask_g4 = QPushButton("Save mask (TIFF G4)…")
        self.btn_save_mask_inverted_tiff = QPushButton("Save mask (TIFF, black text)…")
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
        self.btn_save_mask_inverted_tiff.setEnabled(False)
        self.btn_reconstruct.setEnabled(False)
        self.btn_save_recon.setEnabled(False)
        self.btn_save_mask_jbig2.setVisible(False)
        self.btn_save_fg_jpeg.setVisible(False)
        self.btn_save_mask_g4.setVisible(False)
        self.btn_save_mask_inverted_tiff.setVisible(True)

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
            "Posterized blocks",
            "FG_Posterized",
            "Palette quantization",
            "Palette stepped gradient",
            "FG_PosterizedShading",
        ])

        self.fg_color_mode_label = self._make_label_with_info(
            "Color mode",
            "Режим обработки FG: от простой унификации цвета до постеризации и шейдинга."
        )
        self.fg_block_size_label = self._make_label_with_info(
            "Block size",
            "Размер блока для постеризации; больше блоки — грубее ступени."
        )
        self.fg_block_size = QSpinBox()
        self.fg_block_size.setRange(2, 128)
        self.fg_block_size.setValue(8)
        self.fg_block_levels_label = self._make_label_with_info(
            "Luma levels",
            "Количество уровней яркости в блоках; меньше уровней — сильнее постеризация."
        )
        self.fg_block_levels = QSpinBox()
        self.fg_block_levels.setRange(2, 32)
        self.fg_block_levels.setValue(4)
        self.fg_palette_size_label = self._make_label_with_info(
            "Palette size",
            "Число цветов в палитре (k) для квантования FG."
        )
        self.fg_palette_size = QSpinBox()
        self.fg_palette_size.setRange(2, 64)
        self.fg_palette_size.setValue(8)

        self.fg_posterize_levels_label = self._make_label_with_info(
            "Posterize levels",
            "Количество уровней квантования для постеризации FG."
        )
        self.fg_posterize_levels = QSpinBox()
        self.fg_posterize_levels.setRange(2, 64)
        self.fg_posterize_levels.setValue(4)

        self.fg_stats_group = QGroupBox("FG stats")
        fg_stats_form = QFormLayout(self.fg_stats_group)
        self.fg_color_count_label = QLabel("-")
        fg_stats_form.addRow("Colors in FG", self.fg_color_count_label)

        self.fg_smooth_group = QGroupBox("FG edge-aware smoothing")
        fg_smooth_form = QFormLayout(self.fg_smooth_group)
        self.fg_edge_smooth_mode_label = self._make_label_with_info(
            "Smoothing",
            "Edge-preserving сглаживание FG для снижения шума в плоских областях."
        )
        self.fg_edge_smooth_mode = QComboBox()
        self.fg_edge_smooth_mode.addItem("Bilateral", "BILATERAL")
        self.fg_edge_smooth_mode.addItem("Guided", "GUIDED")
        self.fg_edge_bilateral_d_label = self._make_label_with_info(
            "Bilateral diameter",
            "Диаметр окна билатерального фильтра; больше — сильнее сглаживание."
        )
        self.fg_edge_bilateral_d = QSpinBox()
        self.fg_edge_bilateral_d.setRange(1, 31)
        self.fg_edge_bilateral_d.setValue(9)
        self.fg_edge_bilateral_sigma_color_label = self._make_label_with_info(
            "Bilateral sigma color",
            "Чувствительность к различиям цвета в билатеральном фильтре."
        )
        self.fg_edge_bilateral_sigma_color = QSpinBox()
        self.fg_edge_bilateral_sigma_color.setRange(1, 200)
        self.fg_edge_bilateral_sigma_color.setValue(60)
        self.fg_edge_bilateral_sigma_space_label = self._make_label_with_info(
            "Bilateral sigma space",
            "Влияние расстояния в билатеральном фильтре; больше — более широкое сглаживание."
        )
        self.fg_edge_bilateral_sigma_space = QSpinBox()
        self.fg_edge_bilateral_sigma_space.setRange(1, 200)
        self.fg_edge_bilateral_sigma_space.setValue(60)
        self.fg_edge_guided_radius_label = self._make_label_with_info(
            "Guided radius",
            "Радиус окна guided filter; больше — сильнее сглаживание при сохранении границ."
        )
        self.fg_edge_guided_radius = QSpinBox()
        self.fg_edge_guided_radius.setRange(1, 64)
        self.fg_edge_guided_radius.setValue(8)
        self.fg_edge_guided_eps_label = self._make_label_with_info(
            "Guided eps",
            "Регуляризация guided filter; больше — меньше деталей, стабильнее сглаживание."
        )
        self.fg_edge_guided_eps = QDoubleSpinBox()
        self.fg_edge_guided_eps.setRange(0.0001, 1.0)
        self.fg_edge_guided_eps.setDecimals(4)
        self.fg_edge_guided_eps.setSingleStep(0.001)
        self.fg_edge_guided_eps.setValue(0.01)
        fg_smooth_form.addRow(self.fg_edge_smooth_mode_label, self.fg_edge_smooth_mode)
        fg_smooth_form.addRow(self.fg_edge_bilateral_d_label, self.fg_edge_bilateral_d)
        fg_smooth_form.addRow(self.fg_edge_bilateral_sigma_color_label, self.fg_edge_bilateral_sigma_color)
        fg_smooth_form.addRow(self.fg_edge_bilateral_sigma_space_label, self.fg_edge_bilateral_sigma_space)
        fg_smooth_form.addRow(self.fg_edge_guided_radius_label, self.fg_edge_guided_radius)
        fg_smooth_form.addRow(self.fg_edge_guided_eps_label, self.fg_edge_guided_eps)

        self.fg_smooth_mode_label = self._make_label_with_info(
            "Smoothing",
            "Edge-preserving сглаживание перед постеризацией; снижает шум, сохраняя края."
        )
        self.fg_smooth_mode = QComboBox()
        self.fg_smooth_mode.addItem("Bilateral", "BILATERAL")
        self.fg_smooth_mode.addItem("Guided", "GUIDED")

        self.fg_bilateral_d_label = self._make_label_with_info(
            "Bilateral diameter",
            "Диаметр окна билатерального фильтра; больше — сильнее сглаживание."
        )
        self.fg_bilateral_d = QSpinBox()
        self.fg_bilateral_d.setRange(1, 31)
        self.fg_bilateral_d.setValue(9)
        self.fg_bilateral_sigma_color_label = self._make_label_with_info(
            "Bilateral sigma color",
            "Чувствительность к различиям цвета в билатеральном фильтре."
        )
        self.fg_bilateral_sigma_color = QSpinBox()
        self.fg_bilateral_sigma_color.setRange(1, 200)
        self.fg_bilateral_sigma_color.setValue(60)
        self.fg_bilateral_sigma_space_label = self._make_label_with_info(
            "Bilateral sigma space",
            "Влияние расстояния в билатеральном фильтре; больше — более широкое сглаживание."
        )
        self.fg_bilateral_sigma_space = QSpinBox()
        self.fg_bilateral_sigma_space.setRange(1, 200)
        self.fg_bilateral_sigma_space.setValue(60)

        self.fg_guided_radius_label = self._make_label_with_info(
            "Guided radius",
            "Радиус окна guided filter; больше — сильнее сглаживание при сохранении границ."
        )
        self.fg_guided_radius = QSpinBox()
        self.fg_guided_radius.setRange(1, 64)
        self.fg_guided_radius.setValue(8)
        self.fg_guided_eps_label = self._make_label_with_info(
            "Guided eps",
            "Регуляризация guided filter; больше — меньше деталей, стабильнее сглаживание."
        )
        self.fg_guided_eps = QDoubleSpinBox()
        self.fg_guided_eps.setRange(0.0001, 1.0)
        self.fg_guided_eps.setDecimals(4)
        self.fg_guided_eps.setSingleStep(0.001)
        self.fg_guided_eps.setValue(0.01)

        self.fg_quant_levels_label = self._make_label_with_info(
            "Quant levels",
            "Количество уровней яркости в постеризации (типично 6–12)."
        )
        self.fg_quant_levels = QSpinBox()
        self.fg_quant_levels.setRange(2, 32)
        self.fg_quant_levels.setValue(8)
        self.fg_quant_mode_label = self._make_label_with_info(
            "Quant mode",
            "Метод квантования яркости: равномерно или по кластерам."
        )
        self.fg_quant_mode = QComboBox()
        self.fg_quant_mode.addItem("Uniform", "UNIFORM")
        self.fg_quant_mode.addItem("K-means", "KMEANS")

        self.fg_post_blur_label = self._make_label_with_info(
            "Post blur",
            "Легкое размытие после квантования для смягчения ступеней."
        )
        self.fg_post_blur = QCheckBox("Enable")
        self.fg_post_blur_sigma_label = self._make_label_with_info(
            "Blur sigma",
            "Сила размытия после квантования; больше — мягче переходы."
        )
        self.fg_post_blur_sigma = QDoubleSpinBox()
        self.fg_post_blur_sigma.setRange(0.1, 2.0)
        self.fg_post_blur_sigma.setDecimals(2)
        self.fg_post_blur_sigma.setSingleStep(0.1)
        self.fg_post_blur_sigma.setValue(0.8)

        fgf.addRow(self.fg_color_mode_label, self.fg_color_mode)
        fgf.addRow(self.fg_block_size_label, self.fg_block_size)
        fgf.addRow(self.fg_block_levels_label, self.fg_block_levels)
        fgf.addRow(self.fg_palette_size_label, self.fg_palette_size)
        fgf.addRow(self.fg_posterize_levels_label, self.fg_posterize_levels)
        fgf.addRow(self.fg_smooth_mode_label, self.fg_smooth_mode)
        fgf.addRow(self.fg_bilateral_d_label, self.fg_bilateral_d)
        fgf.addRow(self.fg_bilateral_sigma_color_label, self.fg_bilateral_sigma_color)
        fgf.addRow(self.fg_bilateral_sigma_space_label, self.fg_bilateral_sigma_space)
        fgf.addRow(self.fg_guided_radius_label, self.fg_guided_radius)
        fgf.addRow(self.fg_guided_eps_label, self.fg_guided_eps)
        fgf.addRow(self.fg_quant_levels_label, self.fg_quant_levels)
        fgf.addRow(self.fg_quant_mode_label, self.fg_quant_mode)
        fgf.addRow(self.fg_post_blur_label, self.fg_post_blur)
        fgf.addRow(self.fg_post_blur_sigma_label, self.fg_post_blur_sigma)

        self._method_param_info = {
            "Fixed threshold (dark → mask)": {
                "threshold": "Порог яркости: ниже порога считается FG (темное).",
            },
            "Adaptive Mean (dark → mask)": {
                "window": "Размер локального окна порога; больше сглаживает фон, но теряет мелкие детали.",
                "C": "Смещение порога; больше C -> меньше FG (жестче).",
            },
            "Adaptive Gaussian (dark → mask)": {
                "window": "Размер локального окна порога; больше сглаживает фон, но теряет мелкие детали.",
                "C": "Смещение порога; больше C -> меньше FG (жестче).",
            },
            "Sauvola (dark → mask)": {
                "window": "Размер окна статистики для Sauvola.",
                "k": "Чувствительность к контрасту; выше k -> больше FG.",
                "R": "Динамический диапазон яркости; влияет на порог.",
            },
            "Niblack (dark → mask)": {
                "window": "Размер окна статистики для Niblack.",
                "k": "Смещение относительно среднего; ниже k -> больше FG.",
            },
            "Canny edges": {
                "low": "Нижний порог Canny (гистерезис); влияет на количество слабых краев.",
                "high": "Верхний порог Canny; сильные края.",
            },
            "Text + Lineart (MRC)": {
                "text_mode": "0=Sauvola, 1=AdaptiveMean — выбор детектора текста.",
                "sauvola_auto": "1=автоподбор окна и k для Sauvola.",
                "window": "Окно Sauvola (локальная статистика).",
                "k": "Параметр k Sauvola; выше -> больше FG.",
                "R": "Динамический диапазон Sauvola.",
                "mean_window": "Окно Adaptive Mean.",
                "mean_C": "Смещение порога Adaptive Mean.",
                "line_low": "Нижний порог Canny для линий.",
                "line_high": "Верхний порог Canny для линий.",
                "line_kernel": "Ядро усиления линий перед дилатацией.",
                "line_dilate": "Размер дилатации линий.",
                "close_kernel": "Морфологическое закрытие после объединения; заполняет разрывы.",
                "open_kernel": "Морфологическое открытие после объединения; убирает шум.",
            },
        }

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
        tf.addRow(self.method_label, self.method_combo)

        self.tabs = QTabWidget()

        tab_panels = QWidget()
        tab_panels_layout = QVBoxLayout(tab_panels)
        tab_panels_layout.addWidget(self.order_group)
        tab_panels_layout.addWidget(self.zoom_widget)
        tab_panels_layout.addStretch(1)

        tab_mask = QWidget()
        tab_mask_layout = QVBoxLayout(tab_mask)
        tab_mask_layout.addWidget(self.top_group)
        tab_mask_layout.addWidget(self.params_group)
        tab_mask_layout.addWidget(self.mask_post_group)
        tab_mask_layout.addWidget(self.preprocess_group)
        tab_mask_layout.addStretch(1)

        tab_fg = QWidget()
        tab_fg_layout = QVBoxLayout(tab_fg)
        tab_fg_layout.addWidget(self.fg_color_group)
        tab_fg_layout.addWidget(self.fg_smooth_group)
        tab_fg_layout.addWidget(self.fg_stats_group)
        tab_fg_layout.addStretch(1)

        tab_bg = QWidget()
        tab_bg_layout = QVBoxLayout(tab_bg)
        tab_bg_layout.addWidget(self.bg_fill_group)
        tab_bg_layout.addStretch(1)

        tab_save = QWidget()
        tab_save_layout = QVBoxLayout(tab_save)
        tab_save_layout.addWidget(self.save_group)
        tab_save_layout.addWidget(self.btn_save_mask)
        tab_save_layout.addWidget(self.btn_save_mask_inverted_tiff)
        tab_save_layout.addWidget(self.btn_save_fg)
        tab_save_layout.addWidget(self.btn_save_bg)
        tab_save_layout.addWidget(self.btn_reconstruct)
        tab_save_layout.addWidget(self.btn_save_recon)
        tab_save_layout.addStretch(1)

        self.tabs.addTab(tab_mask,      "Mask")
        self.tabs.addTab(tab_fg,        "Foreground")
        self.tabs.addTab(tab_bg,        "Background")
        self.tabs.addTab(tab_save,      "Save")
        self.tabs.addTab(tab_panels,    "UI")

        cl.addWidget(self.btn_open)
        cl.addWidget(self.tabs)

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
        self.btn_save_mask_inverted_tiff.clicked.connect(self.save_mask_inverted_tiff)
        self.btn_reconstruct.clicked.connect(self.reconstruct_image)
        self.btn_save_recon.clicked.connect(self.save_reconstructed)

        self.method_combo.currentIndexChanged.connect(self.rebuild_params)
        self.fg_format_combo.currentIndexChanged.connect(self._update_format_controls)
        self.bg_format_combo.currentIndexChanged.connect(self._update_format_controls)
        self.mask_format_combo.currentIndexChanged.connect(self.schedule_update)
        self.fg_jpeg_quality.valueChanged.connect(self.schedule_update)
        self.fg_png_level.valueChanged.connect(self.schedule_update)
        self.fg_jp2_quality.valueChanged.connect(self.schedule_update)
        self.bg_jpeg_quality.valueChanged.connect(self.schedule_update)
        self.bg_png_level.valueChanged.connect(self.schedule_update)
        self.bg_jp2_quality.valueChanged.connect(self.schedule_update)
        self.fg_color_group.toggled.connect(self.schedule_update)
        self.fg_color_mode.currentIndexChanged.connect(self.schedule_update)
        self.fg_color_mode.currentIndexChanged.connect(self._update_fg_color_controls)
        self.fg_block_size.valueChanged.connect(self.schedule_update)
        self.fg_block_levels.valueChanged.connect(self.schedule_update)
        self.fg_palette_size.valueChanged.connect(self.schedule_update)
        self.fg_posterize_levels.valueChanged.connect(self.schedule_update)
        self.fg_smooth_group.toggled.connect(self.schedule_update)
        self.fg_smooth_group.toggled.connect(self._update_fg_smooth_controls)
        self.fg_edge_smooth_mode.currentIndexChanged.connect(self.schedule_update)
        self.fg_edge_smooth_mode.currentIndexChanged.connect(self._update_fg_smooth_controls)
        self.fg_edge_bilateral_d.valueChanged.connect(self.schedule_update)
        self.fg_edge_bilateral_sigma_color.valueChanged.connect(self.schedule_update)
        self.fg_edge_bilateral_sigma_space.valueChanged.connect(self.schedule_update)
        self.fg_edge_guided_radius.valueChanged.connect(self.schedule_update)
        self.fg_edge_guided_eps.valueChanged.connect(self.schedule_update)
        self.fg_smooth_mode.currentIndexChanged.connect(self.schedule_update)
        self.fg_smooth_mode.currentIndexChanged.connect(self._update_fg_color_controls)
        self.fg_bilateral_d.valueChanged.connect(self.schedule_update)
        self.fg_bilateral_sigma_color.valueChanged.connect(self.schedule_update)
        self.fg_bilateral_sigma_space.valueChanged.connect(self.schedule_update)
        self.fg_guided_radius.valueChanged.connect(self.schedule_update)
        self.fg_guided_eps.valueChanged.connect(self.schedule_update)
        self.fg_quant_levels.valueChanged.connect(self.schedule_update)
        self.fg_quant_mode.currentIndexChanged.connect(self.schedule_update)
        self.fg_post_blur.toggled.connect(self.schedule_update)
        self.fg_post_blur_sigma.valueChanged.connect(self.schedule_update)
        self.mask_post_group.toggled.connect(self.schedule_update)
        self.mask_dilate_size.valueChanged.connect(self.schedule_update)
        self.mask_dilate_iter.valueChanged.connect(self.schedule_update)
        self.preprocess_group.toggled.connect(self.schedule_update)
        self.preprocess_combo.currentIndexChanged.connect(self.schedule_update)
        self.preprocess_kernel.valueChanged.connect(self.schedule_update)
        self.preprocess_sigma.valueChanged.connect(self.schedule_update)
        self.preprocess_sigma_color.valueChanged.connect(self.schedule_update)
        self.preprocess_sigma_space.valueChanged.connect(self.schedule_update)
        self.bg_fill_group.toggled.connect(self._on_bg_fill_toggle)
        self.bg_fill_combo.currentIndexChanged.connect(self.schedule_update)
        self.panel_order.itemChanged.connect(self.schedule_update)
        self.zoom_slider.valueChanged.connect(self._on_zoom_change)
        self.btn_order_up.clicked.connect(self._move_panel_up)
        self.btn_order_down.clicked.connect(self._move_panel_down)
        self._wire_scroll_sync()

        self.fg_color_group.setChecked(False)
        self.fg_smooth_group.setChecked(False)
        self.mask_post_group.setChecked(False)
        self.preprocess_group.setChecked(False)
        self.bg_fill_group.setChecked(True)
        self.save_group.setChecked(True)
        self.save_group.toggled.connect(self.schedule_update)

        for group in (
            self.top_group,
            self.params_group,
            self.fg_color_group,
            self.fg_smooth_group,
            self.order_group,
            self.mask_post_group,
            self.preprocess_group,
            self.bg_fill_group,
            self.save_group,
        ):
            self._make_group_collapsible(group)

        self._update_format_controls()
        self._update_fg_color_controls()
        self._update_fg_smooth_controls()
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

    def schedule_update(self) -> None:
        self._update_timer.start()

    def _show_info(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def _make_info_button(self, title: str, text: str) -> QToolButton:
        btn = QToolButton()
        btn.setText("ⓘ")
        btn.setAutoRaise(True)
        btn.clicked.connect(lambda: self._show_info(title, text))
        btn.setToolTip("Описание параметра")
        return btn

    def _make_label_with_info(self, label: str, info: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(QLabel(label), 1)
        layout.addWidget(self._make_info_button(label, info), 0)
        return row

    def _trace(self, label: str, start: float) -> None:
        if not self._trace_enabled:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"[trace] {label}: {elapsed_ms:.2f} ms")

    def _set_row_visible(self, label: QWidget, widget: QWidget, visible: bool) -> None:
        label.setVisible(visible)
        widget.setVisible(visible)

    def _update_format_controls(self) -> None:
        fg_fmt = self.fg_format_combo.currentData()
        bg_fmt = self.bg_format_combo.currentData()
        fg_is_jpeg = fg_fmt == "JPEG"
        bg_is_jpeg = bg_fmt == "JPEG"
        fg_is_png = fg_fmt == "PNG"
        bg_is_png = bg_fmt == "PNG"
        fg_is_jp2 = fg_fmt == "JP2"
        bg_is_jp2 = bg_fmt == "JP2"
        self._set_row_visible(self.fg_jpeg_label, self.fg_jpeg_quality, fg_is_jpeg)
        self._set_row_visible(self.fg_png_label, self.fg_png_level, fg_is_png)
        self._set_row_visible(self.fg_jp2_label, self.fg_jp2_quality, fg_is_jp2)
        self._set_row_visible(self.bg_jpeg_label, self.bg_jpeg_quality, bg_is_jpeg)
        self._set_row_visible(self.bg_png_label, self.bg_png_level, bg_is_png)
        self._set_row_visible(self.bg_jp2_label, self.bg_jp2_quality, bg_is_jp2)
        self.schedule_update()

    def _update_fg_color_controls(self) -> None:
        mode = self.fg_color_mode.currentText()
        show_blocks = mode == "Posterized blocks"
        show_palette = mode in ("Palette quantization", "Palette stepped gradient")
        show_posterize = mode == "FG_Posterized"
        show_shading = mode == "FG_PosterizedShading"
        self._set_row_visible(self.fg_block_size_label, self.fg_block_size, show_blocks)
        self._set_row_visible(self.fg_block_levels_label, self.fg_block_levels, show_blocks)
        self._set_row_visible(self.fg_palette_size_label, self.fg_palette_size, show_palette)
        self._set_row_visible(self.fg_posterize_levels_label, self.fg_posterize_levels, show_posterize)
        self._set_row_visible(self.fg_smooth_mode_label, self.fg_smooth_mode, show_shading)
        self._set_row_visible(self.fg_quant_levels_label, self.fg_quant_levels, show_shading)
        self._set_row_visible(self.fg_quant_mode_label, self.fg_quant_mode, show_shading)
        self._set_row_visible(self.fg_post_blur_label, self.fg_post_blur, show_shading)
        self._set_row_visible(self.fg_post_blur_sigma_label, self.fg_post_blur_sigma, show_shading)

        smooth_mode = self.fg_smooth_mode.currentData()
        show_bilateral = show_shading and smooth_mode == "BILATERAL"
        show_guided = show_shading and smooth_mode == "GUIDED"
        self._set_row_visible(self.fg_bilateral_d_label, self.fg_bilateral_d, show_bilateral)
        self._set_row_visible(self.fg_bilateral_sigma_color_label, self.fg_bilateral_sigma_color, show_bilateral)
        self._set_row_visible(self.fg_bilateral_sigma_space_label, self.fg_bilateral_sigma_space, show_bilateral)
        self._set_row_visible(self.fg_guided_radius_label, self.fg_guided_radius, show_guided)
        self._set_row_visible(self.fg_guided_eps_label, self.fg_guided_eps, show_guided)

    def _update_fg_smooth_controls(self) -> None:
        enabled = self.fg_smooth_group.isChecked()
        smooth_mode = self.fg_edge_smooth_mode.currentData()
        show_bilateral = enabled and smooth_mode == "BILATERAL"
        show_guided = enabled and smooth_mode == "GUIDED"
        self._set_row_visible(self.fg_edge_smooth_mode_label, self.fg_edge_smooth_mode, enabled)
        self._set_row_visible(self.fg_edge_bilateral_d_label, self.fg_edge_bilateral_d, show_bilateral)
        self._set_row_visible(self.fg_edge_bilateral_sigma_color_label, self.fg_edge_bilateral_sigma_color, show_bilateral)
        self._set_row_visible(self.fg_edge_bilateral_sigma_space_label, self.fg_edge_bilateral_sigma_space, show_bilateral)
        self._set_row_visible(self.fg_edge_guided_radius_label, self.fg_edge_guided_radius, show_guided)
        self._set_row_visible(self.fg_edge_guided_eps_label, self.fg_edge_guided_eps, show_guided)

    def _update_fg_color_count(self,
                               fg_bgr: Optional[np.ndarray],
                               mask_black_fg: Optional[np.ndarray]) -> None:
        if fg_bgr is None or mask_black_fg is None:
            self.fg_color_count_label.setText("-")
            return

        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            self.fg_color_count_label.setText("0")
            return

        pixels = fg_bgr[fg_mask]
        if pixels.size == 0:
            self.fg_color_count_label.setText("0")
            return

        unique_colors = np.unique(pixels.reshape(-1, 3), axis=0)
        self.fg_color_count_label.setText(str(unique_colors.shape[0]))

    def _wire_scroll_sync(self) -> None:
        for scroll in self._scroll_areas:
            vbar = scroll.verticalScrollBar()
            hbar = scroll.horizontalScrollBar()
            self._scrollbar_map[vbar] = (scroll, True)
            self._scrollbar_map[hbar] = (scroll, False)
            vbar.valueChanged.connect(self._on_scrollbar_changed)
            hbar.valueChanged.connect(self._on_scrollbar_changed)
            scroll.viewport().installEventFilter(self)

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

    def eventFilter(self, obj, event):  # noqa: N802 (Qt signature)
        if event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                step = 5 if delta > 0 else -5
                value = self.zoom_slider.value() + step
                self.zoom_slider.setValue(max(self.zoom_slider.minimum(),
                                              min(self.zoom_slider.maximum(), value)))
                return True
        return super().eventFilter(obj, event)

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

    def _is_sauvola_method(self, method: MethodSpec) -> bool:
        return method.fn is method_sauvola

    def current_bg_fill(self) -> BgFillSpec:
        key = self.bg_fill_combo.currentData()
        return BG_FILL_BY_KEY.get(key, BG_FILL_SPECS[0])

    def _on_bg_fill_toggle(self) -> None:
        enabled = self.bg_fill_group.isChecked()
        self.bg_fill_combo.setEnabled(enabled)
        self.schedule_update()

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        if not self.preprocess_group.isChecked():
            return gray

        mode = self.preprocess_combo.currentData()
        kernel = max(1, int(self.preprocess_kernel.value()))
        sigma = float(self.preprocess_sigma.value())
        sigma_color = float(self.preprocess_sigma_color.value())
        sigma_space = float(self.preprocess_sigma_space.value())

        if mode == "GAUSSIAN":
            k = ensure_odd(kernel, 3)
            return cv2.GaussianBlur(gray, (k, k), sigma)
        if mode == "MEDIAN":
            k = ensure_odd(kernel, 3)
            return cv2.medianBlur(gray, k)
        if mode == "BILATERAL":
            return cv2.bilateralFilter(gray, kernel, sigma_color, sigma_space)
        if mode == "BOX":
            k = max(1, kernel)
            return cv2.blur(gray, (k, k))
        return gray

    def _apply_sauvola_auto_state(self) -> None:
        if "window" in self.param_widgets:
            self.param_widgets["window"].setEnabled(not self.cb_sauvola_auto.isChecked())
        if "k" in self.param_widgets:
            self.param_widgets["k"].setEnabled(not self.cb_sauvola_auto.isChecked())

    def _format_size_text(self, size_bytes: Optional[int]) -> str:
        if size_bytes is None:
            return "-"
        return f"{size_bytes / 1024.0:.1f} KB"

    def _encode_decode_color(self,
                             bgr: np.ndarray,
                             fmt: str,
                             jpeg_quality: int,
                             png_level: int,
                             jp2_quality: int) -> tuple[Optional[np.ndarray], Optional[int]]:
        with tempfile.TemporaryDirectory() as tmp:
            if fmt == "JPEG":
                ext = ".jpg"
                params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            elif fmt == "JP2":
                ext = ".jp2"
                params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(jp2_quality) * 10]
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
                int(self.fg_jp2_quality.value()),
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
                int(self.bg_jp2_quality.value()),
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
                w = LabeledSlider(ps, self.schedule_update)
                self.param_widgets[ps.key] = w
                info = self._method_param_info.get(m.name, {}).get(
                    ps.key,
                    "Описание параметра отсутствует."
                )
                label = self._make_label_with_info(ps.label, info)
                self.params_form.addRow(label, w)

        if self._is_sauvola_method(m):
            auto_row = QWidget()
            auto_layout = QHBoxLayout(auto_row)
            auto_layout.setContentsMargins(0, 0, 0, 0)
            auto_layout.addWidget(self.cb_sauvola_auto)
            auto_layout.addWidget(self.lbl_sauvola_auto, 1)
            auto_label = self._make_label_with_info(
                "Auto",
                "Автоподбор окна и k для Sauvola; отключает ручные параметры."
            )
            self.params_form.addRow(auto_label, auto_row)
            self._apply_sauvola_auto_state()
        else:
            self.cb_sauvola_auto.setChecked(False)
            self.lbl_sauvola_auto.setText("-")

        self.schedule_update()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "E:\\Images\\PDFC", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
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
        self.btn_save_mask_inverted_tiff.setEnabled(True)

        self.schedule_update()

    def gather_params(self) -> Dict[str, Any]:
        return {k: w.value() for k, w in self.param_widgets.items()}

    def update_results(self):
        self._apply_panel_visibility()
        if self.gray is None or self.color is None:
            self.fg_size_label.setText("-")
            self.bg_size_label.setText("-")
            self.mask_size_label.setText("-")
            self.fg_color_count_label.setText("-")
            return

        try:
            t_total = time.perf_counter()
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
                t_mask = time.perf_counter()
                gray_for_mask = self._preprocess_gray(self.gray)
                self._trace("mask.preprocess_gray", t_mask)
                if self._is_sauvola_method(method) and self.cb_sauvola_auto.isChecked():
                    t_sauvola = time.perf_counter()
                    window, k = auto_tune_sauvola(gray_for_mask)
                    self._trace("mask.auto_sauvola", t_sauvola)
                    params["window"] = window
                    params["k"] = k
                    self.lbl_sauvola_auto.setText(f"window={window}, k={k:.3f}")
                    if "window" in self.param_widgets:
                        self.param_widgets["window"].spin.setValue(int(window))
                    if "k" in self.param_widgets:
                        self.param_widgets["k"].spin.setValue(float(k))
                    self._apply_sauvola_auto_state()
                else:
                    if self._is_sauvola_method(method):
                        self.lbl_sauvola_auto.setText("-")
                        self._apply_sauvola_auto_state()
                t_method = time.perf_counter()
                mask_white = method.fn(gray_for_mask, params)
                self._trace(f"mask.method.{method.name}", t_method)
                mask_white = (mask_white > 0).astype(np.uint8) * 255
                if self.mask_post_group.isChecked():
                    t_post = time.perf_counter()
                    k = ensure_odd(int(self.mask_dilate_size.value()), 1)
                    it = int(self.mask_dilate_iter.value())
                    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    mask_white = cv2.dilate(mask_white, se, iterations=it)
                    self._trace("mask.post_dilate", t_post)
                t_mask_conv = time.perf_counter()
                mask_black_fg_raw = to_black_fg(mask_white)
                self._trace("mask.to_black_fg", t_mask_conv)
            else:
                self.mask_black_fg = None
                self.mask_black_fg_raw = None
                self._mask_pix = None
                self.mask_view.clear()

            fg_raw = None
            if need_fg:
                if mask_black_fg_raw is None:
                    self._update_fg_color_count(None, None)
                    return
                if self.fg_color_group.isChecked():
                    mode = self.fg_color_mode.currentText()
                    t_fg = time.perf_counter()
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
                    elif mode == "FG_Posterized":
                        fg_raw = self.make_fg_posterized(
                            self.color,
                            mask_black_fg_raw,
                            levels=int(self.fg_posterize_levels.value()),
                        )
                    elif mode == "Palette quantization":
                        fg_raw = self.make_fg_palette_quantized(
                            self.color,
                            mask_black_fg_raw,
                            colors=int(self.fg_palette_size.value()),
                        )
                    elif mode == "Palette stepped gradient":
                        fg_raw = self.make_fg_palette_stepped_gradient(
                            self.color,
                            mask_black_fg_raw,
                            colors=int(self.fg_palette_size.value()),
                        )
                    elif mode == "FG_PosterizedShading":
                        fg_raw = self.make_fg_posterized_shading(
                            self.color,
                            mask_black_fg_raw,
                            smooth_mode=str(self.fg_smooth_mode.currentData()),
                            bilateral_d=int(self.fg_bilateral_d.value()),
                            bilateral_sigma_color=int(self.fg_bilateral_sigma_color.value()),
                            bilateral_sigma_space=int(self.fg_bilateral_sigma_space.value()),
                            guided_radius=int(self.fg_guided_radius.value()),
                            guided_eps=float(self.fg_guided_eps.value()),
                            levels=int(self.fg_quant_levels.value()),
                            quant_mode=str(self.fg_quant_mode.currentData()),
                            post_blur=self.fg_post_blur.isChecked(),
                            post_blur_sigma=float(self.fg_post_blur_sigma.value()),
                        )
                    else:
                        fg_raw = self.make_fg_unified(
                            self.color,
                            mask_black_fg_raw,
                            mode
                        )
                    self._trace(f"fg.mode.{mode}", t_fg)
                    fg_raw = self._apply_fg_edge_smoothing(fg_raw, mask_black_fg_raw)
                    self._update_fg_color_count(fg_raw, mask_black_fg_raw)
                else:
                    # FG must preserve edges; do not blur, inpaint, or smooth.
                    # ORIGINAL FG: keep original pixels under mask
                    t_fg = time.perf_counter()
                    fg = np.full_like(self.color, 255)
                    fg[mask_black_fg_raw == 0] = self.color[mask_black_fg_raw == 0]
                    fg_raw = fg
                    self._trace("fg.mode.Original", t_fg)
                    fg_raw = self._apply_fg_edge_smoothing(fg_raw, mask_black_fg_raw)
                    self._update_fg_color_count(fg_raw, mask_black_fg_raw)
            else:
                self.fg_bgr = None
                self.fg_bgr_raw = None
                self._fg_pix = None
                self.fg_view.clear()
                self._update_fg_color_count(None, None)

            bg_raw = None
            if need_bg:
                if mask_black_fg_raw is None:
                    return
                # BG bitmap: keep original pixels where mask is WHITE, fill FG holes
                if self.bg_fill_group.isChecked():
                    # BG should remain smooth/low-frequency for compression efficiency.
                    t_bg = time.perf_counter()
                    fg_mask = (mask_black_fg_raw == 0).astype(np.uint8) * 255
                    fill = self.current_bg_fill()
                    bg_raw = fill.fn(self.color, fg_mask)
                    self._trace(f"bg.fill.{fill.key}", t_bg)
                else:
                    t_bg = time.perf_counter()
                    bg_raw = make_bg_image(self.color, mask_black_fg_raw)
                    self._trace("bg.fill.none", t_bg)
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
                t_compress = time.perf_counter()
                self._apply_runtime_compression()
                self._trace("runtime.compression", t_compress)

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
            self._trace("update_results.total", t_total)

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
        elif fmt == "JP2":
            default_name = "fg.jp2"
            flt = "JPEG2000 (*.jp2)"
            # OpenCV JPEG2000 uses a compression ratio (higher = stronger compression).
            ui_quality = float(self.fg_jp2_quality.value())
            jp2_min = 50.0
            jp2_max = 150.0
            jp2_compression = jp2_min + (ui_quality / 100.0) * (jp2_max - jp2_min)
            jp2_compression = int(round(max(jp2_min, min(jp2_max, jp2_compression))))
            params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, jp2_compression]
        elif fmt == "JP2_KDU":
            default_name = "fg.jp2"
            flt = "JPEG2000 (*.jp2)"
            params = None
        else:
            default_name = "fg.png"
            flt = "PNG (*.png)"
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(self.fg_png_level.value())]
        path, _ = QFileDialog.getSaveFileName(self, "Save FG", default_name, flt)
        if path:
            if fmt == "JP2_KDU":
                # Use Kakadu CLI (kdu_compress). JPEG2000 options are native,
                # not JPEG-style quality; pass explicit Kakadu parameters.
                fg_bgr = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_tif = os.path.join(tmp, "fg.tif")
                    cv2.imwrite(tmp_tif, fg_bgr)

                    # Configure Kakadu options here (bind UI to these values as needed).
                    kdu_options = {
                        "rate": None,            # e.g. "0.5" (bpp); omit for lossless or layered control
                        "Clayers": None,         # e.g. "8"
                        "Creversible": "no",     # "yes" for lossless
                        "Clevels": "5",          # decomposition levels
                        "Stiles": None,          # e.g. "{1024,1024}" or None for untiled
                        "Cprecincts": None,      # e.g. "{256,256},{256,256},..."
                        "Corder": "RLCP",        # progression order
                        "Cycc": "yes",           # YCbCr transform on/off
                        "Csubsampling": None,    # e.g. "{2,2}"
                        "Qstep": None,           # chroma quantization strength (Kakadu syntax)
                        "Rroi": None,            # ROI coding (Kakadu syntax)
                    }
                    extra_kdu_args = []  # Any additional Kakadu args, e.g. ["Cblk={64,64}"]

                    kdu_args = []
                    for key, value in kdu_options.items():
                        if value is None:
                            continue
                        kdu_args.append(f"{key}={value}")
                    kdu_args.extend(extra_kdu_args)

                    cmd = ["kdu_compress", "-i", tmp_tif, "-o", path, *kdu_args]
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                    except subprocess.CalledProcessError as exc:
                        err = exc.stderr.strip() or exc.stdout.strip() or "Unknown Kakadu error."
                        QMessageBox.critical(self, "Save FG (JP2) failed", err)
                        return
            else:
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
        elif fmt == "JP2":
            default_name = "bg.jp2"
            flt = "JPEG2000 (*.jp2)"
            params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(self.bg_jp2_quality.value()) * 10]
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

    def save_mask_inverted_tiff(self):
        src = self.mask_black_fg_raw if self.mask_black_fg_raw is not None else self.mask_black_fg
        if src is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save mask (TIFF, black text)",
            "mask_black_text.tif",
            "TIFF (*.tif *.tiff)"
        )
        if path:
            # Save as classic binary: black text (FG), white background (BG).
            cv2.imwrite(path, src)

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

    def _apply_fg_edge_smoothing(self,
                                 fg_bgr: Optional[np.ndarray],
                                 mask_black_fg: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.fg_smooth_group.isChecked():
            return fg_bgr
        if fg_bgr is None or mask_black_fg is None:
            return fg_bgr

        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            return fg_bgr

        fg = fg_bgr.copy()
        fg[~fg_mask] = 255

        smooth_mode = str(self.fg_edge_smooth_mode.currentData())
        if smooth_mode == "GUIDED" and hasattr(cv2, "ximgproc"):
            if hasattr(cv2.ximgproc, "guidedFilter"):
                smoothed = cv2.ximgproc.guidedFilter(
                    guide=fg,
                    src=fg,
                    radius=int(max(1, self.fg_edge_guided_radius.value())),
                    eps=float(max(0.0, self.fg_edge_guided_eps.value())),
                )
            else:
                smoothed = cv2.bilateralFilter(
                    fg,
                    int(max(1, self.fg_edge_bilateral_d.value())),
                    int(max(1, self.fg_edge_bilateral_sigma_color.value())),
                    int(max(1, self.fg_edge_bilateral_sigma_space.value())),
                )
        else:
            smoothed = cv2.bilateralFilter(
                fg,
                int(max(1, self.fg_edge_bilateral_d.value())),
                int(max(1, self.fg_edge_bilateral_sigma_color.value())),
                int(max(1, self.fg_edge_bilateral_sigma_space.value())),
            )

        if smoothed.dtype != np.uint8:
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

        smoothed[~fg_mask] = 255
        return smoothed

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

    def make_fg_posterized(self,
                           color_bgr: np.ndarray,
                           mask_black_fg: np.ndarray,
                           levels: int = 2) -> np.ndarray:
        """
        Posterize FG colors with a fixed number of uniform levels per channel.
        """
        fg = np.full_like(color_bgr, 255)
        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            return fg

        levels = max(2, int(levels))
        indices = np.arange(0, 256, dtype=np.float32)
        divider = 255.0 / float(levels)
        quantiz = np.linspace(0, 255, levels, dtype=np.float32)
        color_levels = np.clip((indices / divider).astype(np.int32), 0, levels - 1)
        palette = quantiz[color_levels].astype(np.uint8)

        posterized = palette[color_bgr]
        fg[fg_mask] = posterized[fg_mask]
        return fg

    def make_fg_palette_quantized(self,
                                  color_bgr: np.ndarray,
                                  mask_black_fg: np.ndarray,
                                  colors: int = 8) -> np.ndarray:
        """
        Create FG where colors are quantized to a fixed palette size.
        """
        fg = np.full_like(color_bgr, 255)
        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            return fg

        pixels = color_bgr[fg_mask].astype(np.float32)
        if pixels.size == 0:
            return fg

        k = max(1, int(colors))
        k = min(k, pixels.shape[0])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
        quantized = centers_u8[labels.flatten()]
        fg[fg_mask] = quantized
        return fg

    def make_fg_palette_stepped_gradient(self,
                                         color_bgr: np.ndarray,
                                         mask_black_fg: np.ndarray,
                                         colors: int = 8) -> np.ndarray:
        """
        Create FG with a stepped gradient mapped to a fixed palette size.
        """
        fg = np.full_like(color_bgr, 255)
        fg_mask = (mask_black_fg == 0)
        if not fg_mask.any():
            return fg

        pixels = color_bgr[fg_mask].astype(np.float32)
        if pixels.size == 0:
            return fg

        k = max(1, int(colors))
        k = min(k, pixels.shape[0])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        centers_u8 = np.clip(centers, 0, 255).astype(np.uint8)
        centers_y = cv2.cvtColor(
            centers_u8.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb
        )[:, 0, 0].astype(np.float32)
        order = np.argsort(centers_y)
        palette = centers_u8[order]

        fg_pixels = color_bgr[fg_mask].astype(np.float32)
        fg_y = cv2.cvtColor(
            fg_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2YCrCb
        )[:, 0, 0].astype(np.float32)
        y_min = float(fg_y.min())
        y_max = float(fg_y.max())
        if y_max <= y_min:
            fg[fg_mask] = palette[-1]
            return fg

        t = (fg_y - y_min) / (y_max - y_min)
        idx = np.clip((t * (k - 1)).round().astype(np.int32), 0, k - 1)
        fg[fg_mask] = palette[idx]
        return fg

    def make_fg_posterized_shading(self,
                                   color_bgr: np.ndarray,
                                   mask_black_fg: np.ndarray,
                                   *,
                                   smooth_mode: str,
                                   bilateral_d: int,
                                   bilateral_sigma_color: int,
                                   bilateral_sigma_space: int,
                                   guided_radius: int,
                                   guided_eps: float,
                                   levels: int,
                                   quant_mode: str,
                                   post_blur: bool,
                                   post_blur_sigma: float) -> np.ndarray:
        """
        Create FG with stepwise (posterized) shading limited to the FG mask.
        """
        fg_mask = (mask_black_fg == 0)
        fg = color_bgr.copy()
        fg[~fg_mask] = 255
        if not fg_mask.any():
            return fg

        if smooth_mode == "GUIDED" and hasattr(cv2, "ximgproc"):
            if hasattr(cv2.ximgproc, "guidedFilter"):
                smoothed = cv2.ximgproc.guidedFilter(
                    guide=fg,
                    src=fg,
                    radius=int(max(1, guided_radius)),
                    eps=float(max(0.0, guided_eps)),
                )
            else:
                smoothed = cv2.bilateralFilter(
                    fg,
                    int(max(1, bilateral_d)),
                    int(max(1, bilateral_sigma_color)),
                    int(max(1, bilateral_sigma_space)),
                )
        else:
            smoothed = cv2.bilateralFilter(
                fg,
                int(max(1, bilateral_d)),
                int(max(1, bilateral_sigma_color)),
                int(max(1, bilateral_sigma_space)),
            )

        if smoothed.dtype != np.uint8:
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

        smoothed[~fg_mask] = 255
        ycrcb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        y_fg = y[fg_mask]

        levels = max(2, int(levels))
        if quant_mode == "KMEANS":
            samples = y_fg.reshape(-1, 1).astype(np.float32)
            k = min(levels, samples.shape[0])
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(
                samples,
                k,
                None,
                criteria,
                3,
                cv2.KMEANS_PP_CENTERS,
            )
            centers = centers.flatten()
            y_quant = y.copy()
            y_quant[fg_mask] = centers[labels.flatten()]
        else:
            y_min = float(y_fg.min())
            y_max = float(y_fg.max())
            if y_max <= y_min:
                y_quant = y.copy()
                y_quant[fg_mask] = y_min
            else:
                step = (y_max - y_min) / float(levels - 1)
                y_quant = y.copy()
                y_quant[fg_mask] = (
                    np.round((y_fg - y_min) / step) * step + y_min
                )

        if post_blur and post_blur_sigma > 0:
            y_blur = cv2.GaussianBlur(
                y_quant.astype(np.float32),
                (0, 0),
                sigmaX=float(post_blur_sigma),
                sigmaY=float(post_blur_sigma),
            )
            y_quant[fg_mask] = y_blur[fg_mask]

        ycrcb[:, :, 0] = np.clip(y_quant, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        out[~fg_mask] = 255
        return out

def main():
    app = QApplication(sys.argv)
    w = MRCTool()
    w.resize(1800, 900)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
