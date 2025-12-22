"""Reusable Qt widgets for parameter editing."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QSlider, QSpinBox, QWidget

from .specs import ParamSpec


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
