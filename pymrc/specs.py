"""Dataclasses describing parameter and method specifications."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np


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
