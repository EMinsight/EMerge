from dataclasses import dataclass
from ....elements.nedleg2 import FieldFunctionClass
from emsutil import Saveable
import numpy as np
from typing import Literal, Callable
from loguru import logger


@dataclass
class PortMode(Saveable):
    modefield: np.ndarray
    E_function: FieldFunctionClass
    H_function: FieldFunctionClass
    k0: float
    beta: float
    residual: float
    energy: float = 0
    norm_factor: float = 1
    freq: float = 0
    neff: float = 1
    Z0: float = 50.0
    polarity: float = 1.0
    modetype: Literal["TEM", "TE", "TM"] = "TEM"

    def __post_init__(self):
        self.neff = self.beta / self.k0
        self.energy = np.mean(np.abs(self.modefield) ** 2)

    def __str__(self):
        return f"PortMode(k0={self.k0}, beta={self.beta}({self.neff:.3f}))"

    def set_power(self, power: complex) -> None:
        self.norm_factor = np.sqrt(1 / np.abs(power))
        logger.info(f"Setting port mode amplitude to: {self.norm_factor:.2f} ")
