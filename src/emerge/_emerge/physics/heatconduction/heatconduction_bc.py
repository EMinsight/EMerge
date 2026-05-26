from __future__ import annotations
import numpy as np
from loguru import logger
from typing import Callable, Literal, Generator
from dataclasses import dataclass
from collections import defaultdict
from ...selection import Selection, FaceSelection, DomainSelection
from ...cs import CoordinateSystem, Axis, GCS, _parse_axis
from ...coord import Line
from ...geometry import GeoSurface, GeoVolume
from ...bc import (
    BoundaryCondition,
    BoundaryConditionSet,
    Periodic,
    BoundaryConditionError,
)
from ...periodic import PeriodicCell, HexCell, RectCell
from emsutil import Material, AIR, Saveable
from ...const import Z0, C0, EPS0, MU0
from ...logsettings import DEBUG_COLLECTOR


class HCBoundaryConditionSet(BoundaryConditionSet):
    def __init__(self, periodic_cell: PeriodicCell | None):
        super().__init__()

        self.FixedTemperatureBoundary: type[FixedTemperatureBoundary] = (
            self._construct_bc(FixedTemperatureBoundary)
        )
        self.FixedTemperatureVolume: type[FixedTemperatureVolume] = self._construct_bc(
            FixedTemperatureVolume
        )
        self.HeatFluxBoundary: type[HeatFluxBoundary] = self._construct_bc(
            HeatFluxBoundary
        )
        self.HeatFluxVolume: type[HeatFluxVolume] = self._construct_bc(HeatFluxVolume)
        self.ThermalContact: type[ThermalContact] = self._construct_bc(ThermalContact)
        self.ThinConductor: type[ThinConductor] = self._construct_bc(ThinConductor)
        self.Convection: type[Convection] = self._construct_bc(Convection)
        self.BlackBodyRadiation: type[BlackBodyRadiation] = self._construct_bc(
            BlackBodyRadiation
        )

    def get_type(
        self,
        bctype: Literal[
            "FixedTemperatureBoundary", "FixedTemperatureVolume", "HeatFluxBoundary"
        ],
    ) -> Selection:
        tags = []
        for bc in self.boundary_conditions:
            if bctype in str(bc.__class__):
                tags.extend(bc.selection.tags)
        return FaceSelection(tags)

    def CoupledEMHeating(
        self,
        selection: DomainSelection | GeoVolume,
        mwfield,
        excitation_W: list[float] | None = None,
    ):
        from ..microwave.microwave_data import MWField

        mwfield: MWField = mwfield
        if excitation_W is not None:
            mwfield.set_excitations(*[x**0.5 for x in excitation_W])

        def qcallable(x, y, z):
            Q = mwfield.interpolate(x, y, z, False).scalar("Qv", "real").F
            return Q

        return self.HeatFluxVolume(selection, None, heatflux_func=qcallable)


class FixedTemperatureBoundary(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "FixedTemperature"
    _texture: str = "tex1.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, temperature_K: float):
        super().__init__(face)
        self.T: float = temperature_K


class FixedTemperatureVolume(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "FixedTemperature"
    _texture: str = "tex1.png"
    dim: int = 3

    def __init__(self, face: DomainSelection | GeoVolume, temperature_K: float):
        super().__init__(face)
        self.T: float = temperature_K


class HeatFluxBoundary(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "HeatFluxBoundary"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, heatflux: float):
        super().__init__(face)
        self.qm: float = heatflux


class HeatFluxVolume(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "HeatFluxVolume"
    _texture: str = "tex2.png"
    dim: int = 3

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        heatflux: float,
        heatflux_func: Callable | None = None,
    ):
        super().__init__(face)
        self.qm: float = heatflux
        if heatflux_func is None:
            self.fqm: Callable = lambda x, y, z: np.ones_like(x) * self.qm
        else:
            self.fqm: callable = heatflux_func


class ThermalContact(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThermalContact"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface, heat_transfer_coeff: float):
        super().__init__(face)
        self.h: float = heat_transfer_coeff


class ThinConductor(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThinConductor"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(
        self, face: FaceSelection | GeoSurface, material: Material, thickness: float
    ):
        super().__init__(face)
        self.material: Material = material
        self.thickness: float = thickness


class Convection(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "ThermalContact"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        heat_transfer_coeff: float,
        Tamb_K: float,
    ):
        super().__init__(face)
        self.h: float = heat_transfer_coeff
        self.Tamb: float = Tamb_K


class BlackBodyRadiation(BoundaryCondition, Saveable):
    _color: str = "#0effa7"
    _name: str = "BlackBodyRadiation"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(
        self,
        face: FaceSelection | GeoSurface,
        emissivity: float,
        Tamb_K: float,
    ):
        super().__init__(face)
        self.emissivity: float = emissivity
        self.Tamb: float = Tamb_K


class Isolated(HeatFluxBoundary):
    _color: str = "#0effa7"
    _name: str = "Isolated"
    _texture: str = "tex2.png"
    dim: int = 2

    def __init__(self, face: FaceSelection | GeoSurface):
        super().__init__(face, 0.0)
