import numpy as np
from .._emerge.geo import XYPolygon
from .._emerge.geometry import GeoSurface
from .._emerge.cs import CoordinateSystem, GCS


def vivaldi_taper(
    length: float,
    gap: float,
    opening: float,
    curve_coefficient: float = 200.0,
    mirrored: bool = True,
    dilation: float = 0.0,
    cs: CoordinateSystem = GCS,
) -> GeoSurface:
    L = length
    g = gap
    W = opening
    K = curve_coefficient
    A = (g / 2) - (W - g * K) / (2 - 2 * K)
    fx = lambda t: t * L
    fy = lambda t: (g / 2) * K**t + (W - g * K) / (2 - 2 * K) * (1 - K**t)
    if dilation == 0:
        if mirrored:
            exp_taper = (
                XYPolygon()
                .parametric(fx, fy, tolerance=1e-5)
                .parametric(fx, lambda t: -fy(t), reverse=True, tolerance=1e-5)
                .geo(cs)
            )
        else:
            exp_taper = (
                XYPolygon()
                .parametric(fx, fy, tolerance=1e-5)
                .extend([L, 0], [0, 0])
                .geo(cs)
            )
        return exp_taper
    else:
        dfx = lambda t: A * np.log(K) / L * K ** (t)
        R = lambda t: 1 / np.sqrt(1 + dfx(t) ** 2)
        fx2 = lambda t: t * L - dilation * R(t) * dfx(t)
        fy2 = lambda t: fy(t) + dilation * R(t)
        if mirrored:
            exp_taper_dialated = (
                XYPolygon()
                .parametric(fx2, fy2, tolerance=1e-5, tmax=1.1)
                .parametric(
                    fx2, lambda t: -fy2(t), reverse=True, tolerance=1e-5, tmax=1.1
                )
                .geo(cs)
            )
        else:
            exp_taper_dialated = (
                XYPolygon()
                .parametric(fx2, fy2, tolerance=1e-5, tmax=1.1)
                .extend([L, 0], [0, 0])
                .geo(cs)
            )
        return exp_taper_dialated
