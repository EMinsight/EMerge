# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.

# Last Cleanup: 2025-01-01
from .bcs import PortBC
from ...mth.integrals import surface_integral
import numpy as np
from typing import Callable
from ...const import Z0


def sparam_mode_power(
    nodes: np.ndarray,
    tri_vertices: np.ndarray,
    bc: PortBC,
    mode_nr: int,
    k0: float,
    const: np.ndarray,
    gq_order: int = 4,
):
    """Compute the S-parameters assuming a wave port mode

    Arguments:
    ----------
    nodes: np.ndarray = (3,:) np.ndarray of all nodes in the mesh.
    tri_vertices: np.ndarray = (3,:) np.ndarray of triangle indices that need to be integrated,
    bc: RobinBC = The port boundary condition object
    freq: float = The frequency at which to do the calculation
    fielf: Callable = The interpolation fuction that computes the E-field from the simulation
    gq_order: int = 4 the Gauss-Quadrature order (default = 4)
    """

    def inproduct2(x, y, z):
        Ex1, Ey1, Ez1 = bc.port_mode_3d_global(
            x, y, z, k0, which="Exy", mode_nr=mode_nr
        )
        Ex2 = np.conj(Ex1)
        Ey2 = np.conj(Ey1)
        Ez2 = np.conj(Ez1)
        return (Ex1 * Ex2 + Ey1 * Ey2 + Ez1 * Ez2) / (2 * Z0)

    norm = surface_integral(nodes, tri_vertices, inproduct2, const, gq_order=gq_order)
    return norm


def sparam_field_power(
    nodes: np.ndarray,
    tri_vertices: np.ndarray,
    bc: PortBC,
    mode_nr: int,
    active: bool,
    k0: float,
    fieldf: Callable,
    const: np.ndarray,
    gq_order: int = 4,
) -> complex:
    """Compute the S-parameters assuming a wave port mode

    Arguments:
    ----------
    nodes: np.ndarray = (3,:) np.ndarray of all nodes in the mesh.
    tri_vertices: np.ndarray = (3,:) np.ndarray of triangle indices that need to be integrated,
    bc: RobinBC = The port boundary condition object
    freq: float = The frequency at which to do the calculation
    fielf: Callable = The interpolation fuction that computes the E-field from the simulation
    gq_order: int = 4 the Gauss-Quadrature order (default = 4)
    """

    Q = 0
    if active:
        Q = 1

    def inproduct1(x, y, z):
        mode_field = bc.port_mode_3d_global(x, y, z, k0, which="Exy", mode_nr=mode_nr)
        Ex1, Ey1, Ez1 = fieldf(x, y, z) - Q * mode_field
        Ex2, Ey2, Ez2 = np.conj(mode_field)
        return (Ex1 * Ex2 + Ey1 * Ey2 + Ez1 * Ez2) / (2 * Z0)

    mode_dot_field = surface_integral(
        nodes, tri_vertices, inproduct1, const, gq_order=gq_order
    )
    svec = mode_dot_field
    return svec
