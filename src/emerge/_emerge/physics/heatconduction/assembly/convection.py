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

import numpy as np
from numba import njit, f8, i8, types, prange
from .heatflux import TRI_DPTS, assemble_surface_flux
from ....elements.leg2 import Legrange2
from ....mth.optimized import calc_area

KC = 5.670374419e-8

############################################################
#              TRIANGLE SURFACE MASS ASSEMBLY             #
############################################################


# Precompute the mass matrix stencil for NiNj gauss quadrature integration
@njit(f8[:, :](), cache=True, nogil=True)
def _tri_mass_stencil():
    weights = TRI_DPTS[0, :]
    n_gausquad = weights.shape[0]

    # Local edge map
    edge_vertex_1 = np.array([0, 1, 0])
    edge_vertex_2 = np.array([1, 2, 2])
    M = np.zeros((6, 6), dtype=np.float64)

    for iq in range(n_gausquad):
        L1 = TRI_DPTS[1, iq]
        L2 = TRI_DPTS[2, iq]
        L3 = TRI_DPTS[3, iq]
        w = weights[iq]

        Ls = np.empty(3, dtype=np.float64)
        Ls[0] = L1
        Ls[1] = L2
        Ls[2] = L3

        N = np.empty(6, dtype=np.float64)

        # Mass contributions as: λ(2λ - 1)
        for iv in range(3):
            N[iv] = Ls[iv] * (2.0 * Ls[iv] - 1.0)

        # Mass edge contribution as: 4λ₁λ₂
        for ie in range(3):
            li = Ls[edge_vertex_1[ie]]
            lj = Ls[edge_vertex_2[ie]]
            N[3 + ie] = 4.0 * li * lj

        for i in range(6):
            for j in range(6):
                M[i, j] += w * N[i] * N[j]
    return M


# Pre-compute only the different basis functions for each Gauss-Quadrature point.
@njit(f8[:, :](), cache=True, nogil=True)
def _tri_mass_Nonly():
    weights = TRI_DPTS[0, :]
    n_gausquad = weights.shape[0]

    # Local edge map
    edge_vertex_1 = np.array([0, 1, 0])
    edge_vertex_2 = np.array([1, 2, 2])
    N = np.zeros((6, 10), dtype=np.float64)

    for iq in range(n_gausquad):
        L1 = TRI_DPTS[1, iq]
        L2 = TRI_DPTS[2, iq]
        L3 = TRI_DPTS[3, iq]

        Ls = np.empty(3, dtype=np.float64)
        Ls[0] = L1
        Ls[1] = L2
        Ls[2] = L3

        # Mass contributions as: λ(2λ - 1)
        for iv in range(3):
            N[iv, iq] = Ls[iv] * (2.0 * Ls[iv] - 1.0)

        # Mass edge contribution as: 4λ₁λ₂
        for ie in range(3):
            li = Ls[edge_vertex_1[ie]]
            lj = Ls[edge_vertex_2[ie]]
            N[3 + ie, iq] = 4.0 * li * lj
    return N


############################################################
#                OPTIMIZED ASSEMBLER ROUTINE               #
############################################################


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _robin_stiffness_builder(nodes, tris, tri_to_field, tri_ids_2d, h_coeff):
    NDOF_TRI = 36
    n_triangles = tri_ids_2d.shape[1]
    nnz = n_triangles * NDOF_TRI

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    M = _tri_mass_stencil()

    for idx in prange(n_triangles):
        p = idx * NDOF_TRI
        itri = tri_ids_2d[0, idx]

        iv1 = tris[0, itri]
        iv2 = tris[1, itri]
        iv3 = tris[2, itri]

        A = calc_area(nodes[:, iv1], nodes[:, iv2], nodes[:, iv3])

        scale = h_coeff * 2.0 * A

        field_ids = tri_to_field[:, itri]

        for i in range(6):
            for j in range(6):
                k = p + 6 * i + j
                rows[k] = field_ids[i]
                cols[k] = field_ids[j]
                values[k] = M[i, j] * scale

    return values, rows, cols


@njit(
    types.Tuple((f8[:], i8[:], i8[:], f8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8[:], f8, f8, i8
    ),
    cache=True,
    nogil=True,
    parallel=False,
)
def _radiation_builder(
    nodes,
    tris,
    tri_to_field,
    tri_ids_2d,
    T_dofs,
    emissivity,
    T_amb,
    n_field,
):
    n_triangles = tri_ids_2d.shape[1]
    nnz = n_triangles * 36
    sigma = KC

    K_values = np.empty(nnz, dtype=np.float64)
    K_rows = np.empty(nnz, dtype=np.int64)
    K_cols = np.empty(nnz, dtype=np.int64)
    f_global = np.zeros(n_field, dtype=np.float64)

    weights = TRI_DPTS[0, :]
    n_gausquad = weights.shape[0]
    T_amb2 = T_amb * T_amb

    T_local = np.empty(6, dtype=np.float64)

    N_prebuild = _tri_mass_Nonly()

    for i_triangle in range(n_triangles):
        p = i_triangle * 36
        itri = tri_ids_2d[0, i_triangle]

        iv1 = tris[0, itri]
        iv2 = tris[1, itri]
        iv3 = tris[2, itri]

        A = calc_area(nodes[:, iv1], nodes[:, iv2], nodes[:, iv3])

        field_ids = tri_to_field[:, itri]

        for i in range(6):
            T_local[i] = T_dofs[field_ids[i]]

        K_local = np.zeros((6, 6), dtype=np.float64)
        f_local = np.zeros(6, dtype=np.float64)

        for iq in range(n_gausquad):
            w = weights[iq]

            T_q = np.sum(N_prebuild[:, iq] * T_local)

            wh = w * emissivity * sigma * (T_q * T_q + T_amb2) * (T_q + T_amb)
            wht = wh * T_amb
            for i in range(6):
                f_local[i] += wht * N_prebuild[i, iq]
                for j in range(6):
                    K_local[i, j] += wh * N_prebuild[i, iq] * N_prebuild[j, iq]

        scale = 2.0 * A

        for i in range(6):
            f_global[field_ids[i]] += f_local[i] * scale
            for j in range(6):
                k = p + 6 * i + j
                K_rows[k] = field_ids[i]
                K_cols[k] = field_ids[j]
                K_values[k] = K_local[i, j] * scale

    return K_values, K_rows, K_cols, f_global


############################################################
#                 PYTHON ASSEMBLER INTERFACE               #
############################################################


def assemble_radiation_bc(
    field: Legrange2,
    face_tags: list[int],
    emissivity: float,
    T_amb: float,
    T_solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble blackbody radiation boundary condition.

    Linearized as Robin BC with temperature-dependent coefficient:
        h_rad = eps * sigma * (T^2 + T_amb^2) * (T + T_amb)

    Args:
        field: Legrange2 field
        face_tags: GMSH face tags for radiation boundary
        emissivity: surface emissivity (0 to 1)
        T_amb: ambient radiation temperature [K]
        T_solution: current temperature solution vector (n_field,)
                    used to evaluate h_rad at quadrature points

    Returns:
        K_values, K_rows, K_cols: COO stiffness contribution
        f_robin: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return _radiation_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        T_solution,
        float(emissivity),
        float(T_amb),
        field.n_field,
    )


def assemble_robin_bc(
    field: Legrange2, face_tags: list[int], h_coeff: float, T_amb: float
):
    """Assembles the Robin boundary condition which is the Convection boundary condition to an ambient temperature.

    Args:
        field (Legrange2): The problems Legrange basis function field object
        face_tags (list[int]): A list of face tag integers
        h_coeff (float): The heat flux coefficient in W/m^2
        T_amb (float): The ambient temperature
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    K_values, K_rows, K_cols = _robin_stiffness_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        float(h_coeff),
    )

    f_robin = assemble_surface_flux(field, face_tags, float(h_coeff * T_amb))

    return K_values, K_rows, K_cols, f_robin
