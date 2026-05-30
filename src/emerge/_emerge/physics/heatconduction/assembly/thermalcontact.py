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
from ....elements.leg2 import Legrange2
from ....mth.optimized import calc_area
from .convection import _tri_mass_stencil


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:], f8
    ),
    cache=True,
    nogil=True,
    parallel=False,
)
def _thermal_contact_builder(
    nodes,
    tris,
    tri_to_field_a,
    tri_to_field_b,
    tri_ids,
    h_c,
):
    """Parallel assembly of thermal contact coupling in COO format.

    The thermal contact is due to a split face where the DoF of the faces are separated between
    side A and side B.

    The assembly of this boundary condition both involves the self terms M_AA and M_BB for the
    different sides plus a coupling term M_AB and M_BA.

    The matrix is assembled in COO form which includes:
       values: np.ndarray(float)
       rows: np.ndarray(int)
       cols: np.ndarray(int)

    Returns:
        values, rows, cols: COO data
    """
    n_triangles = tri_ids.shape[0]
    nnz = n_triangles * 4 * (6 * 6)  # 4 x (6x6 matrix blocks)

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    M = _tri_mass_stencil()

    for idx in prange(n_triangles):
        p = idx * 144
        itri = tri_ids[idx]

        # Triangle area
        iv1 = tris[0, itri]
        iv2 = tris[1, itri]
        iv3 = tris[2, itri]

        A = calc_area(nodes[:, iv1], nodes[:, iv2], nodes[:, iv3])

        # Scale: weights sum to 1/2, so multiply by 2*area*h_c
        scale = h_c * 2.0 * A

        # DOF indices for both sides
        fids_a = tri_to_field_a[:, idx]
        fids_b = tri_to_field_b[:, idx]

        # Write 4 blocks of 6x6
        for i in range(6):
            for j in range(6):
                mij = M[i, j] * scale

                stride = 6 * i + j
                k = p + stride
                rows[k] = fids_a[i]
                cols[k] = fids_a[j]
                values[k] = mij

                k = p + 36 + stride
                rows[k] = fids_b[i]
                cols[k] = fids_b[j]
                values[k] = mij

                k = p + 72 + stride
                rows[k] = fids_a[i]
                cols[k] = fids_b[j]
                values[k] = -mij

                k = p + 108 + stride
                rows[k] = fids_b[i]
                cols[k] = fids_a[j]
                values[k] = -mij

    return values, rows, cols


def assemble_thermal_contact(field: Legrange2, face_tags: np.ndarray, h_c: float):
    """Assemble thermal contact coupling between two volume regions.

    Args:
        field: Legrange2 field (with _dof_mapping populated)
        face_tags: GMSH face tags of the contact interface
        h_c: thermal contact conductance [W/(m^2K)]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)
    n_triangles = len(tri_ids)

    # Build the DOF mapping arrays from dict
    dof_map = field._dof_mapping

    # Build tri_to_field for A side (original) and B side (remapped)
    tri_to_field_a = np.empty((6, n_triangles), dtype=np.int64)
    tri_to_field_b = np.empty((6, n_triangles), dtype=np.int64)

    for i in range(n_triangles):
        itri = tri_ids[i]

        fids_a = field.tri_to_field[:, itri].copy()
        tri_to_field_a[:, i] = fids_a

        fids_b = np.empty(6, dtype=np.int64)
        for j in range(6):
            fids_b[j] = dof_map[int(fids_a[j])]
        tri_to_field_b[:, i] = fids_b

    return _thermal_contact_builder(
        mesh.nodes,
        mesh.tris,
        tri_to_field_a,
        tri_to_field_b,
        tri_ids,
        float(h_c),
    )
