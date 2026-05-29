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
from .heatflux import TRI_DPTS


@njit(f8[:, :](f8, f8, f8, f8), cache=True, nogil=True)
def _K_tri(inv_det: float, g11: float, g12: float, g22: float):

    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

    gi11 = g22 * inv_det
    gi12 = -g12 * inv_det
    gi22 = g11 * inv_det

    dLdxi = np.array([-1.0, 1.0, 0.0])
    dLdeta = np.array([-1.0, 0.0, 1.0])
    edge_vertex_1 = np.array([0, 1, 0])
    edge_vertex_2 = np.array([1, 2, 2])

    K_local = np.zeros((6, 6), dtype=np.float64)
    for iq in range(nq):
        L1 = TRI_DPTS[1, iq]
        L2 = TRI_DPTS[2, iq]
        L3 = TRI_DPTS[3, iq]
        w = weights[iq]

        Ls = np.empty(3, dtype=np.float64)
        Ls[0] = L1
        Ls[1] = L2
        Ls[2] = L3

        dNdxi = np.empty(6, dtype=np.float64)
        dNdeta = np.empty(6, dtype=np.float64)

        for iv in range(3):
            q = 4.0 * Ls[iv] - 1.0
            dNdxi[iv] = q * dLdxi[iv]
            dNdeta[iv] = q * dLdeta[iv]

        for ie in range(3):
            ei = edge_vertex_1[ie]
            ej = edge_vertex_2[ie]
            dNdxi[3 + ie] = 4.0 * (Ls[ej] * dLdxi[ei] + Ls[ei] * dLdxi[ej])
            dNdeta[3 + ie] = 4.0 * (Ls[ej] * dLdeta[ei] + Ls[ei] * dLdeta[ej])

        for i in range(6):
            for j in range(6):
                K_local[i, j] += w * (
                    dNdxi[i] * gi11 * dNdxi[j]
                    + dNdxi[i] * gi12 * dNdeta[j]
                    + dNdeta[i] * gi12 * dNdxi[j]
                    + dNdeta[i] * gi22 * dNdeta[j]
                )
    return K_local


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _conductive_sheet_builder(nodes, tris, tri_to_field, tri_ids_2d, kappa_t):
    n_sel = tri_ids_2d.shape[1]
    nnz = n_sel * 36  # 6x6 per face

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    for idx in prange(n_sel):
        p = idx * 36
        itri = tri_ids_2d[0, idx]

        # Triangle vertex coordinates
        v0 = tris[0, itri]
        v1 = tris[1, itri]
        v2 = tris[2, itri]

        x0 = nodes[0, v0]
        y0 = nodes[1, v0]
        z0 = nodes[2, v0]
        x1 = nodes[0, v1]
        y1 = nodes[1, v1]
        z1 = nodes[2, v1]
        x2 = nodes[0, v2]
        y2 = nodes[1, v2]
        z2 = nodes[2, v2]

        # Edge vectors
        e1x = x1 - x0
        e1y = y1 - y0
        e1z = z1 - z0
        e2x = x2 - x0
        e2y = y2 - y0
        e2z = z2 - z0

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        area2 = np.sqrt(nx * nx + ny * ny + nz * nz)
        area = 0.5 * area2

        g11 = e1x * e1x + e1y * e1y + e1z * e1z
        g12 = e1x * e2x + e1y * e2y + e1z * e2z
        g22 = e2x * e2x + e2y * e2y + e2z * e2z
        det_g = g11 * g22 - g12 * g12
        inv_det = 1.0 / det_g

        field_ids = tri_to_field[:, itri]

        K_local = _K_tri(inv_det, g11, g12, g22)

        scale = kappa_t * 2.0 * area

        for i in range(6):
            for j in range(6):
                k = p + 6 * i + j
                rows[k] = field_ids[i]
                cols[k] = field_ids[j]
                values[k] = K_local[i, j] * scale

    return values, rows, cols


def assemble_conductive_sheet(field, face_tags, kappa_t):
    """Assemble thin conductive sheet stiffness matrix contribution.

    Models lateral heat conduction through a thin metallic layer
    (e.g. copper foil, ground plane) without volume meshing.

    Args:
        field: Legrange2 field
        mesh: Mesh3D
        face_tags: GMSH face tags for the sheet surface
        kappa: thermal conductivity of the sheet material [W/(m·K)]
        thickness: sheet thickness [m]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return _conductive_sheet_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        kappa_t,
    )
