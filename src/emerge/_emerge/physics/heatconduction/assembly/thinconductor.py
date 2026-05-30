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
from ....mth.optimized import cross


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0, :] = B[0, 0] * data[0, :] + B[0, 1] * data[1, :] + B[0, 2] * data[2, :]
    dnew[1, :] = B[1, 0] * data[0, :] + B[1, 1] * data[1, :] + B[1, 2] * data[2, :]
    dnew[2, :] = B[2, 0] * data[0, :] + B[2, 1] * data[1, :] + B[2, 2] * data[2, :]
    return dnew


@njit(cache=True, nogil=True)
def normalize(a: np.ndarray):
    return a / ((a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5)


@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:]), cache=True, nogil=True)
def tri_coefficients(vxs, vys):

    x1, x2, x3 = vxs
    y1, y2, y3 = vys

    a1 = x2 * y3 - y2 * x3
    a2 = x3 * y1 - y3 * x1
    a3 = x1 * y2 - y1 * x2
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    sA = 0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3]) * sign
    Bs = np.array([b1, b2, b3]) * sign
    Cs = np.array([c1, c2, c3]) * sign
    return As, Bs, Cs, A


# --- Gradients ∇N
@njit(f8[:, :](f8[:], f8[:, :]), cache=True, nogil=True)
def grad_n1(coeff, coords):
    a, b, c = coeff
    xs = coords[0, :]
    ys = coords[1, :]

    out = np.empty((3, xs.shape[0]), dtype=np.float64)
    q = 4 * (b * xs + c * ys) + (4 * a - 1)
    out[0, :] = q * b
    out[1, :] = q * c
    return out


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True, nogil=True)
def grad_n2(coeff, coord):
    a1, b1, c1 = coeff[:, 0]
    a2, b2, c2 = coeff[:, 1]

    xs = coord[0, :]
    ys = coord[1, :]

    out = np.empty((3, xs.shape[0]), dtype=np.float64)
    L1 = 4 * (a1 + b1 * xs + c1 * ys)
    L2 = 4 * (a2 + b2 * xs + c2 * ys)

    out[0, :] = L2 * b1 + L1 * b2
    out[1, :] = L2 * c1 + L1 * c2

    return out


@njit(f8[:, :](f8[:, :], f8), cache=True, nogil=True)
def leg2_tri_stiff_triangle(glob_vertices: np.ndarray, kappa_t) -> np.ndarray:
    """Assembly of the legrange 2 stiffness terms for thin conductors."""
    Kmat = np.empty((6, 6), dtype=np.float64)

    # Compute local coordinate system for the triangles
    origin = glob_vertices[:, 0]
    v2 = glob_vertices[:, 1]
    v3 = glob_vertices[:, 2]

    e1 = v2 - origin
    e2 = v3 - origin
    zhat = normalize(cross(e1, e2))
    xhat = normalize(e1)
    yhat = normalize(cross(zhat, xhat))

    basis = np.zeros((3, 3), dtype=np.float64)
    basis[0, :] = xhat
    basis[1, :] = yhat
    basis[2, :] = zhat

    lcs_vertices = optim_matmul(basis, glob_vertices - origin[:, np.newaxis])

    xs = lcs_vertices[0, :]
    ys = lcs_vertices[1, :]

    # Compute the barycentric coordinate coefficients and area
    aas, bbs, ccs, A = tri_coefficients(xs, ys)

    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas / (2 * A)
    coeff[1, :] = bbs / (2 * A)
    coeff[2, :] = ccs / (2 * A)

    WEIGHTS = TRI_DPTS[0, :]
    DPTS1 = TRI_DPTS[1, :]
    DPTS2 = TRI_DPTS[2, :]
    DPTS3 = TRI_DPTS[3, :]

    # Compute all Gauss-Quadrature points
    xs_gq = xs[0] * DPTS1 + xs[1] * DPTS2 + xs[2] * DPTS3
    ys_gq = ys[0] * DPTS1 + ys[1] * DPTS2 + ys[2] * DPTS3

    coords = np.empty((2, xs_gq.shape[0]), dtype=np.float64)
    coords[0, :] = xs_gq
    coords[1, :] = ys_gq

    local_edge_map = np.array([[0, 1, 0], [1, 2, 2]], dtype=np.int64)
    for iv1 in range(6):
        if iv1 < 3:
            val_gn1 = grad_n1(coeff[:, iv1], coords)
        else:
            ie1 = local_edge_map[:, iv1 - 3]
            val_gn1 = grad_n2(coeff[:, ie1], coords)

        for iv2 in range(6):
            if iv2 < 3:
                val_gn2 = grad_n1(coeff[:, iv2], coords)
            else:
                ie2 = local_edge_map[:, iv2 - 3]
                val_gn2 = grad_n2(coeff[:, ie2], coords)

            gN1_dot_gN2 = val_gn1[0, :] * val_gn2[0, :] + val_gn1[1, :] * val_gn2[1, :]
            Kmat[iv1, iv2] = np.sum(gN1_dot_gN2 * WEIGHTS)

    Kmat = Kmat * kappa_t * 2.0 * A
    return Kmat


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8),
    cache=True,
    nogil=True,
    parallel=True,
)
def leg2_tri_stiff(nodes, tris, tri_to_field, tri_ids_2d, kappa_t):
    n_sel = tri_ids_2d.shape[1]
    nnz = n_sel * 36  # 6x6 per face

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    for idx in prange(n_sel):
        p = idx * 36
        itri = tri_ids_2d[0, idx]

        tri_nodes = nodes[:, tris[:, itri]]
        Ksub = leg2_tri_stiff_triangle(tri_nodes, kappa_t)

        field_ids = tri_to_field[:, itri]
        for i in range(6):
            rows[p + 6 * i : p + 6 * (i + 1)] = field_ids[i]
            cols[p + i : p + 36 + i : 6] = field_ids[i]

        values[p : p + 36] = Ksub.ravel()

    return values, rows, cols


def assemble_conductive_sheet(field, face_tags, kappa_t):
    """Assemble thin conductive sheet stiffness matrix contribution.

    Models lateral heat conduction through a thin metallic layer
    (e.g. copper foil, ground plane) without volume meshing.

    Args:
        field: Legrange2 field
        face_tags: GMSH face tags for the sheet surface
        kappa_t: thermal conductivity of the sheet material times thickness [W/(K)]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return leg2_tri_stiff(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        kappa_t,
    )
