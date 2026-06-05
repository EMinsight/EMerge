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
# Last Cleanup: 2025-03-12
from numba import njit, f8, c16, i8, types, prange  # type: ignore
from numba import get_thread_id as get_thread_id
from ..volakis import (
    SCALE_LENGTH,
    _ne1,
    _ne2,
    _ne1_curl,
    _ne2_curl,
    _nf1,
    _nf2,
    _nf1_curl,
    _nf2_curl,
)


import numpy as np
from ...mth.optimized import compute_distances

EPS = 1e-8


@njit(f8[:, :](f8[:, :]), cache=True, nogil=True)
def incl_length(lengths):
    if SCALE_LENGTH == False:
        lengths[:, :] = 1.0
    return lengths


@njit(f8[:, :](f8[:, :]), cache=True, nogil=True)
def matinv(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.empty((3, 3), dtype=np.float64)

    det = (
        M[0, 0] * M[1, 1] * M[2, 2]
        - M[0, 0] * M[1, 2] * M[2, 1]
        - M[0, 1] * M[1, 0] * M[2, 2]
        + M[0, 1] * M[1, 2] * M[2, 0]
        + M[0, 2] * M[1, 0] * M[2, 1]
        - M[0, 2] * M[1, 1] * M[2, 0]
    )
    out[0, 0] = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]
    out[0, 1] = -M[0, 1] * M[2, 2] + M[0, 2] * M[2, 1]
    out[0, 2] = M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]
    out[1, 0] = -M[1, 0] * M[2, 2] + M[1, 2] * M[2, 0]
    out[1, 1] = M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]
    out[1, 2] = -M[0, 0] * M[1, 2] + M[0, 2] * M[1, 0]
    out[2, 0] = M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]
    out[2, 1] = -M[0, 0] * M[2, 1] + M[0, 1] * M[2, 0]
    out[2, 2] = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    out = out / det
    return out


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]),
    cache=True,
    nogil=True,
)
def tet_coefficients(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(
        -x1 * y2 * z3 / 6
        + x1 * y2 * z4 / 6
        + x1 * y3 * z2 / 6
        - x1 * y3 * z4 / 6
        - x1 * y4 * z2 / 6
        + x1 * y4 * z3 / 6
        + x2 * y1 * z3 / 6
        - x2 * y1 * z4 / 6
        - x2 * y3 * z1 / 6
        + x2 * y3 * z4 / 6
        + x2 * y4 * z1 / 6
        - x2 * y4 * z3 / 6
        - x3 * y1 * z2 / 6
        + x3 * y1 * z4 / 6
        + x3 * y2 * z1 / 6
        - x3 * y2 * z4 / 6
        - x3 * y4 * z1 / 6
        + x3 * y4 * z2 / 6
        + x4 * y1 * z2 / 6
        - x4 * y1 * z3 / 6
        - x4 * y2 * z1 / 6
        + x4 * y2 * z3 / 6
        + x4 * y3 * z1 / 6
        - x4 * y3 * z2 / 6
    )

    aas[0] = (
        x2 * y3 * z4
        - x2 * y4 * z3
        - x3 * y2 * z4
        + x3 * y4 * z2
        + x4 * y2 * z3
        - x4 * y3 * z2
    )
    aas[1] = (
        -x1 * y3 * z4
        + x1 * y4 * z3
        + x3 * y1 * z4
        - x3 * y4 * z1
        - x4 * y1 * z3
        + x4 * y3 * z1
    )
    aas[2] = (
        x1 * y2 * z4
        - x1 * y4 * z2
        - x2 * y1 * z4
        + x2 * y4 * z1
        + x4 * y1 * z2
        - x4 * y2 * z1
    )
    aas[3] = (
        -x1 * y2 * z3
        + x1 * y3 * z2
        + x2 * y1 * z3
        - x2 * y3 * z1
        - x3 * y1 * z2
        + x3 * y2 * z1
    )
    bbs[0] = -y2 * z3 + y2 * z4 + y3 * z2 - y3 * z4 - y4 * z2 + y4 * z3
    bbs[1] = y1 * z3 - y1 * z4 - y3 * z1 + y3 * z4 + y4 * z1 - y4 * z3
    bbs[2] = -y1 * z2 + y1 * z4 + y2 * z1 - y2 * z4 - y4 * z1 + y4 * z2
    bbs[3] = y1 * z2 - y1 * z3 - y2 * z1 + y2 * z3 + y3 * z1 - y3 * z2
    ccs[0] = x2 * z3 - x2 * z4 - x3 * z2 + x3 * z4 + x4 * z2 - x4 * z3
    ccs[1] = -x1 * z3 + x1 * z4 + x3 * z1 - x3 * z4 - x4 * z1 + x4 * z3
    ccs[2] = x1 * z2 - x1 * z4 - x2 * z1 + x2 * z4 + x4 * z1 - x4 * z2
    ccs[3] = -x1 * z2 + x1 * z3 + x2 * z1 - x2 * z3 - x3 * z1 + x3 * z2
    dds[0] = -x2 * y3 + x2 * y4 + x3 * y2 - x3 * y4 - x4 * y2 + x4 * y3
    dds[1] = x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3
    dds[2] = -x1 * y2 + x1 * y4 + x2 * y1 - x2 * y4 - x4 * y1 + x4 * y2
    dds[3] = x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2

    return aas, bbs, ccs, dds, V


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

    # A = 0.5*(b1*c2 - b2*c1)
    sA = 0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3]) * sign
    Bs = np.array([b1, b2, b3]) * sign
    Cs = np.array([c1, c2, c3]) * sign
    return As, Bs, Cs, A


@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex 0.1005964238ers of one tetrahedron, in *its* order
        (v0, v1, v2, v3).

    triangle_ids : 2-D int64 array (nTri × 3)
        Each row is a global-ID triple of one face that belongs to this tet.

    Returns
    -------
    local_tris   : 2-D int64 array (nTri × 3)
        Same triangles, but every entry replaced by the local index
        0,1,2,3 that the vertex has inside this tetrahedron.
        (Guaranteed to be ∈{0,1,2,3}; no -1 ever appears if the input
        really belongs to the tet.)
    """
    ndim = triangle_ids.shape[0]
    ntri = triangle_ids.shape[1]
    out = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):  # each triangle
        for j in range(ndim):  # each vertex in that triangle
            gid = triangle_ids[j, t]  # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k  # store local index 0-3
                    break  # stop the k-loop

    return out


@njit(types.Tuple((i8[:], i8[:], i8[:]))(i8[:]), cache=True, nogil=True)
def get_group_indices(assigned_sorted):
    # Count how many unique tets we have
    if len(assigned_sorted) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    # Pre-calculate unique count to allocate arrays
    n_unique = 1
    for i in range(1, len(assigned_sorted)):
        if assigned_sorted[i] != assigned_sorted[i - 1]:
            n_unique += 1

    unique_tets = np.empty(n_unique, dtype=assigned_sorted.dtype)
    first_indices = np.empty(n_unique, dtype=np.int64)
    last_indices = np.empty(n_unique, dtype=np.int64)

    # Fill the arrays
    curr_idx = 0
    unique_tets[0] = assigned_sorted[0]
    first_indices[0] = 0

    for i in range(1, len(assigned_sorted)):
        if assigned_sorted[i] != assigned_sorted[i - 1]:
            last_indices[curr_idx] = i
            curr_idx += 1
            unique_tets[curr_idx] = assigned_sorted[i]
            first_indices[curr_idx] = i

    last_indices[curr_idx] = len(assigned_sorted)

    return unique_tets, first_indices, last_indices


@njit(
    types.Tuple((c16[:], c16[:], c16[:]))(
        f8[:, :],
        c16[:],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        f8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:],
        i8[:],
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def ned2_tet_interp(
    coords: np.ndarray,
    solutions: np.ndarray,
    tets: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    nodes: np.ndarray,
    tet_to_field: np.ndarray,
    tet_to_edge: np.ndarray,
    tet_to_tri: np.ndarray,
    tetids: np.ndarray,
    tet_mapping: np.ndarray,
):
    # """Nedelec 2 tetrahedral interpolation"""
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]

    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]

    Ex = np.zeros((nNodes,), dtype=np.complex128)
    Ey = np.zeros((nNodes,), dtype=np.complex128)
    Ez = np.zeros((nNodes,), dtype=np.complex128)
    setnan = np.zeros((nNodes,), dtype=np.int64)

    sort_idx = np.argsort(tet_mapping)
    xs_s = xs[sort_idx]
    ys_s = ys[sort_idx]
    zs_s = zs[sort_idx]
    assigned_sorted = tet_mapping[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side="right")

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]

        if start == end:
            continue

        xvs = nodes[0, tets[:, itet]]
        yvs = nodes[1, tets[:, itet]]
        zvs = nodes[2, tets[:, itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        Ds = incl_length(compute_distances(xvs, yvs, zvs))

        g_node_ids = tets[:, itet]
        l_edge_ids = local_mapping(g_node_ids, edges[:, tet_to_edge[:, itet]])
        l_tri_ids = local_mapping(g_node_ids, tris[:, tet_to_tri[:, itet]])

        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]

        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]

        coeff = np.empty((4, 4), dtype=np.float64)
        coeff[0, :] = a_s / (6 * V)
        coeff[1, :] = b_s / (6 * V)
        coeff[2, :] = c_s / (6 * V)
        coeff[3, :] = d_s / (6 * V)

        # Pack all quadrature points for this tet into coords array
        npts = end - start
        pt_coords = np.empty((3, npts), dtype=np.float64)
        pt_coords[0, :] = xs_s[start:end]
        pt_coords[1, :] = ys_s[start:end]
        pt_coords[2, :] = zs_s[start:end]

        for ie in range(6):
            i1 = l_edge_ids[0, ie]
            j1 = l_edge_ids[1, ie]
            L = Ds[i1, j1]
            F = (
                _ne1(coeff, pt_coords, i1, j1, 0) * Em1s[ie]
                + _ne2(coeff, pt_coords, i1, j1, 0) * Em2s[ie]
            )
            for i in range(npts):
                idx = sort_idx[start + i]
                Ex[idx] += L * F[0, i]
                Ey[idx] += L * F[1, i]
                Ez[idx] += L * F[2, i]

        for ie in range(4):
            i1 = l_tri_ids[0, ie]
            j1 = l_tri_ids[1, ie]
            k1 = l_tri_ids[2, ie]
            L1 = Ds[i1, k1]
            L2 = Ds[i1, j1]
            F = (
                _nf1(coeff, pt_coords, i1, j1, k1) * Ef1s[ie] * L1
                + _nf2(coeff, pt_coords, i1, j1, k1) * Ef2s[ie] * L2
            )
            for i in range(npts):
                idx = sort_idx[start + i]
                Ex[idx] += F[0, i]
                Ey[idx] += F[1, i]
                Ez[idx] += F[2, i]

        inside = sort_idx[start:end]
        setnan[inside] = 1

    Ex[setnan == 0] = np.nan
    Ey[setnan == 0] = np.nan
    Ez[setnan == 0] = np.nan
    return Ex, Ey, Ez


@njit(
    types.Tuple((c16[:], c16[:], c16[:]))(
        f8[:, :],
        c16[:],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        f8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        c16[:],
        i8[:],
        i8[:],
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def ned2_tet_interp_curl(
    coords: np.ndarray,
    solutions: np.ndarray,
    tets: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    nodes: np.ndarray,
    tet_to_field: np.ndarray,
    tet_to_edge: np.ndarray,
    tet_to_tri: np.ndarray,
    c: np.ndarray,
    tetids: np.ndarray,
    tet_mapping: np.ndarray,
):
    """Nedelec 2 tetrahedral interpolation of the analytic curl"""
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]

    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]

    Ex = np.zeros((nNodes,), dtype=np.complex128)
    Ey = np.zeros((nNodes,), dtype=np.complex128)
    Ez = np.zeros((nNodes,), dtype=np.complex128)
    setnan = np.zeros((nNodes,), dtype=np.int64)

    sort_idx = np.argsort(tet_mapping)
    xs_s = xs[sort_idx]
    ys_s = ys[sort_idx]
    zs_s = zs[sort_idx]
    assigned_sorted = tet_mapping[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side="right")

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]

        if start == end:
            continue

        xvs = nodes[0, tets[:, itet]]
        yvs = nodes[1, tets[:, itet]]
        zvs = nodes[2, tets[:, itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        Ds = incl_length(compute_distances(xvs, yvs, zvs))

        g_node_ids = tets[:, itet]
        l_edge_ids = local_mapping(g_node_ids, edges[:, tet_to_edge[:, itet]])
        l_tri_ids = local_mapping(g_node_ids, tris[:, tet_to_tri[:, itet]])

        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]

        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]

        const = c[itet]

        coeff = np.empty((4, 4), dtype=np.float64)
        coeff[0, :] = a_s / (6 * V)
        coeff[1, :] = b_s / (6 * V)
        coeff[2, :] = c_s / (6 * V)
        coeff[3, :] = d_s / (6 * V)

        npts = end - start
        pt_coords = np.empty((3, npts), dtype=np.float64)
        pt_coords[0, :] = xs_s[start:end]
        pt_coords[1, :] = ys_s[start:end]
        pt_coords[2, :] = zs_s[start:end]

        for ie in range(6):
            i1 = l_edge_ids[0, ie]
            j1 = l_edge_ids[1, ie]
            L = Ds[i1, j1]
            F = (
                _ne1_curl(coeff, pt_coords, i1, j1, 0) * Em1s[ie]
                + _ne2_curl(coeff, pt_coords, i1, j1, 0) * Em2s[ie]
            )
            for i in range(npts):
                idx = sort_idx[start + i]
                Ex[idx] += const * L * F[0, i]
                Ey[idx] += const * L * F[1, i]
                Ez[idx] += const * L * F[2, i]

        for ie in range(4):
            i1 = l_tri_ids[0, ie]
            j1 = l_tri_ids[1, ie]
            k1 = l_tri_ids[2, ie]
            L1 = Ds[i1, k1]
            L2 = Ds[i1, j1]
            F = (
                _nf1_curl(coeff, pt_coords, i1, j1, k1) * Ef1s[ie] * L1
                + _nf2_curl(coeff, pt_coords, i1, j1, k1) * Ef2s[ie] * L2
            )
            for i in range(npts):
                idx = sort_idx[start + i]
                Ex[idx] += const * F[0, i]
                Ey[idx] += const * F[1, i]
                Ez[idx] += const * F[2, i]

        inside = sort_idx[start:end]
        setnan[inside] = 1

    Ex[setnan == 0] = np.nan
    Ey[setnan == 0] = np.nan
    Ez[setnan == 0] = np.nan
    return Ex, Ey, Ez


############################################################
#                    LEGRANGE FUNCTIONS                   #
############################################################


@njit(
    f8[:](f8[:, :], f8[:], i8[:, :], i8[:, :], f8[:, :], i8[:, :], i8[:], i8[:]),
    cache=True,
    nogil=True,
    parallel=False,
)
def leg2_tet_interp(
    coords: np.ndarray,
    solutions: np.ndarray,
    tets: np.ndarray,
    edges: np.ndarray,
    nodes: np.ndarray,
    tet_to_field: np.ndarray,
    tetids: np.ndarray,
    tet_mapping: np.ndarray,
):
    """Lagrange P2 tetrahedral scalar interpolation.

    Basis functions in barycentric coords L1..L4:
      Vertex i:        N_i = L_i(2L_i - 1)
      Edge midpoint ij: N_ij = 4 L_i L_j
    """
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]

    result = np.full((nNodes,), np.nan, dtype=np.float64)

    sort_idx = np.argsort(tet_mapping)
    assigned_sorted = tet_mapping[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side="right")

    # --- Phase 2: evaluate P2 basis functions ---
    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]

        if start == end:
            continue

        xvs = nodes[0, tets[:, itet]]
        yvs = nodes[1, tets[:, itet]]
        zvs = nodes[2, tets[:, itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)

        # Normalisation factor: L_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
        inv6V = 1.0 / (6.0 * V)

        # Get edge local indices (which two vertices each edge connects)
        l_edge_ids = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])
        field_ids = tet_to_field[:, itet]
        Tloc = solutions[field_ids]  # 10 DOFs: 4 vertex + 6 edge

        for i in range(start, end):
            idx = sort_idx[i]
            x = xs[idx]
            y = ys[idx]
            z = zs[idx]

            # Barycentric coordinates
            L1 = (a_s[0] + b_s[0] * x + c_s[0] * y + d_s[0] * z) * inv6V
            L2 = (a_s[1] + b_s[1] * x + c_s[1] * y + d_s[1] * z) * inv6V
            L3 = (a_s[2] + b_s[2] * x + c_s[2] * y + d_s[2] * z) * inv6V
            L4 = (a_s[3] + b_s[3] * x + c_s[3] * y + d_s[3] * z) * inv6V

            Ls = np.empty(4, dtype=np.float64)
            Ls[0] = L1
            Ls[1] = L2
            Ls[2] = L3
            Ls[3] = L4

            # Vertex basis: N_i = L_i(2L_i - 1)
            val = (
                Tloc[0] * L1 * (2.0 * L1 - 1.0)
                + Tloc[1] * L2 * (2.0 * L2 - 1.0)
                + Tloc[2] * L3 * (2.0 * L3 - 1.0)
                + Tloc[3] * L4 * (2.0 * L4 - 1.0)
            )

            # Edge basis: N_ij = 4 L_i L_j
            for ie in range(6):
                li = Ls[l_edge_ids[0, ie]]
                lj = Ls[l_edge_ids[1, ie]]
                val += Tloc[4 + ie] * 4.0 * li * lj

            result[idx] = val

    return result


@njit(
    types.Tuple((f8[:], f8[:], f8[:]))(
        f8[:, :], f8[:], i8[:, :], i8[:, :], f8[:, :], i8[:, :], i8[:], i8[:]
    ),
    cache=True,
    nogil=True,
    parallel=False,
)
def leg2_tet_interp_grad(
    coords, solutions, tets, edges, nodes, tet_to_field, tetids, tet_mapping
):
    """Lagrange P2 tetrahedral gradient interpolation.

    Returns (dT/dx, dT/dy, dT/dz) at each coordinate point.

    Gradient of basis functions:
      Vertex i:        grad N_i = (4*L_i - 1) * grad L_i
      Edge midpoint ij: grad N_ij = 4*(L_j * grad L_i + L_i * grad L_j)

    where grad L_i = (b_i, c_i, d_i) / (6V)
    """
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]

    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]

    gx = np.full(nNodes, np.nan, dtype=np.float64)
    gy = np.full(nNodes, np.nan, dtype=np.float64)
    gz = np.full(nNodes, np.nan, dtype=np.float64)

    sort_idx = np.argsort(tet_mapping)
    assigned_sorted = tet_mapping[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side="right")

    # --- Phase 2: evaluate gradient ---
    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]

        if start == end:
            continue

        xvs = nodes[0, tets[:, itet]]
        yvs = nodes[1, tets[:, itet]]
        zvs = nodes[2, tets[:, itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        inv6V = 1.0 / (6.0 * V)

        # Precompute grad L_i = (b_i, c_i, d_i) / 6V
        gb = np.empty(4, dtype=np.float64)
        gc = np.empty(4, dtype=np.float64)
        gd = np.empty(4, dtype=np.float64)
        for iv in range(4):
            gb[iv] = b_s[iv] * inv6V
            gc[iv] = c_s[iv] * inv6V
            gd[iv] = d_s[iv] * inv6V

        l_edge_ids = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])

        field_ids = tet_to_field[:, itet]
        Tloc = solutions[field_ids]

        for i in range(start, end):
            idx = sort_idx[i]
            x = xs[idx]
            y = ys[idx]
            z = zs[idx]

            # Barycentric coordinates
            Ls = np.empty(4, dtype=np.float64)
            for iv in range(4):
                Ls[iv] = (a_s[iv] + b_s[iv] * x + c_s[iv] * y + d_s[iv] * z) * inv6V

            vx = 0.0
            vy = 0.0
            vz = 0.0

            # Vertex basis: grad N_i = (4*L_i - 1) * grad L_i
            for iv in range(4):
                q = (4.0 * Ls[iv] - 1.0) * Tloc[iv]
                vx += q * gb[iv]
                vy += q * gc[iv]
                vz += q * gd[iv]

            # Edge basis: grad N_ij = 4*(L_j * grad L_i + L_i * grad L_j)
            for ie in range(6):
                ei = l_edge_ids[0, ie]
                ej = l_edge_ids[1, ie]
                T_e = Tloc[4 + ie]
                Li = Ls[ei]
                Lj = Ls[ej]
                # 4 * (Lj * gradLi + Li * gradLj)
                vx += T_e * 4.0 * (Lj * gb[ei] + Li * gb[ej])
                vy += T_e * 4.0 * (Lj * gc[ei] + Li * gc[ej])
                vz += T_e * 4.0 * (Lj * gd[ei] + Li * gd[ej])

            gx[idx] = vx
            gy[idx] = vy
            gz[idx] = vz

    return gx, gy, gz


@njit(
    types.Tuple((c16[:], c16[:], c16[:]))(
        f8[:, :], c16[:], i8[:, :], f8[:, :], i8[:, :]
    ),
    cache=True,
    nogil=True,
)
def ned2_tri_interp(
    coords: np.ndarray,
    solutions: np.ndarray,
    tris: np.ndarray,
    nodes: np.ndarray,
    tri_to_field: np.ndarray,
):
    """Nedelec 2 tetrahedral interpolation"""
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0, :]
    ys = coords[1, :]

    Ex = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes,), np.nan, dtype=np.complex128)

    nodes = nodes[:2, :]

    l_edge_ids = np.array([[0, 1, 0], [1, 2, 2]])

    for itri in range(tris.shape[1]):
        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:, iv1]
        v2 = nodes[:, iv2]
        v3 = nodes[:, iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2, 2))
        blocal[:, 0] = bv1
        blocal[:, 1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:, np.newaxis]
        coords_local = basis @ (coords_offset)

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = (
            ((coords_local[0, :] + coords_local[1, :]) <= 1 + EPS)
            & (coords_local[0, :] >= -EPS)
            & (coords_local[1, :] >= -EPS)
        )

        if inside.sum() == 0:
            continue

        ######### INSIDE THE TETRAHEDRON #########

        x = xs[inside == 1]
        y = ys[inside == 1]

        xvs = nodes[0, tris[:, itri]]
        yvs = nodes[1, tris[:, itri]]

        Ds = compute_distances(xvs, yvs, 0 * xvs)

        L1 = Ds[0, 1]
        L2 = Ds[1, 2]
        L3 = Ds[0, 2]

        mult = np.array([L1, L2, L3, L3, L1, L2, L3, L1])

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)

        Etri = Etri * mult

        Em1s = Etri[:3]
        Ef1s = Etri[3]
        Em2s = Etri[4:7]
        Ef2s = Etri[7]

        Exl = np.zeros(x.shape, dtype=np.complex128)
        Eyl = np.zeros(x.shape, dtype=np.complex128)

        for ie in range(3):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]

            ex = (
                (Em1 * (a1 + b1 * x + c1 * y) + Em2 * (a2 + b2 * x + c2 * y))
                * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                / (8 * A**3)
            )
            ey = (
                (Em1 * (a1 + b1 * x + c1 * y) + Em2 * (a2 + b2 * x + c2 * y))
                * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                / (8 * A**3)
            )

            Exl += ex
            Eyl += ey

        Em1, Em2 = Ef1s, Ef2s
        triids = np.array([0, 1, 2])

        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]

        ex = (
            -Em1
            * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
            * (a2 + b2 * x + c2 * y)
            + Em2
            * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
            * (a3 + b3 * x + c3 * y)
        ) / (8 * A**3)
        ey = (
            -Em1
            * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
            * (a2 + b2 * x + c2 * y)
            + Em2
            * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
            * (a3 + b3 * x + c3 * y)
        ) / (8 * A**3)

        Exl += ex
        Eyl += ey

        Ex[inside] = Exl
        Ey[inside] = Eyl
    return Ex, Ey, Ez


@njit(
    types.Tuple((c16[:], c16[:], c16[:]))(
        f8[:, :], c16[:], i8[:, :], f8[:, :], i8[:, :]
    ),
    cache=True,
    nogil=True,
)
def ned2_tri_interp_full(
    coords: np.ndarray,
    solutions: np.ndarray,
    tris: np.ndarray,
    nodes: np.ndarray,
    tri_to_field: np.ndarray,
):
    """Nedelec 2 tetrahedral interpolation"""
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0, :]
    ys = coords[1, :]

    Ex = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes,), np.nan, dtype=np.complex128)

    nodes = nodes[:2, :]

    for itri in range(tris.shape[1]):
        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:, iv1]
        v2 = nodes[:, iv2]
        v3 = nodes[:, iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2, 2))
        blocal[:, 0] = bv1
        blocal[:, 1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:, np.newaxis]
        coords_local = basis @ (coords_offset)

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = (
            ((coords_local[0, :] + coords_local[1, :]) <= 1.0 + EPS)
            & (coords_local[0, :] >= -EPS)
            & (coords_local[1, :] >= -EPS)
        )

        if inside.sum() == 0:
            continue

        ######### INSIDE THE TRIANGLE #########

        x = xs[inside == 1]
        y = ys[inside == 1]

        xvs = nodes[0, tris[:, itri]]
        yvs = nodes[1, tris[:, itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s

        # New Nedelec-1 order 2 formulation
        ex = (
            -2
            * A
            * (
                e1 * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                + e2 * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                + e3 * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
            )
            - e4
            * (
                (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                * (a2 + b2 * x + c2 * y)
                + (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                * (a1 + b1 * x + c1 * y)
            )
            - e5
            * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
            * (a1 - a2 + b1 * x - b2 * x + c1 * y - c2 * y)
            - e6
            * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
            * (a2 - a3 + b2 * x - b3 * x + c2 * y - c3 * y)
            - e7
            * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
            * (a1 - a3 + b1 * x - b3 * x + c1 * y - c3 * y)
            + e8
            * (
                (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                * (a3 + b3 * x + c3 * y)
                + (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                * (a2 + b2 * x + c2 * y)
            )
        ) / (8 * A**3)
        ey = (
            -2
            * A
            * (
                e1 * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                + e2 * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                + e3 * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
            )
            - e4
            * (
                (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                * (a2 + b2 * x + c2 * y)
                + (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                * (a1 + b1 * x + c1 * y)
            )
            - e5
            * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
            * (a1 - a2 + b1 * x - b2 * x + c1 * y - c2 * y)
            - e6
            * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
            * (a2 - a3 + b2 * x - b3 * x + c2 * y - c3 * y)
            - e7
            * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
            * (a1 - a3 + b1 * x - b3 * x + c1 * y - c3 * y)
            + e8
            * (
                (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                * (a3 + b3 * x + c3 * y)
                + (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                * (a2 + b2 * x + c2 * y)
            )
        ) / (8 * A**3)
        ez = (
            -e10 * (a2 + b2 * x + c2 * y) * (A - a2 - b2 * x - c2 * y) / 2
            - e11 * (a3 + b3 * x + c3 * y) * (A - a3 - b3 * x - c3 * y) / 2
            + e12 * (a1 + b1 * x + c1 * y) * (a2 + b2 * x + c2 * y)
            + e13 * (a2 + b2 * x + c2 * y) * (a3 + b3 * x + c3 * y)
            + e14 * (a1 + b1 * x + c1 * y) * (a3 + b3 * x + c3 * y)
            - e9 * (a1 + b1 * x + c1 * y) * (A - a1 - b1 * x - c1 * y) / 2
        ) / A**2
        Ex[inside] = ex
        Ey[inside] = ey
        Ez[inside] = ez
    return Ex, Ey, Ez


@njit(
    types.Tuple((c16[:], c16[:], c16[:]))(
        f8[:, :], c16[:], i8[:, :], f8[:, :], i8[:, :], c16[:, :, :], c16
    ),
    cache=True,
    nogil=True,
)
def ned2_tri_interp_curl(
    coords: np.ndarray,
    solutions: np.ndarray,
    tris: np.ndarray,
    nodes: np.ndarray,
    tri_to_field: np.ndarray,
    diadic: np.ndarray,
    beta: float,
):
    """Nedelec 2 tetrahedral interpolation"""
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0, :]
    ys = coords[1, :]
    jB = 1j * beta
    Ex = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes,), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes,), np.nan, dtype=np.complex128)

    nodes = nodes[:2, :]

    for itri in range(tris.shape[1]):
        dc = diadic[:, :, itri]

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:, iv1]
        v2 = nodes[:, iv2]
        v3 = nodes[:, iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2, 2))
        blocal[:, 0] = bv1
        blocal[:, 1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:, np.newaxis]
        coords_local = basis @ (coords_offset)

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = (
            ((coords_local[0, :] + coords_local[1, :]) <= 1.0 + EPS)
            & (coords_local[0, :] >= -EPS)
            & (coords_local[1, :] >= -EPS)
        )

        if inside.sum() == 0:
            continue

        ######### INSIDE THE TETRAHEDRON #########

        x = xs[inside == 1]
        y = ys[inside == 1]

        xvs = nodes[0, tris[:, itri]]
        yvs = nodes[1, tris[:, itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s

        # New Nedelec-1 order 2 formulation
        hx = (
            4
            * A
            * (
                2 * c1 * e12 * (a2 + b2 * x + c2 * y)
                + 2 * c1 * e14 * (a3 + b3 * x + c3 * y)
                + c1 * e9 * (a1 + b1 * x + c1 * y)
                - c1 * e9 * (A - a1 - b1 * x - c1 * y)
                + c2 * e10 * (a2 + b2 * x + c2 * y)
                - c2 * e10 * (A - a2 - b2 * x - c2 * y)
                + 2 * c2 * e12 * (a1 + b1 * x + c1 * y)
                + 2 * c2 * e13 * (a3 + b3 * x + c3 * y)
                + c3 * e11 * (a3 + b3 * x + c3 * y)
                - c3 * e11 * (A - a3 - b3 * x - c3 * y)
                + 2 * c3 * e13 * (a2 + b2 * x + c2 * y)
                + 2 * c3 * e14 * (a1 + b1 * x + c1 * y)
            )
            + jB
            * (
                2
                * A
                * (
                    e1 * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                    + e2 * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                    + e3 * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                )
                + e4
                * (
                    (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                    * (a2 + b2 * x + c2 * y)
                    + (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                    * (a1 + b1 * x + c1 * y)
                )
                + e5
                * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                * (a1 - a2 + b1 * x - b2 * x + c1 * y - c2 * y)
                + e6
                * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                * (a2 - a3 + b2 * x - b3 * x + c2 * y - c3 * y)
                + e7
                * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                * (a1 - a3 + b1 * x - b3 * x + c1 * y - c3 * y)
                - e8
                * (
                    (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                    * (a3 + b3 * x + c3 * y)
                    + (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                    * (a2 + b2 * x + c2 * y)
                )
            )
        ) / (8 * A**3)
        hy = (
            4
            * A
            * (
                -2 * b1 * e12 * (a2 + b2 * x + c2 * y)
                - 2 * b1 * e14 * (a3 + b3 * x + c3 * y)
                - b1 * e9 * (a1 + b1 * x + c1 * y)
                + b1 * e9 * (A - a1 - b1 * x - c1 * y)
                - b2 * e10 * (a2 + b2 * x + c2 * y)
                + b2 * e10 * (A - a2 - b2 * x - c2 * y)
                - 2 * b2 * e12 * (a1 + b1 * x + c1 * y)
                - 2 * b2 * e13 * (a3 + b3 * x + c3 * y)
                - b3 * e11 * (a3 + b3 * x + c3 * y)
                + b3 * e11 * (A - a3 - b3 * x - c3 * y)
                - 2 * b3 * e13 * (a2 + b2 * x + c2 * y)
                - 2 * b3 * e14 * (a1 + b1 * x + c1 * y)
            )
            - jB
            * (
                2
                * A
                * (
                    e1 * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                    + e2 * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                    + e3 * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                )
                + e4
                * (
                    (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                    * (a2 + b2 * x + c2 * y)
                    + (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                    * (a1 + b1 * x + c1 * y)
                )
                + e5
                * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                * (a1 - a2 + b1 * x - b2 * x + c1 * y - c2 * y)
                + e6
                * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                * (a2 - a3 + b2 * x - b3 * x + c2 * y - c3 * y)
                + e7
                * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                * (a1 - a3 + b1 * x - b3 * x + c1 * y - c3 * y)
                - e8
                * (
                    (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                    * (a3 + b3 * x + c3 * y)
                    + (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                    * (a2 + b2 * x + c2 * y)
                )
            )
        ) / (8 * A**3)
        hz = (
            4
            * A
            * (
                e1 * (b1 * c2 - b2 * c1)
                + e2 * (b2 * c3 - b3 * c2)
                + e3 * (b1 * c3 - b3 * c1)
            )
            - e4
            * (
                b1 * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
                + b2 * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                - (b1 * c3 - b3 * c1) * (a2 + b2 * x + c2 * y)
                - (b2 * c3 - b3 * c2) * (a1 + b1 * x + c1 * y)
            )
            + e4
            * (
                c1 * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
                + c2 * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                + (b1 * c3 - b3 * c1) * (a2 + b2 * x + c2 * y)
                + (b2 * c3 - b3 * c2) * (a1 + b1 * x + c1 * y)
            )
            - e5
            * (b1 - b2)
            * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
            + e5
            * (c1 - c2)
            * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
            + 2
            * e5
            * (b1 * c2 - b2 * c1)
            * (a1 - a2 + b1 * x - b2 * x + c1 * y - c2 * y)
            - e6
            * (b2 - b3)
            * (c2 * (a3 + b3 * x + c3 * y) - c3 * (a2 + b2 * x + c2 * y))
            + e6
            * (c2 - c3)
            * (b2 * (a3 + b3 * x + c3 * y) - b3 * (a2 + b2 * x + c2 * y))
            + 2
            * e6
            * (b2 * c3 - b3 * c2)
            * (a2 - a3 + b2 * x - b3 * x + c2 * y - c3 * y)
            - e7
            * (b1 - b3)
            * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
            + e7
            * (c1 - c3)
            * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
            + 2
            * e7
            * (b1 * c3 - b3 * c1)
            * (a1 - a3 + b1 * x - b3 * x + c1 * y - c3 * y)
            + e8
            * (
                b2 * (c1 * (a3 + b3 * x + c3 * y) - c3 * (a1 + b1 * x + c1 * y))
                + b3 * (c1 * (a2 + b2 * x + c2 * y) - c2 * (a1 + b1 * x + c1 * y))
                - (b1 * c2 - b2 * c1) * (a3 + b3 * x + c3 * y)
                - (b1 * c3 - b3 * c1) * (a2 + b2 * x + c2 * y)
            )
            - e8
            * (
                c2 * (b1 * (a3 + b3 * x + c3 * y) - b3 * (a1 + b1 * x + c1 * y))
                + c3 * (b1 * (a2 + b2 * x + c2 * y) - b2 * (a1 + b1 * x + c1 * y))
                + (b1 * c2 - b2 * c1) * (a3 + b3 * x + c3 * y)
                + (b1 * c3 - b3 * c1) * (a2 + b2 * x + c2 * y)
            )
        ) / (8 * A**3)

        Ex[inside] = hx * dc[0, 0]
        Ey[inside] = hy * dc[1, 1]
        Ez[inside] = hz * dc[2, 2]
    return Ex, Ey, Ez


@njit(f8[:](f8[:, :], i8[:, :], f8[:, :], i8[:], f8[:]), cache=True, nogil=True)
def constant_interp(
    coords: np.ndarray,
    tets: np.ndarray,
    nodes: np.ndarray,
    tetids: np.ndarray,
    value: np.ndarray,
):
    """Nedelec 2 tetrahedral interpolation of the analytic curl"""
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]

    prop = np.full((nNodes,), 0, dtype=np.float64)

    for i_iter in range(tetids.shape[0]):
        itet = tetids[i_iter]

        iv1, iv2, iv3, iv4 = tets[:, itet]

        v1 = nodes[:, iv1]
        v2 = nodes[:, iv2]
        v3 = nodes[:, iv3]
        v4 = nodes[:, iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.zeros((3, 3))
        blocal[:, 0] = bv1
        blocal[:, 1] = bv2
        blocal[:, 2] = bv3
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:, np.newaxis]
        coords_local = basis @ (coords_offset)

        inside = (
            (
                (coords_local[0, :] + coords_local[1, :] + coords_local[2, :])
                <= 1.00000001
            )
            & (coords_local[0, :] >= -1e-6)
            & (coords_local[1, :] >= -1e-6)
            & (coords_local[2, :] >= -1e-6)
        )

        if inside.sum() == 0:
            continue

        prop[inside] = value[itet]

    return prop
