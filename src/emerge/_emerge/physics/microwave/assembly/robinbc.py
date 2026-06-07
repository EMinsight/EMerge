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
import numpy as np
from numba import njit, f8, c16, i8, types, prange
from ....mth.optimized import cross
from ....elements import Nedelec2
from ....compiled.volakis import (
    SCALE_LENGTH,
    _ne1_tri,
    _ne2_tri,
    _nf1_tri,
    _nf2_tri,
)
from typing import Callable
from loguru import logger
import functools

#
# Toggle this to True when you want to use standard Python breakpoints
DEBUG_MODE = False


def njit(*args, **kwargs):
    """
    Drop-in replacement for numba.njit.
    If DEBUG_MODE is True, it turns into a transparent 'do-nothing' wrapper.
    If DEBUG_MODE is False, it forwards everything to the real Numba compiler.
    """
    if DEBUG_MODE:
        # Case A: Used without parentheses -> @njit
        if len(args) == 1 and callable(args[0]):
            return args[0]

        # Case B: Used with signatures/kwargs -> @njit(cache=True)
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                return func(*func_args, **func_kwargs)

            return wrapper

        return decorator
    else:
        # Import Numba lazily only when debugging is turned off
        import numba

        return numba.njit(*args, **kwargs)


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0, :] = B[0, 0] * data[0, :] + B[0, 1] * data[1, :] + B[0, 2] * data[2, :]
    dnew[1, :] = B[1, 0] * data[0, :] + B[1, 1] * data[1, :] + B[1, 2] * data[2, :]
    dnew[2, :] = B[2, 0] * data[0, :] + B[2, 1] * data[1, :] + B[2, 2] * data[2, :]
    return dnew


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul_vec(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros((3,), dtype=data.dtype)
    dnew[0] = B[0, 0] * data[0] + B[0, 1] * data[1] + B[0, 2] * data[2]
    dnew[1] = B[1, 0] * data[0] + B[1, 1] * data[1] + B[1, 2] * data[2]
    dnew[2] = B[2, 0] * data[0] + B[2, 1] * data[1] + B[2, 2] * data[2]
    return dnew


@njit(c16[:](c16[:, :], c16[:, :]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray):
    return a[0, :] * b[0, :] + a[1, :] * b[1, :]


@njit(
    types.Tuple((f8[:], f8[:]))(f8[:, :], i8[:, :], f8[:, :], i8[:]),
    cache=True,
    nogil=True,
)
def generate_points(vertices_local, tris, DPTs, surf_triangle_indices):
    NS = surf_triangle_indices.shape[0]
    xall = np.zeros((DPTs.shape[1], NS))
    yall = np.zeros((DPTs.shape[1], NS))

    for i in range(NS):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]

        x1, x2, x3 = vertices_local[0, vertex_ids]
        y1, y2, y3 = vertices_local[1, vertex_ids]

        xall[:, i] = x1 * DPTs[1, :] + x2 * DPTs[2, :] + x3 * DPTs[3, :]
        yall[:, i] = y1 * DPTs[1, :] + y2 * DPTs[2, :] + y3 * DPTs[3, :]

    xflat = xall.flatten()
    yflat = yall.flatten()
    return xflat, yflat


@njit(
    types.Tuple((f8[:], f8[:], f8[:]))(f8[:, :], i8[:, :], f8[:, :], i8[:]),
    cache=True,
    nogil=True,
)
def generate_points_3d(vertices, tris, DPTs, surf_triangle_indices):
    NS = surf_triangle_indices.shape[0]
    xall = np.zeros((DPTs.shape[1], NS))
    yall = np.zeros((DPTs.shape[1], NS))
    zall = np.zeros((DPTs.shape[1], NS))
    for i in range(NS):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]

        x1, x2, x3 = vertices[0, vertex_ids]
        y1, y2, y3 = vertices[1, vertex_ids]
        z1, z2, z3 = vertices[2, vertex_ids]

        xall[:, i] = x1 * DPTs[1, :] + x2 * DPTs[2, :] + x3 * DPTs[3, :]
        yall[:, i] = y1 * DPTs[1, :] + y2 * DPTs[2, :] + y3 * DPTs[3, :]
        zall[:, i] = z1 * DPTs[1, :] + z2 * DPTs[2, :] + z3 * DPTs[3, :]
    xflat = xall.flatten()
    yflat = yall.flatten()
    zflat = zall.flatten()
    return xflat, yflat, zflat


@njit(f8[:, :](f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            Ds[i, j] = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            Ds[j, i] = Ds[i, j]
    return Ds


@njit(cache=True, nogil=True)
def normalize(a: np.ndarray):
    return a / ((a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5)


@njit(c16[:, :](c16[:, :], c16[:, :]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 2x2 Matrix with a Vector
    Vout = np.empty((2, Vec.shape[1]), dtype=np.complex128)
    Vout[0, :] = Mat[0, 0] * Vec[0, :] + Mat[0, 1] * Vec[1, :]
    Vout[1, :] = Mat[1, 0] * Vec[0, :] + Mat[1, 1] * Vec[1, :]
    return Vout


@njit(types.Tuple((f8[:, :], f8[:, :]))(f8[:, :]), cache=True, nogil=True)
def construct_local_vertices(glob_vertices):
    origin = glob_vertices[:, 0]
    vertex_2 = glob_vertices[:, 1]
    vertex_3 = glob_vertices[:, 2]

    edge_1 = vertex_2 - origin
    edge_2 = vertex_3 - origin

    zhat = normalize(cross(edge_1, edge_2))
    xhat = normalize(edge_1)
    yhat = normalize(cross(zhat, xhat))

    basis = np.zeros((3, 3), dtype=np.float64)
    basis[0, :] = xhat
    basis[1, :] = yhat
    basis[2, :] = zhat

    return basis, optim_matmul(basis, glob_vertices - origin[:, np.newaxis])


############################################################
#                         ASSEMBLY                        #
############################################################
# fmt: off
DPTS = np.array([
    [0.10995174365532, 0.10995174365532, 0.10995174365532, 0.22338158967801, 0.22338158967801, 0.22338158967801],  # weights
    [0.81684757298046, 0.09157621350977, 0.09157621350977, 0.10810301816807, 0.44594849091597, 0.44594849091597],  # L1
    [0.09157621350977, 0.81684757298046, 0.09157621350977, 0.44594849091597, 0.10810301816807, 0.44594849091597],  # L2
    [0.09157621350977, 0.09157621350977, 0.81684757298046, 0.44594849091597, 0.44594849091597, 0.10810301816807],  # L3
], dtype=np.float64)
# fmt: on


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

    sA = 0.5 * (b1 * c2 - b2 * c1)
    As = np.array([a1, a2, a3]) / (2 * sA)
    Bs = np.array([b1, b2, b3]) / (2 * sA)
    Cs = np.array([c1, c2, c3]) / (2 * sA)
    return As, Bs, Cs, np.abs(sA)


@njit(c16[:](f8[:, :], c16[:, :]), cache=True, nogil=True, parallel=False)
def ned2_tri_force(glob_vertices, glob_Uinc):
    """Nedelec-2 Triangle forcing vector (For Boundary Condition of the Third Kind)"""
    bvec = np.zeros((8,), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]
    Ds = compute_distances(txs, tys)
    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas
    coeff[1, :] = bbs
    coeff[2, :] = ccs

    lcs_Uinc = optim_matmul(basis, glob_Uinc)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    Ux = lcs_Uinc[0, :]
    Uy = lcs_Uinc[1, :]
    Uinc_2d = np.empty((2, xs.shape[0]), dtype=np.complex128)
    Uinc_2d[0, :] = Ux
    Uinc_2d[1, :] = Uy

    ivec = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    jvec = np.array([1, 2, 2, 1, 1, 2, 2, 1])
    kvec = np.array([0, 0, 0, 2, 0, 0, 0, 2])

    Lvec = np.empty(8, dtype=np.float64)
    for idof in range(8):
        Lvec[idof] = (
            Ds[ivec[idof], jvec[idof]]
            if idof < 3 or (4 <= idof < 7)
            else Ds[jvec[idof], kvec[idof]]
        )

    for idof in range(8):
        i1 = ivec[idof]
        j1 = jvec[idof]
        k1 = kvec[idof]

        if idof < 3:
            fdof = _ne1_tri(coeff, coords, i1, j1, k1)
        elif idof == 3:
            fdof = _nf1_tri(coeff, coords, i1, j1, k1)
        elif idof < 7:
            fdof = _ne2_tri(coeff, coords, i1, j1, k1)
        else:
            fdof = _nf2_tri(coeff, coords, i1, j1, k1)

        bvec[idof] = -A * np.sum(WEIGHTS * (fdof[0, :] * Ux + fdof[1, :] * Uy))
    if SCALE_LENGTH == True:
        bvec = bvec * Lvec
    return bvec


@njit(
    c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16[:, :, :], i8[:, :]),
    cache=True,
    nogil=True,
    parallel=False,
)
def compute_force_entries(
    vertices_global, tris, Bvec, surf_triangle_indices, Uglobal_all, tri_to_field
):
    Niter = surf_triangle_indices.shape[0]
    for i in prange(Niter):  # type: ignore
        itri = surf_triangle_indices[i]

        vertex_ids = tris[:, itri]

        Ulocal = Uglobal_all[:, :, i]

        bvec = ned2_tri_force(vertices_global[:, vertex_ids], Ulocal)

        indices = tri_to_field[:, itri]

        Bvec[indices] += bvec
    return Bvec


@njit(c16[:, :](f8[:, :], c16), cache=True, nogil=True, parallel=False)
def ned2_tri_stiff(glob_vertices, gamma):
    """Nedelec-2 Triangle Stiffness matrix and forcing vector (For Boundary Condition of the Third Kind)"""
    Bmat = np.zeros((8, 8), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]

    Ds = compute_distances(txs, tys)
    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    A = np.abs(A)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas  # / (2 * A)
    coeff[1, :] = bbs  # / (2 * A)
    coeff[2, :] = ccs  # / (2 * A)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    ivec = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    jvec = np.array([1, 2, 2, 1, 1, 2, 2, 1])
    kvec = np.array([0, 0, 0, 2, 0, 0, 0, 2])

    Ls = np.ones((8, 8), dtype=np.float64)
    for idof in range(8):
        Le = (
            Ds[ivec[idof], jvec[idof]]
            if idof < 3 or (4 <= idof < 7)
            else Ds[jvec[idof], kvec[idof]]
        )
        Ls[idof, :] *= Le
        Ls[:, idof] *= Le

    for idof1 in range(8):
        i1 = ivec[idof1]
        j1 = jvec[idof1]
        k1 = kvec[idof1]

        if idof1 < 3:
            fdof1 = _ne1_tri(coeff, coords, i1, j1, k1)
        elif idof1 == 3:
            fdof1 = _nf1_tri(coeff, coords, i1, j1, k1)
        elif idof1 < 7:
            fdof1 = _ne2_tri(coeff, coords, i1, j1, k1)
        else:
            fdof1 = _nf2_tri(coeff, coords, i1, j1, k1)

        for idof2 in range(8):
            i2 = ivec[idof2]
            j2 = jvec[idof2]
            k2 = kvec[idof2]

            if idof2 < 3:
                fdof2 = _ne1_tri(coeff, coords, i2, j2, k2)
            elif idof2 == 3:
                fdof2 = _nf1_tri(coeff, coords, i2, j2, k2)
            elif idof2 < 7:
                fdof2 = _ne2_tri(coeff, coords, i2, j2, k2)
            else:
                fdof2 = _nf2_tri(coeff, coords, i2, j2, k2)

            Bmat[idof1, idof2] = gamma * np.sum(dot(fdof1, fdof2) * WEIGHTS)

    if SCALE_LENGTH == True:
        Bmat = Bmat * Ls
    return Bmat * A


@njit(
    c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16),
    cache=True,
    nogil=True,
    parallel=False,
)
def compute_bc_entries(vertices, tris, Bmat, surf_triangle_indices, gamma):
    N = 64
    Niter = surf_triangle_indices.shape[0]
    for i in prange(Niter):  # type: ignore
        itri = surf_triangle_indices[i]

        vertex_ids = tris[:, itri]
        Bsub = ned2_tri_stiff(vertices[:, vertex_ids], gamma)

        Bmat[itri * N : (itri + 1) * N] = Bmat[itri * N : (itri + 1) * N] + Bsub.ravel()
    return Bmat


def assemble_robin_bc_bvec(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
):
    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices = field.mesh.nodes

    xflat, yflat, zflat = generate_points_3d(
        vertices, field.mesh.tris, DPTS, surf_triangle_indices
    )

    U_global = Ufunc(xflat, yflat, zflat)

    U_global_all = U_global.reshape((3, DPTS.shape[1], surf_triangle_indices.shape[0]))

    Bvec = compute_force_entries(
        vertices,
        field.mesh.tris,
        Bvec,
        surf_triangle_indices,
        U_global_all,
        field.tri_to_field,
    )
    return Bvec


def assemble_robin_bc(
    field: Nedelec2,
    Bmat: np.ndarray,
    surf_triangle_indices: np.ndarray,
    gamma: np.ndarray,
):

    vertices = field.mesh.nodes
    Bmat = compute_bc_entries(
        vertices, field.mesh.tris, Bmat, surf_triangle_indices, gamma
    )

    return Bmat


############################################################
#                      SCATTERED FIELD                     #
############################################################


@njit(
    c16[:](f8[:, :], c16[:, :], c16[:, :], f8[:]),
    cache=True,
    nogil=True,
    parallel=False,
)
def ned2_tri_force_scat(glob_vertices, glob_Uinc, glob_Uinc_curl, nhat):
    """Nedelec-2 Triangle forcing vector (scattered field, Robin BC)"""
    bvec = np.zeros((8,), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]

    Ds = compute_distances(txs, tys)

    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas
    coeff[1, :] = bbs
    coeff[2, :] = ccs

    lcs_Uinc = optim_matmul(basis, glob_Uinc)
    lcs_Uinc_curl = optim_matmul(basis, glob_Uinc_curl)
    lcs_nhat = optim_matmul_vec(basis, nhat)
    sgn = np.sign(lcs_nhat[2])

    WEIGHTS = DPTS[0, :]
    xs = txs[0] * DPTS[1, :] + txs[1] * DPTS[2, :] + txs[2] * DPTS[3, :]
    ys = tys[0] * DPTS[1, :] + tys[1] * DPTS[2, :] + tys[2] * DPTS[3, :]

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    Ux = lcs_Uinc[0, :] + lcs_Uinc_curl[1, :] * sgn
    Uy = lcs_Uinc[1, :] - lcs_Uinc_curl[0, :] * sgn

    ivec = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    jvec = np.array([1, 2, 2, 1, 1, 2, 2, 1])
    kvec = np.array([0, 0, 0, 2, 0, 0, 0, 2])

    Lvec = np.empty(8, dtype=np.float64)
    for idof in range(8):
        Lvec[idof] = (
            Ds[ivec[idof], jvec[idof]]
            if idof < 3 or (4 <= idof < 7)
            else Ds[jvec[idof], kvec[idof]]
        )

    for idof in range(8):
        i1 = ivec[idof]
        j1 = jvec[idof]
        k1 = kvec[idof]

        if idof < 3:
            fdof = _ne1_tri(coeff, coords, i1, j1, k1)
        elif idof == 3:
            fdof = _nf1_tri(coeff, coords, i1, j1, k1)
        elif idof < 7:
            fdof = _ne2_tri(coeff, coords, i1, j1, k1)
        else:
            fdof = _nf2_tri(coeff, coords, i1, j1, k1)

        bvec[idof] = -A * np.sum(WEIGHTS * (fdof[0, :] * Ux + fdof[1, :] * Uy))
    if SCALE_LENGTH == True:
        bvec = bvec * Lvec
    return bvec


@njit(
    c16[:](
        f8[:, :],
        i8[:, :],
        c16[:],
        i8[:],
        c16[:, :, :],
        c16[:, :, :],
        i8[:, :],
        f8[:, :],
    ),
    cache=True,
    nogil=True,
    parallel=False,
)
def compute_force_entries_scat(
    vertices_global,
    tris,
    Bvec,
    surf_triangle_indices,
    Uglobal_all,
    Uglobal_all_curl,
    tri_to_field,
    normals,
):
    Niter = surf_triangle_indices.shape[0]
    for i in prange(Niter):  # type: ignore
        itri = surf_triangle_indices[i]

        vertex_ids = tris[:, itri]

        Uglobal = Uglobal_all[:, :, i]
        UglobalCurl = Uglobal_all_curl[:, :, i]

        bvec = ned2_tri_force_scat(
            vertices_global[:, vertex_ids], Uglobal, UglobalCurl, normals[:, i]
        )

        indices = tri_to_field[:, itri]

        Bvec[indices] += bvec
    return Bvec


def assemble_robin_bc_bvec_scat(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Ufunc: Callable,
    UfuncCurl: Callable,
    normals: np.ndarray,
):

    Bvec = np.zeros((field.n_field,), dtype=np.complex128)

    vertices = field.mesh.nodes

    xflat, yflat, zflat = generate_points_3d(
        vertices, field.mesh.tris, DPTS, surf_triangle_indices
    )

    U_global = Ufunc(xflat, yflat, zflat)
    U_global_curl = UfuncCurl(xflat, yflat, zflat)

    U_global_all = U_global.reshape((3, DPTS.shape[1], surf_triangle_indices.shape[0]))
    U_global_all_curl = U_global_curl.reshape(
        (3, DPTS.shape[1], surf_triangle_indices.shape[0])
    )

    Bvec = compute_force_entries_scat(
        vertices,
        field.mesh.tris,
        Bvec,
        surf_triangle_indices,
        U_global_all,
        U_global_all_curl,
        field.tri_to_field,
        normals,
    )
    return Bvec
