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

DEBUG_MODE = False


def njit(*args, **kwargs):
    if DEBUG_MODE:
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                return func(*func_args, **func_kwargs)

            return wrapper

        return decorator
    else:
        import numba

        return numba.njit(*args, **kwargs)


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0, :] = B[0, 0] * data[0, :] + B[0, 1] * data[1, :] + B[0, 2] * data[2, :]
    dnew[1, :] = B[1, 0] * data[0, :] + B[1, 1] * data[1, :] + B[1, 2] * data[2, :]
    dnew[2, :] = B[2, 0] * data[0, :] + B[2, 1] * data[1, :] + B[2, 2] * data[2, :]
    return dnew


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
    return xall.flatten(), yall.flatten(), zall.flatten()


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
#                     QUADRATURE POINTS                    #
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

    sA = 0.5 * ((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
    As = np.array([a1, a2, a3])
    Bs = np.array([b1, b2, b3])
    Cs = np.array([c1, c2, c3])
    return As, Bs, Cs, sA


############################################################
#               PER-TRIANGLE INTEGRALS                     #
############################################################


@njit(c16[:](f8[:, :], c16[:, :]), cache=True, nogil=True, parallel=False)
def ned2_tri_project(glob_vertices, glob_F):
    """v_i = ∫_S F · N_i dS for one triangle (in-plane components of F used)."""
    bvec = np.zeros((8,), dtype=np.complex128)

    basis, local_vertices = construct_local_vertices(glob_vertices)
    txs = local_vertices[0, :]
    tys = local_vertices[1, :]
    Ds = compute_distances(txs, tys)
    aas, bbs, ccs, A = tri_coefficients(txs, tys)
    sign = np.sign(A)
    A = np.abs(A)
    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas / (2 * A)
    coeff[1, :] = bbs / (2 * A)
    coeff[2, :] = ccs / (2 * A)

    lcs_F = optim_matmul(basis, glob_F)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    coords = np.empty((2, xs.shape[0]), dtype=np.float64)
    coords[0, :] = xs
    coords[1, :] = ys

    Fx = lcs_F[0, :]
    Fy = lcs_F[1, :]

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

        bvec[idof] = sign * A * np.sum(WEIGHTS * (fdof[0, :] * Fx + fdof[1, :] * Fy))

    if SCALE_LENGTH == True:
        bvec = bvec * Lvec
    return bvec


@njit(c16(f8[:, :], c16[:, :], c16[:, :]), cache=True, nogil=True)
def ned2_tri_scalar_int(glob_vertices, glob_A, glob_B):
    """∫_S (A_t · B_t) dS for one triangle, in-plane components only."""
    basis, _ = construct_local_vertices(glob_vertices)

    e1 = glob_vertices[:, 1] - glob_vertices[:, 0]
    e2 = glob_vertices[:, 2] - glob_vertices[:, 0]
    nrm = np.empty(3, dtype=np.float64)
    nrm[0] = e1[1] * e2[2] - e1[2] * e2[1]
    nrm[1] = e1[2] * e2[0] - e1[0] * e2[2]
    nrm[2] = e1[0] * e2[1] - e1[1] * e2[0]
    A = 0.5 * np.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2)

    lcs_A = optim_matmul(basis, glob_A)
    lcs_B = optim_matmul(basis, glob_B)

    WEIGHTS = DPTS[0, :]
    integrand = lcs_A[0, :] * lcs_B[0, :] + lcs_A[1, :] * lcs_B[1, :]
    return A * np.sum(WEIGHTS * integrand)


############################################################
#                  MATRIX-LEVEL DRIVERS                    #
############################################################


@njit(
    c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16[:, :, :], i8[:, :]),
    cache=True,
    nogil=True,
    parallel=False,
)
def compute_projection(
    vertices_global, tris, vvec, surf_triangle_indices, Fglobal_all, tri_to_field
):
    """v_i = Σ_tris ∫ F · N_i dS"""
    Niter = surf_triangle_indices.shape[0]
    for i in prange(Niter):  # type: ignore
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]
        Flocal = Fglobal_all[:, :, i]
        bvec = ned2_tri_project(vertices_global[:, vertex_ids], Flocal)
        indices = tri_to_field[:, itri]
        vvec[indices] += bvec
    return vvec


@njit(
    c16(f8[:, :], i8[:, :], i8[:], c16[:, :, :], c16[:, :, :]),
    cache=True,
    nogil=True,
)
def compute_kappa(
    vertices_global, tris, surf_triangle_indices, Aglobal_all, Bglobal_all
):
    """Scalar surface integral Σ_tris ∫ A_t · B_t dS"""
    out = 0.0 + 0.0j
    Niter = surf_triangle_indices.shape[0]
    for i in range(Niter):
        itri = surf_triangle_indices[i]
        vertex_ids = tris[:, itri]
        out += ned2_tri_scalar_int(
            vertices_global[:, vertex_ids],
            Aglobal_all[:, :, i],
            Bglobal_all[:, :, i],
        )
    return out


############################################################
#                 INTERNAL EH → F, κ HELPER                #
############################################################


def _prepare_mode_data(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Efunc: Callable,
    Hfunc: Callable,
    nhat: np.ndarray,
    omega: float,
    mu: complex,
):
    """Evaluate E, H at port quadrature points and derive F = -jωμ·(n̂×H) and κ."""
    vertices = field.mesh.nodes

    xflat, yflat, zflat = generate_points_3d(
        vertices, field.mesh.tris, DPTS, surf_triangle_indices
    )
    E_global = Efunc(xflat, yflat, zflat)
    H_global = Hfunc(xflat, yflat, zflat)

    nxH = np.empty_like(H_global)
    nxH[0, :] = nhat[1] * H_global[2, :] - nhat[2] * H_global[1, :]
    nxH[1, :] = nhat[2] * H_global[0, :] - nhat[0] * H_global[2, :]
    nxH[2, :] = nhat[0] * H_global[1, :] - nhat[1] * H_global[0, :]

    F_global = nxH  # (-1j * omega * mu) * nxH

    npts = DPTS.shape[1]
    ntri = surf_triangle_indices.shape[0]
    E_all = E_global.reshape((3, npts, ntri))
    nxH_all = nxH.reshape((3, npts, ntri))
    F_all = F_global.reshape((3, npts, ntri))

    kappa = compute_kappa(
        vertices,
        field.mesh.tris,
        surf_triangle_indices,
        E_all,
        nxH_all,
    )
    return F_all, kappa


############################################################
#                    PYTHON INTERFACE                      #
############################################################


def assemble_wpbc(
    field: Nedelec2,
    K,
    surf_triangle_indices: np.ndarray,
    Efunc: Callable,
    Hfunc: Callable,
    nhat: np.ndarray,
    omega: float,
    mu: complex,
):
    """
    Add the WPBC rank-1 matrix contribution to K:

        K[i,j] += (1/κ) · v_i · v_j

    where:
        v_i = ∫_S F · N_i dS
        F   = -jωμ · (n̂ × H)
        κ   = ∫_S E_t · (n̂ × H_t) dS

    Args:
        field:                 Nedelec2 field object
        K:                     sparse system matrix supporting fancy indexing assignment
        surf_triangle_indices: triangles on the port surface
        Efunc:                 callable Efunc(x, y, z) -> (3, N) global E field of the mode
        Hfunc:                 callable Hfunc(x, y, z) -> (3, N) global H field of the mode
        nhat:                  (3,) outward unit normal of the port (constant for flat port)
        omega:                 angular frequency
        mu:                    permeability at the port

    Returns:
        K with the rank-1 WPBC update added.
    """
    F_all, kappa = _prepare_mode_data(
        field, surf_triangle_indices, Efunc, Hfunc, nhat, omega, mu
    )

    v = np.zeros((field.n_field,), dtype=np.complex128)
    v = compute_projection(
        field.mesh.nodes,
        field.mesh.tris,
        v,
        surf_triangle_indices,
        F_all,
        field.tri_to_field,
    )

    port_dofs = np.nonzero(np.abs(v) > 0)[0]
    v_port = v[port_dofs]
    K[np.ix_(port_dofs, port_dofs)] += 1.0 / (kappa) * np.outer(v_port, v_port)
    return K


def assemble_wpbc_bvec(
    field: Nedelec2,
    surf_triangle_indices: np.ndarray,
    Efunc: Callable,
    Hfunc: Callable,
    nhat: np.ndarray,
    omega: float,
    mu: complex,
    amplitude: complex,
) -> np.ndarray:
    """
    Build the WPBC forcing vector for an excited port:

        b_i = 2·a · v_i,    v_i = ∫_S F · N_i dS,    F = -jωμ · (n̂ × H)

    Args:
        field:                 Nedelec2 field object
        surf_triangle_indices: triangles on the port surface
        Efunc:                 callable Efunc(x, y, z) -> (3, N) global E field of the mode
        Hfunc:                 callable Hfunc(x, y, z) -> (3, N) global H field of the mode
        nhat:                  (3,) outward unit normal of the port
        omega:                 angular frequency
        mu:                    permeability at the port
        amplitude:             excitation amplitude

    Returns:
        bvec: (field.n_field,) complex forcing vector.
    """
    F_all, _ = _prepare_mode_data(
        field, surf_triangle_indices, Efunc, Hfunc, nhat, omega, mu
    )

    bvec = np.zeros((field.n_field,), dtype=np.complex128)
    bvec = compute_projection(
        field.mesh.nodes,
        field.mesh.tris,
        bvec,
        surf_triangle_indices,
        F_all,
        field.tri_to_field,
    )
    return 2.0 * amplitude * bvec
