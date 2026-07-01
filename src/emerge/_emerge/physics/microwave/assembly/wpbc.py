# # EMerge is an open source Python based FEM EM simulation module.
# # Copyright (C) 2025  Robert Fennis.

# # This program is free software; you can redistribute it and/or
# # modify it under the terms of the GNU General Public License
# # as published by the Free Software Foundation; either version 2
# # of the License, or (at your option) any later version.

# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.

# # You should have received a copy of the GNU General Public License
# # along with this program; if not, see
# # <https://www.gnu.org/licenses/>.


# # Last Cleanup: 2025-01-01
# import numpy as np

# from ....elements import Nedelec2
# from numba import njit, f8, types, c16, prange, i8
# from typing import Callable
# from loguru import logger
# from .robinbc import (
#     normalize,
#     cross,
#     DPTS,
#     tri_coefficients,
#     compute_distances,
#     optim_matmul,
#     generate_points_3d,
# )
# import functools
# from ....const import MU0, C0

# DEBUG_MODE = False
# SCALE_LENGTH = False

# @njit(types.Tuple((f8[:, :], f8[:, :]))(f8[:, :], f8[:]), cache=True, nogil=True)
# def construct_local_vertices(glob_vertices, normal):
#     origin = glob_vertices[:, 0]
#     vertex_2 = glob_vertices[:, 1]
#     vertex_3 = glob_vertices[:, 2]

#     edge_1 = vertex_2 - origin
#     edge_2 = vertex_3 - origin

#     zhat = normalize(cross(edge_1, edge_2))
#     xhat = normalize(edge_1)
#     yhat = normalize(cross(zhat, xhat))

#     basis = np.zeros((3, 3), dtype=np.float64)
#     basis[0, :] = xhat
#     basis[1, :] = yhat
#     basis[2, :] = zhat

#     # # not sure if this should be include?
#     # if np.sum(basis[2, :] * normal) < 0:
#     #     basis[1, :] = -basis[1, :]
#     #     basis[2, :] = -basis[2, :]

#     return basis, optim_matmul(basis, glob_vertices - origin[:, np.newaxis])


# @njit(c16[:](f8[:, :], c16[:, :], f8[:]), cache=True, nogil=True, parallel=False)
# def ned2_tri_force(glob_vertices, glob_Uinc, normal):
#     """Nedelec-2 Triangle forcing vector (For Boundary Condition of the Third Kind)"""
#     bvec = np.zeros((8,), dtype=np.complex128)

#     basis, local_vertices = construct_local_vertices(glob_vertices, normal)

#     txs = local_vertices[0, :]
#     tys = local_vertices[1, :]
#     Ds = compute_distances(txs, tys)
#     aas, bbs, ccs, A = tri_coefficients(txs, tys)
#     coeff = np.empty((3, 3), dtype=np.float64)
#     coeff[0, :] = aas
#     coeff[1, :] = bbs
#     coeff[2, :] = ccs

#     lcs_Uinc = optim_matmul(basis, glob_Uinc)

#     WEIGHTS = DPTS[0, :]
#     DPTS1 = DPTS[1, :]
#     DPTS2 = DPTS[2, :]
#     DPTS3 = DPTS[3, :]

#     xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
#     ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

#     coords = np.empty((2, xs.shape[0]), dtype=np.float64)
#     coords[0, :] = xs
#     coords[1, :] = ys

#     Ux = lcs_Uinc[0, :]
#     Uy = lcs_Uinc[1, :]
#     Uinc_2d = np.empty((2, xs.shape[0]), dtype=np.complex128)
#     Uinc_2d[0, :] = Ux
#     Uinc_2d[1, :] = Uy

#     ivec = np.array([0, 1, 0, 0, 0, 1, 0, 0])
#     jvec = np.array([1, 2, 2, 1, 1, 2, 2, 1])
#     kvec = np.array([0, 0, 0, 2, 0, 0, 0, 2])

#     Lvec = np.empty(8, dtype=np.float64)
#     for idof in range(8):
#         Lvec[idof] = (
#             Ds[ivec[idof], jvec[idof]]
#             if idof < 3 or (4 <= idof < 7)
#             else Ds[jvec[idof], kvec[idof]]
#         )

#     for idof in range(8):
#         i1 = ivec[idof]
#         j1 = jvec[idof]
#         k1 = kvec[idof]

#         if idof < 3:
#             fdof = _ne1_tri(coeff, coords, i1, j1, k1)
#         elif idof == 3:
#             fdof = _nf1_tri(coeff, coords, i1, j1, k1)
#         elif idof < 7:
#             fdof = _ne2_tri(coeff, coords, i1, j1, k1)
#         else:
#             fdof = _nf2_tri(coeff, coords, i1, j1, k1)

#         bvec[idof] = -A * np.sum(WEIGHTS * (fdof[0, :] * Ux + fdof[1, :] * Uy))
#     if SCALE_LENGTH == True:
#         bvec = bvec * Lvec
#     return bvec


# @njit(
#     c16[:](f8[:, :], i8[:, :], c16[:], i8[:], c16[:, :, :], i8[:, :], f8[:]),
#     cache=True,
#     nogil=True,
#     parallel=False,
# )
# def compute_force_entries(
#     vertices_global,
#     tris,
#     Bvec,
#     surf_triangle_indices,
#     Uglobal_all,
#     tri_to_field,
#     normal,
# ):
#     Niter = surf_triangle_indices.shape[0]
#     for i in prange(Niter):  # type: ignore
#         itri = surf_triangle_indices[i]

#         vertex_ids = tris[:, itri]

#         Ulocal = Uglobal_all[:, :, i]

#         bvec = ned2_tri_force(vertices_global[:, vertex_ids], Ulocal, normal)

#         indices = tri_to_field[:, itri]

#         Bvec[indices] += bvec
#     return Bvec


# def assemble_Bmatrix_entries(
#     G_global: np.ndarray,
#     constant: complex,
#     active_dof_ids: np.ndarray,
# ):

#     # 1. Extract only the vector entries that matter
#     G_active = G_global[active_dof_ids]

#     # 2. Vectorized outer product: Bdense shape is (nnz, nnz)
#     Bdense = constant * np.outer(G_active, G_active)

#     # 3. Use np.meshgrid to create the global row and column index maps
#     # This maps our local (nnz, nnz) block to the exact global sparse coordinates
#     cols_grid, rows_grid = np.meshgrid(active_dof_ids, active_dof_ids)

#     # Flatten everything so it's ready for a COO/CSC sparse matrix format
#     return Bdense.ravel(), rows_grid.ravel(), cols_grid.ravel()


# def assemble_wpbc(
#     field: Nedelec2,
#     surf_triangle_indices: np.ndarray,
#     mprof: Callable,
#     mode_xy: Callable,
#     kappa_m: complex,
#     gamma_m: complex,
#     k0: float,
#     port_normal: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:

#     vertices = field.mesh.nodes
#     bvec = np.zeros((field.n_field,), dtype=np.complex128)
#     bvec2 = np.zeros((field.n_field,), dtype=np.complex128)
#     tris = field.mesh.tris

#     vertices = field.mesh.nodes

#     xflat, yflat, zflat = generate_points_3d(
#         vertices, field.mesh.tris, DPTS, surf_triangle_indices
#     )

#     U_global = mprof(xflat, yflat, zflat)
#     U_global_all = U_global.reshape((3, DPTS.shape[1], surf_triangle_indices.shape[0]))

#     U_global_xy = mode_xy(xflat, yflat, zflat)
#     U_global_xy_all = U_global_xy.reshape(
#         (3, DPTS.shape[1], surf_triangle_indices.shape[0])
#     )

#     G = compute_force_entries(
#         vertices,
#         tris,
#         bvec,
#         surf_triangle_indices,
#         U_global_all,
#         field.tri_to_field,
#         port_normal,
#     )
#     G_xy = compute_force_entries(
#         vertices,
#         tris,
#         bvec2,
#         surf_triangle_indices,
#         U_global_xy_all,
#         field.tri_to_field,
#         port_normal,
#     )
#     ids = np.argwhere(G != 0).ravel()
#     w0 = C0 * k0
#     Bvec, rows, cols = assemble_Bmatrix_entries(G, 1.0 / (1j * w0 * MU0 * kappa_m), ids)
#     # Bvec, rows, cols = assemble_Bmatrix_entries(G, -gamma_m, ids)
#     return Bvec, rows, cols, -2 * gamma_m * G_xy
