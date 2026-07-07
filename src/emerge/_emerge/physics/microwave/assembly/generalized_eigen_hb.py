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
from ....elements.nedleg2 import NedelecLegrange2
from scipy.sparse import csr_matrix
from ....mth.optimized import local_mapping, matinv
from numba import c16, types, f8, i8, njit, prange
from ....compiled.legrange import _ne_grad_tri, _ne_tri, _nv_tri, _nv_grad_tri
from ....compiled.ccbf import (
    _eval_f_2d, _eval_curl_f_2d, parse_dofcode, _eval_div_f_2d
)

############################################################
#                      FIELD MAPPING                      #
############################################################


@njit(i8[:, :](i8, i8[:, :], i8[:, :], i8[:, :]), cache=True, nogil=True)
def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:, itri]]
    return local_mapping(tris[:, itri], global_edge_map)


############################################################
#                     PYTHON INTERFACE                     #
############################################################


def generelized_eigenvalue_matrix(
    field: NedelecLegrange2,
    er: np.ndarray,
    ur: np.ndarray,
    inward_normal: np.ndarray,
    k0: float,
) -> tuple[csr_matrix, csr_matrix]:
    dofcodes = field.dofcodes
    tris = field.mesh.tris
    edges = field.mesh.edges

    nT = tris.shape[1]
    tri_to_field = field.tri_to_field

    nodes = field.local_nodes
    tri_to_edge = field.mesh.tri_to_edge
    dataE, dataB, rows, cols = _matrix_builder(
        nodes, tris, edges, tri_to_edge, tri_to_field, ur, er, k0, dofcodes,
    )

    nfield = field.n_field

    E = csr_matrix((dataE, (rows, cols)), shape=(nfield, nfield))
    B = csr_matrix((dataB, (rows, cols)), shape=(nfield, nfield))

    return E, B


############################################################
#                   MATRIX MULTIPLICATION                  #
############################################################


@njit(c16[:, :](c16[:, :], c16[:, :]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((2, b.shape[1]), dtype=np.complex128)
    out[0, :] = a[0, 0] * b[0, :] + a[0, 1] * b[1, :]
    out[1, :] = a[1, 0] * b[0, :] + a[1, 1] * b[1, :]
    return out


############################################################
#     TRIANGLE BARYCENTRIC COORDINATE LIN. COEFFICIENTS    #
############################################################


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


############################################################
#                    CONSTANT DEFINITION                   #
############################################################


DPTS = np.array([
    [0.22500000000000001, 0.12593918054482717, 0.12593918054482717, 0.12593918054482717, 0.13239415278850616, 0.13239415278850616, 0.13239415278850616],  # weights
    [0.33333333333333331, 0.79742698535308720, 0.10128650732345633, 0.10128650732345633, 0.05971587178976981, 0.47014206410511505, 0.47014206410511505],  # L1
    [0.33333333333333331, 0.10128650732345633, 0.79742698535308720, 0.10128650732345633, 0.47014206410511505, 0.05971587178976981, 0.47014206410511505],  # L2
    [0.33333333333333343, 0.10128650732345647, 0.10128650732345645, 0.79742698535308731, 0.47014206410511516, 0.47014206410511516, 0.05971587178976989],  # L3
], dtype=np.float64)

############################################################
#                 NUMBA OPTIMIZED ASSEMBLER                #
############################################################


@njit(
    c16(c16[:], c16[:], types.Array(types.float64, 1, "A", readonly=True)),
    cache=True,
    nogil=True,
)
def _gqi(v1, v2, W):
    return np.sum(v1 * v2 * W)


@njit(
    c16(c16[:, :], c16[:, :], types.Array(types.float64, 1, "A", readonly=True)),
    cache=True,
    nogil=True,
)
def _gqi2(v1, v2, W):
    return np.sum(W * np.sum(v1 * v2, axis=0))


@njit(
    types.Tuple((c16[:, :], c16[:, :]))(f8[:, :], i8[:, :], c16[:, :], c16[:, :], f8, i8[:]),
    cache=True,
    nogil=True,
)
def generalized_matrix_GQ(tri_vertices, local_edge_map, urinv, er, k0, dofcodes):
    """Nedelec-2 Triangle stiffness and mass submatrix"""

    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]

    n = ndof + 6
    Att_stiff = np.zeros((ndof, ndof), dtype=np.complex128)
    Att_mass = np.zeros((ndof, ndof), dtype=np.complex128)
    
    # Ptt = np.zeros((ndof, ndof), dtype=np.complex128)
    # Ptz = np.zeros((ndof, 6), dtype=np.complex128)
    # Pzz = np.zeros((6, 6), dtype=np.complex128)

    Btt = np.zeros((ndof, ndof), dtype=np.complex128)
    Btz = np.zeros((ndof, 6), dtype=np.complex128)

    Bzz_stiff = np.zeros((6, 6), dtype=np.complex128)
    Bzz_mass = np.zeros((6, 6), dtype=np.complex128)

    ivec = np.array([0, 1, 0])
    jvec = np.array([1, 2, 2])
    kvec = np.array([0, 0, 0])

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]

    txs = tri_vertices[0, :]
    tys = tri_vertices[1, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3

    cs = np.empty((2, xs.shape[0]), dtype=np.float64)
    cs[0, :] = xs
    cs[1, :] = ys

    aas, bbs, ccs, Area = tri_coefficients(txs, tys)

    coeff = np.empty((3, 3), dtype=np.float64)
    coeff[0, :] = aas
    coeff[1, :] = bbs
    coeff[2, :] = ccs

    urinv_z = urinv[2, 2]
    er_z = er[2, 2]
    urinv = urinv[:2, :2]
    er = er[:2, :2]

    all_fdof = np.empty((ndof, 2, cs.shape[1]), dtype=np.complex128)
    all_fdof_curl = np.empty((ndof, cs.shape[1]), dtype=np.complex128)
    #all_fdof_div = np.empty((ndof, cs.shape[1]), dtype=np.complex128)
    all_fdof_grad = np.empty((6, 2, cs.shape[1]), dtype=np.complex128)
    all_fdof_z = np.empty((6, cs.shape[1]), dtype=np.complex128)


    for idof in range(ndof):
        i_type = typearry[idof]
        i_index = indexarry[idof]
        if i_type==0: 
            # edge mode
            i1 = ivec[i_index]
            j1 = jvec[i_index]
            k1 = 0
        else:
            i1 = 0
            j1 = 1
            k1 = 2
        
        _eval_f_2d(coeff, cs, i1, j1, k1, dofcodes[idof], all_fdof[idof,:,:])
        _eval_curl_f_2d(coeff, cs, i1, j1, k1, dofcodes[idof], all_fdof_curl[idof,:])
        

    for idof in range(6):

        if idof < 3:
            all_fdof_grad[idof,:,:] = _nv_grad_tri(coeff, cs, idof, 0, 0)
            all_fdof_z[idof,:] = _nv_tri(coeff, cs, idof, 0, 0)
        else:
            i1 = ivec[idof-3]
            j1 = jvec[idof-3]
            k1 = 0
            all_fdof_grad[idof,:,:] = _ne_grad_tri(coeff, cs, i1, j1, k1)
            all_fdof_z[idof,:] = _ne_tri(coeff, cs, i1, j1, k1)
    
    # TT block
    for idof1 in range(ndof):
        f1 = all_fdof[idof1,:,:]
        fcurl1 = all_fdof_curl[idof1,:]
        #fdiv1 = all_fdof_div[idof1, :]
        for idof2 in range(ndof):
            f2 = all_fdof[idof2,:,:]
            fcurl2 = all_fdof_curl[idof2,:]
            #fdiv2 = all_fdof_div[idof2, :]

            Att_stiff[idof1, idof2] = _gqi(urinv_z*fcurl1, fcurl2, WEIGHTS)
            Att_mass[idof1, idof2] = _gqi2(matmul(er, f1), f2, WEIGHTS)
            #Ptt[idof1, idof2] = alpha*_gqi(fdiv1, fdiv2, WEIGHTS)

            Btt[idof1, idof2] = _gqi2(matmul(urinv, f1), f2, WEIGHTS)

        for idof2 in range(6):
            f2 = all_fdof_grad[idof2,:,:]
            Btz[idof1, idof2] = _gqi2(matmul(urinv, f1), f2, WEIGHTS)
            #Ptz[idof1, idof2] = alpha*_gqi(fdiv1, all_fdof_z[idof2,:], WEIGHTS)

    for idof1 in range(6):
        f1 = all_fdof_grad[idof1, :,:]
        f1z = all_fdof_z[idof1, :]
        for idof2 in range(6):
            f2 = all_fdof_grad[idof2,:,:]
            f2z = all_fdof_z[idof2,:]
            Bzz_stiff[idof1, idof2] = _gqi2(matmul(urinv, f1), f2, WEIGHTS)
            Bzz_mass[idof1, idof2] = _gqi(er_z*f1z, f2z, WEIGHTS)
            #Pzz[idof1, idof2] = alpha*_gqi(f1z, f2z, WEIGHTS)
    A = np.zeros((n,n), dtype=np.complex128)
    B = np.zeros((n,n), dtype=np.complex128)

    A[:ndof, :ndof] = Att_stiff - k0**2 * Att_mass

    B[:ndof, :ndof] = Btt
    B[ndof:, :ndof] = Btz.T
    B[:ndof, ndof:] = Btz
    B[ndof:, ndof:] = Bzz_stiff - k0**2 * Bzz_mass

    B = B * np.abs(Area)
    A = A * np.abs(Area)
    return A, B


@njit(
    types.Tuple((c16[:], c16[:], i8[:], i8[:]))(
        f8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        i8[:, :],
        c16[:, :, :],
        c16[:, :, :],
        f8,
        i8[:],
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _matrix_builder(nodes, tris, edges, tri_to_edge, tri_to_field, ur, er, k0, dofcodes):
    typearry, indexarry = parse_dofcode(dofcodes)
    ndof = dofcodes.shape[0]

    ntritot = tris.shape[1]
    n = (ndof + 6)
    nsq = n**2
    nnz = ntritot * nsq

    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    dataE = np.zeros_like(rows, dtype=np.complex128)
    dataB = np.zeros_like(rows, dtype=np.complex128)

    for itri in prange(ntritot):  # type: ignore
        p = itri * nsq
        urt = ur[:, :, itri]
        ert = er[:, :, itri]

        # Construct a local mapping to global triangle orientations
        local_edge_map = local_tri_to_edgeid(itri, tris, edges, tri_to_edge)

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:, itri]]
        Esub, Bsub = generalized_matrix_GQ(
            tri_nodes, local_edge_map, matinv(urt), ert, k0, dofcodes
        )

        indices = tri_to_field[:, itri]
        for ii in range(n):
            rows[p + n * ii : p + n * (ii + 1)] = indices[ii]
            cols[p + ii : p + ii + nsq : n] = indices[ii]

        dataE[p : p + nsq] = Esub.ravel()
        dataB[p : p + nsq] = Bsub.ravel()
    return dataE, dataB, rows, cols
