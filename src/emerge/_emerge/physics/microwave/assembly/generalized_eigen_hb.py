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
from ....compiled.volakis import (
    _ne1_curl_tri,
    _ne2_curl_tri,
    _nf1_curl_tri,
    _nf2_curl_tri,
    _ne1_tri,
    _ne2_tri,
    _nf1_tri,
    _nf2_tri,
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
    basis: np.ndarray,
    k0: float,
) -> tuple[csr_matrix, csr_matrix]:

    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    nT = tris.shape[1]
    tri_to_field = field.tri_to_field

    nodes = field.local_nodes

    dataE, dataB, rows, cols = _matrix_builder(
        nodes, tris, edges, tri_to_field, ur, er, k0
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


DPTS = np.array(
    [
        [
            0.10995174365532,
            0.10995174365532,
            0.10995174365532,
            0.22338158967801,
            0.22338158967801,
            0.22338158967801,
        ],  # weights
        [
            0.81684757298046,
            0.09157621350977,
            0.09157621350977,
            0.10810301816807,
            0.44594849091597,
            0.44594849091597,
        ],  # L1
        [
            0.09157621350977,
            0.81684757298046,
            0.09157621350977,
            0.44594849091597,
            0.10810301816807,
            0.44594849091597,
        ],  # L2
        [
            0.09157621350977,
            0.09157621350977,
            0.81684757298046,
            0.44594849091597,
            0.44594849091597,
            0.10810301816807,
        ],  # L3
    ],
    dtype=np.float64,
)

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
    types.Tuple((c16[:, :], c16[:, :]))(f8[:, :], i8[:, :], c16[:, :], c16[:, :], f8),
    cache=True,
    nogil=True,
)
def generalized_matrix_GQ(tri_vertices, local_edge_map, Ms, Mm, k0):
    """Nedelec-2 Triangle stiffness and mass submatrix"""
    Att = np.zeros((8, 8), dtype=np.complex128)
    Btt = np.zeros((8, 8), dtype=np.complex128)

    Dtt = np.zeros((8, 8), dtype=np.complex128)
    Dzt = np.zeros((6, 8), dtype=np.complex128)

    Dzz1 = np.zeros((6, 6), dtype=np.complex128)
    Dzz2 = np.zeros((6, 6), dtype=np.complex128)

    ivec = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    jvec = np.array([1, 2, 2, 1, 1, 2, 2, 1])
    kvec = np.array([0, 0, 0, 2, 0, 0, 0, 2])

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

    Msz = Ms[2, 2]
    Mmz = Mm[2, 2]
    Ms = Ms[:2, :2]
    Mm = Mm[:2, :2]

    for iv1 in range(3):
        i1 = ivec[iv1]
        j1 = jvec[iv1]
        k1 = kvec[iv1]

        F1 = _ne1_curl_tri(coeff, cs, i1, j1, k1)
        F2 = _ne2_curl_tri(coeff, cs, i1, j1, k1)
        F3 = _ne1_tri(coeff, cs, i1, j1, k1)
        F4 = _ne2_tri(coeff, cs, i1, j1, k1)
        F5 = _nv_grad_tri(coeff, cs, iv1, j1, k1)
        F6 = _ne_grad_tri(coeff, cs, i1, j1, k1)

        for iv2 in range(3):
            i2 = ivec[iv2]
            j2 = jvec[iv2]
            k2 = kvec[iv2]

            H1 = matmul(Ms, _ne1_tri(coeff, cs, i2, j2, k2))
            H2 = matmul(Ms, _ne2_tri(coeff, cs, i2, j2, k2))

            Att[iv1, iv2] = _gqi(
                F1, Msz * _ne1_curl_tri(coeff, cs, i2, j2, k2), WEIGHTS
            )
            Att[iv1 + 4, iv2] = _gqi(
                F2, Msz * _ne1_curl_tri(coeff, cs, i2, j2, k2), WEIGHTS
            )
            Att[iv1, iv2 + 4] = _gqi(
                F1, Msz * _ne2_curl_tri(coeff, cs, i2, j2, k2), WEIGHTS
            )
            Att[iv1 + 4, iv2 + 4] = _gqi(
                F2, Msz * _ne2_curl_tri(coeff, cs, i2, j2, k2), WEIGHTS
            )

            Btt[iv1, iv2] = _gqi2(
                F3, matmul(Mm, _ne1_tri(coeff, cs, i2, j2, k2)), WEIGHTS
            )
            Btt[iv1 + 4, iv2] = _gqi2(
                F4, matmul(Mm, _ne1_tri(coeff, cs, i2, j2, k2)), WEIGHTS
            )
            Btt[iv1, iv2 + 4] = _gqi2(
                F3, matmul(Mm, _ne2_tri(coeff, cs, i2, j2, k2)), WEIGHTS
            )
            Btt[iv1 + 4, iv2 + 4] = _gqi2(
                F4, matmul(Mm, _ne2_tri(coeff, cs, i2, j2, k2)), WEIGHTS
            )

            Dtt[iv1, iv2] = _gqi2(F3, H1, WEIGHTS)
            Dtt[iv1 + 4, iv2] = _gqi2(F4, H1, WEIGHTS)
            Dtt[iv1, iv2 + 4] = _gqi2(F3, H2, WEIGHTS)
            Dtt[iv1 + 4, iv2 + 4] = _gqi2(F4, H2, WEIGHTS)

            Dzt[iv1, iv2] = _gqi2(F5, H1, WEIGHTS)
            Dzt[iv1 + 3, iv2] = _gqi2(F6, H1, WEIGHTS)
            Dzt[iv1, iv2 + 4] = _gqi2(F5, H2, WEIGHTS)
            Dzt[iv1 + 3, iv2 + 4] = _gqi2(F6, H2, WEIGHTS)

            Dzz1[iv1, iv2] = _gqi2(
                _nv_grad_tri(coeff, cs, iv1, j1, k1),
                matmul(Ms, _nv_grad_tri(coeff, cs, iv2, j2, k2)),
                WEIGHTS,
            )
            Dzz1[iv1, iv2 + 3] = _gqi2(
                _nv_grad_tri(coeff, cs, iv1, j1, k1),
                matmul(Ms, _ne_grad_tri(coeff, cs, i2, j2, k2)),
                WEIGHTS,
            )
            Dzz1[iv1 + 3, iv2] = _gqi2(
                _ne_grad_tri(coeff, cs, i1, j1, k1),
                matmul(Ms, _nv_grad_tri(coeff, cs, iv2, j2, k2)),
                WEIGHTS,
            )
            Dzz1[iv1 + 3, iv2 + 3] = _gqi2(
                _ne_grad_tri(coeff, cs, i1, j1, k1),
                matmul(Ms, _ne_grad_tri(coeff, cs, i2, j2, k2)),
                WEIGHTS,
            )

            Dzz2[iv1, iv2] = _gqi(
                _nv_tri(coeff, cs, iv1, j1, k1),
                Mmz * _nv_tri(coeff, cs, iv2, j2, k2),
                WEIGHTS,
            )
            Dzz2[iv1, iv2 + 3] = _gqi(
                _nv_tri(coeff, cs, iv1, j1, k1),
                Mmz * _ne_tri(coeff, cs, i2, j2, k2),
                WEIGHTS,
            )
            Dzz2[iv1 + 3, iv2] = _gqi(
                _ne_tri(coeff, cs, i1, j1, k1),
                Mmz * _nv_tri(coeff, cs, iv2, j2, k2),
                WEIGHTS,
            )
            Dzz2[iv1 + 3, iv2 + 3] = _gqi(
                _ne_tri(coeff, cs, i1, j1, k1),
                Mmz * _ne_tri(coeff, cs, i2, j2, k2),
                WEIGHTS,
            )

        G1 = matmul(Mm, _nf1_tri(coeff, cs, 0, 1, 2))
        G2 = matmul(Mm, _nf2_tri(coeff, cs, 0, 1, 2))
        G3 = matmul(Ms, _nf1_tri(coeff, cs, 0, 1, 2))
        G4 = matmul(Ms, _nf2_tri(coeff, cs, 0, 1, 2))

        Att[iv1, 3] = _gqi(F1, Msz * _nf1_curl_tri(coeff, cs, 0, 1, 2), WEIGHTS)
        Att[iv1 + 4, 3] = _gqi(
            _ne2_curl_tri(coeff, cs, i1, j1, k1),
            Msz * _nf1_curl_tri(coeff, cs, 0, 1, 2),
            WEIGHTS,
        )
        Att[iv1, 7] = _gqi(F1, Msz * _nf2_curl_tri(coeff, cs, 0, 1, 2), WEIGHTS)
        Att[iv1 + 4, 7] = _gqi(
            _ne2_curl_tri(coeff, cs, i1, j1, k1),
            Msz * _nf2_curl_tri(coeff, cs, 0, 1, 2),
            WEIGHTS,
        )

        Att[3, iv1] = Att[iv1, 3]
        Att[7, iv1] = Att[iv1, 7]
        Att[3, iv1 + 4] = Att[iv1 + 4, 3]
        Att[7, iv1 + 4] = Att[iv1 + 4, 7]

        Btt[iv1, 3] = _gqi2(F3, G1, WEIGHTS)
        Btt[iv1 + 4, 3] = _gqi2(F4, G1, WEIGHTS)
        Btt[iv1, 7] = _gqi2(F3, G2, WEIGHTS)
        Btt[iv1 + 4, 7] = _gqi2(F4, G2, WEIGHTS)

        Btt[3, iv1] = Btt[iv1, 3]
        Btt[7, iv1] = Btt[iv1, 7]
        Btt[3, iv1 + 4] = Btt[iv1 + 4, 3]
        Btt[7, iv1 + 4] = Btt[iv1 + 4, 7]

        Dtt[iv1, 3] = _gqi2(F3, G3, WEIGHTS)
        Dtt[iv1 + 4, 3] = _gqi2(F4, G3, WEIGHTS)
        Dtt[iv1, 7] = _gqi2(F3, G4, WEIGHTS)
        Dtt[iv1 + 4, 7] = _gqi2(F4, G4, WEIGHTS)

        Dtt[3, iv1] = Dtt[iv1, 3]
        Dtt[7, iv1] = Dtt[iv1, 7]
        Dtt[3, iv1 + 4] = Dtt[iv1 + 4, 3]
        Dtt[7, iv1 + 4] = Dtt[iv1 + 4, 7]

        Dzt[iv1, 3] = _gqi2(F5, G3, WEIGHTS)
        Dzt[iv1, 7] = _gqi2(F5, G4, WEIGHTS)
        Dzt[iv1 + 3, 3] = _gqi2(F6, G3, WEIGHTS)
        Dzt[iv1 + 3, 7] = _gqi2(F6, G4, WEIGHTS)

    Att[3, 3] = _gqi(
        _nf1_curl_tri(coeff, cs, 0, 1, 2),
        Msz * _nf1_curl_tri(coeff, cs, 0, 1, 2),
        WEIGHTS,
    )
    Att[7, 3] = _gqi(
        _nf2_curl_tri(coeff, cs, 0, 1, 2),
        Msz * _nf1_curl_tri(coeff, cs, 0, 1, 2),
        WEIGHTS,
    )
    Att[3, 7] = _gqi(
        _nf1_curl_tri(coeff, cs, 0, 1, 2),
        Msz * _nf2_curl_tri(coeff, cs, 0, 1, 2),
        WEIGHTS,
    )
    Att[7, 7] = _gqi(
        _nf2_curl_tri(coeff, cs, 0, 1, 2),
        Msz * _nf2_curl_tri(coeff, cs, 0, 1, 2),
        WEIGHTS,
    )

    Btt[3, 3] = _gqi2(_nf1_tri(coeff, cs, 0, 1, 2), G1, WEIGHTS)
    Btt[7, 3] = _gqi2(_nf2_tri(coeff, cs, 0, 1, 2), G1, WEIGHTS)
    Btt[3, 7] = _gqi2(_nf1_tri(coeff, cs, 0, 1, 2), G2, WEIGHTS)
    Btt[7, 7] = _gqi2(_nf2_tri(coeff, cs, 0, 1, 2), G2, WEIGHTS)

    A = np.zeros((14, 14), dtype=np.complex128)
    B = np.zeros((14, 14), dtype=np.complex128)

    A[:8, :8] = Att - k0**2 * Btt

    B[:8, :8] = Dtt
    B[8:, :8] = Dzt
    B[:8, 8:] = Dzt.T
    B[8:, 8:] = Dzz1 - k0**2 * Dzz2

    B = B * np.abs(Area)
    A = A * np.abs(Area)
    return A, B


@njit(
    types.Tuple((c16[:], c16[:], i8[:], i8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], c16[:, :, :], c16[:, :, :], f8
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _matrix_builder(nodes, tris, edges, tri_to_field, ur, er, k0):

    ntritot = tris.shape[1]
    nnz = ntritot * 196

    rows = np.zeros(nnz, dtype=np.int64)
    cols = np.zeros(nnz, dtype=np.int64)
    dataE = np.zeros_like(rows, dtype=np.complex128)
    dataB = np.zeros_like(rows, dtype=np.complex128)

    tri_to_edge = tri_to_field[:3, :]

    for itri in prange(ntritot):  # type: ignore
        p = itri * 196
        urt = ur[:, :, itri]
        ert = er[:, :, itri]

        # Construct a local mapping to global triangle orientations
        local_edge_map = local_tri_to_edgeid(itri, tris, edges, tri_to_edge)

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:, itri]]
        Esub, Bsub = generalized_matrix_GQ(
            tri_nodes, local_edge_map, matinv(urt), ert, k0
        )

        indices = tri_to_field[:, itri]
        for ii in range(14):
            rows[p + 14 * ii : p + 14 * (ii + 1)] = indices[ii]
            cols[p + ii : p + ii + 196 : 14] = indices[ii]

        dataE[p : p + 196] = Esub.ravel()
        dataB[p : p + 196] = Bsub.ravel()
    return dataE, dataB, rows, cols
