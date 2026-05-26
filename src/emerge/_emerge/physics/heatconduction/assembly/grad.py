import numpy as np
from ....elements.leg2 import Legrange2
from scipy.sparse import csr_matrix
from ....mth.optimized import local_mapping, matinv, compute_distances, gaus_quad_tet
from numba import c16, types, f8, i8, njit, prange
from ....mth.csc_cast import CSCMapping

############################################################
#                    UTILITY FUNCTIONS                    #
############################################################


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((3, b.shape[1]), dtype=np.float64)
    out[0, :] = a[0, 0] * b[0, :] + a[0, 1] * b[1, :] + a[0, 2] * b[2, :]
    out[1, :] = a[1, 0] * b[0, :] + a[1, 1] * b[1, :] + a[1, 2] * b[2, :]
    out[2, :] = a[2, 0] * b[0, :] + a[2, 1] * b[1, :] + a[2, 2] * b[2, :]
    return out


############################################################
#            BARYCENTRIC COORDINATE COEFFICIENTS           #
############################################################


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]),
    cache=True,
    nogil=True,
)
def tet_coefficients(xs, ys, zs):
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


############################################################
#                      BASIS FUNCTIONS                     #
############################################################


# --- Gradients ∇N
@njit(f8[:, :](f8[:], f8[:, :]), cache=True, nogil=True)
def grad_n1(coeff, coords):
    a, b, c, d = coeff
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]

    out = np.empty((3, xs.shape[0]), dtype=np.float64)
    q = 4 * (b * xs + c * ys + d * zs) + (4 * a - 1)
    out[0, :] = q * b
    out[1, :] = q * c
    out[2, :] = q * d
    return out


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True, nogil=True)
def grad_n2(coeff, coord):
    a1, b1, c1, d1 = coeff[:, 0]
    a2, b2, c2, d2 = coeff[:, 1]

    xs = coord[0, :]
    ys = coord[1, :]
    zs = coord[2, :]

    out = np.empty((3, xs.shape[0]), dtype=np.float64)
    L1 = 4 * (a1 + b1 * xs + c1 * ys + d1 * zs)
    L2 = 4 * (a2 + b2 * xs + c2 * ys + d2 * zs)

    out[0, :] = L2 * b1 + L1 * b2
    out[1, :] = L2 * c1 + L1 * c2
    out[2, :] = L2 * d1 + L1 * d2

    return out


############################################################
#                  GAUSS QUADRATURE POINTS                 #
############################################################
# fmt: off
LOCAL_EDGE_MAP = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])
DPTS = np.array([
    [-0.078933,    0.04573333,  0.04573333,  0.04573333,  0.04573333,
      0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333],
    [ 0.25,        0.78571429,  0.07142857,  0.07142857,  0.07142857,
      0.39940358,  0.39940358,  0.39940358,  0.10059642,  0.10059642,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.07142857,  0.78571429,
      0.10059642,  0.10059642,  0.39940358,  0.39940358,  0.39940358,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.78571429,  0.07142857,
      0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.10059642,  0.39940358],
    [ 0.25,        0.07142857,  0.78571429,  0.07142857,  0.07142857,
      0.10059642,  0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.39940358],
], dtype=np.float64)
# fmt: on

############################################################
#                      MAIN ASSEMBLER                     #
############################################################


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True, nogil=True)
def leg2_tet_mass(tet_vertices, cond_tet):

    Amat = np.empty((10, 10), dtype=np.float64)

    txs, tys, tzs = tet_vertices
    aas, bbs, ccs, dds, V = tet_coefficients(txs, tys, tzs)

    coeff = np.empty((4, 4), dtype=np.float64)
    coeff[0, :] = aas / (6 * V)
    coeff[1, :] = bbs / (6 * V)
    coeff[2, :] = ccs / (6 * V)
    coeff[3, :] = dds / (6 * V)

    WEIGHTS = DPTS[0, :]
    DPTS1 = DPTS[1, :]
    DPTS2 = DPTS[2, :]
    DPTS3 = DPTS[3, :]
    DPTS4 = DPTS[4, :]

    xs = txs[0] * DPTS1 + txs[1] * DPTS2 + txs[2] * DPTS3 + txs[3] * DPTS4
    ys = tys[0] * DPTS1 + tys[1] * DPTS2 + tys[2] * DPTS3 + tys[3] * DPTS4
    zs = tzs[0] * DPTS1 + tzs[1] * DPTS2 + tzs[2] * DPTS3 + tzs[3] * DPTS4

    cs = np.empty((3, xs.shape[0]), dtype=np.float64)
    cs[0, :] = xs
    cs[1, :] = ys
    cs[2, :] = zs

    # Assemble the vertex contributions

    for iv1 in range(10):
        if iv1 < 4:
            val_gn1 = matmul(cond_tet, grad_n1(coeff[:, iv1], cs))
        else:
            ie1 = LOCAL_EDGE_MAP[:, iv1 - 4]
            val_gn1 = matmul(cond_tet, grad_n2(coeff[:, ie1], cs))

        for iv2 in range(10):
            if iv2 < 4:
                val_gn2 = grad_n1(coeff[:, iv2], cs)
            else:
                ie2 = LOCAL_EDGE_MAP[:, iv2 - 4]
                val_gn2 = grad_n2(coeff[:, ie2], cs)

            dotNiNj = (
                val_gn1[0, :] * val_gn2[0, :]
                + val_gn1[1, :] * val_gn2[1, :]
                + val_gn1[2, :] * val_gn2[2, :]
            )

            Amat[iv1, iv2] = np.sum(dotNiNj * WEIGHTS)

    return Amat * V


############################################################
#                  MAIN ASSEMBLER ROUTINE                 #
############################################################


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(f8[:, :], i8[:, :], i8[:, :], f8[:, :, :]),
    cache=True,
    nogil=True,
    parallel=True,
)
def _matrix_builder(nodes, tets, tet_to_field, cond_termal):
    nT = tets.shape[1]

    nnz = nT * 100

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    Amatrix = np.empty_like(rows, dtype=np.float64)

    for itet in prange(nT):
        p = itet * 100
        cond_tet = cond_termal[:, :, itet]

        Asub = leg2_tet_mass(nodes[:, tets[:, itet]], cond_tet)

        indices = tet_to_field[:, itet]
        for ii in range(10):
            rows[p + 10 * ii : p + 10 * (ii + 1)] = indices[ii]
            cols[p + ii : p + 100 + ii : 10] = indices[ii]

        Amatrix[p : p + 100] = Asub.ravel()

    return Amatrix, rows, cols


def tet_mass_matrix(
    field: Legrange2, cond_termal: np.ndarray
) -> tuple[np.ndarray, CSCMapping]:
    """Main assembly function of the mass matrix for stationary heat transfer

    Args:
        field (Legrange2): The Legrange2 Field object
        cond_termal (np.ndarray): Array with 3x3 conductivity tensors for each tet

    Returns:
        tuple[np.ndarray, CSCMapping]: The A-matrix entry and CSCMapping object
    """
    tets = field.mesh.tets
    nodes = field.mesh.nodes

    tet_to_field = field.tet_to_field

    Amatrix, rows, cols = _matrix_builder(nodes, tets, tet_to_field, cond_termal)

    cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return Amatrix, cscmap
