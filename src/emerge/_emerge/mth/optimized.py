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
from numba import njit, f8, i8, types, c16
import numpy as np
from . import _cache_check
from .keast_rules import _KEAST_PTS
from .strang_rules import _STRANG_PTS

############################################################
#                         FUNCTIONS                        #
############################################################


def gaus_quad_tri(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature triangle sample points W, L1, L2, L3, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    return _STRANG_PTS[p]


def gaus_quad_tet(p: int) -> np.ndarray:
    """
    Returns the duvanant quadrature tetrahedron sample points W, L1, L2, L3, L4, coordinates for a given order p.

    Parameters
    ----------
    p : int
        The order of the quadrature rule.
    Returns
    -------
    pts : np.ndarray
        The sample points W, L1, L2, L3.
    -------

    P = dunavant_points(p)
    P[0,:] = Weights
    P[1,:] = L1 values
    P[2,:] = L2 values
    P[3,:] = L3 values
    """

    return _KEAST_PTS[p]


############################################################
#                      NUMBA COMPILED                     #
############################################################


@njit(
    types.Tuple((f8[:], f8[:], f8[:], i8[:]))(f8[:, :], i8[:, :], f8[:, :]),
    cache=True,
    nogil=True,
)
def generate_int_points_tri(nodes: np.ndarray, triangles: np.ndarray, PTS: np.ndarray):

    nDPTs = PTS.shape[1]
    xall = np.zeros((nDPTs, triangles.shape[1]))
    yall = np.zeros((nDPTs, triangles.shape[1]))
    zall = np.zeros((nDPTs, triangles.shape[1]))

    for it in range(triangles.shape[1]):
        vertex_ids = triangles[:, it]

        x1, x2, x3 = nodes[0, vertex_ids]
        y1, y2, y3 = nodes[1, vertex_ids]
        z1, z2, z3 = nodes[2, vertex_ids]

        xspts = x1 * PTS[1, :] + x2 * PTS[2, :] + x3 * PTS[3, :]
        yspts = y1 * PTS[1, :] + y2 * PTS[2, :] + y3 * PTS[3, :]
        zspts = z1 * PTS[1, :] + z2 * PTS[2, :] + z3 * PTS[3, :]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    shape = np.array((nDPTs, triangles.shape[1]))

    return xall_flat, yall_flat, zall_flat, shape


@njit(f8(f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Computes a dot product of two 3D vectors

    Args:
        a (np.ndarray): (3,) array
        b (np.ndarray): (3,) array

    Returns:
        float: a · b
    """
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]

    return a0 * b0 + a1 * b1 + a2 * b2


@njit(c16(c16[:], c16[:]), cache=True, fastmath=True, nogil=True)
def dot_c(a: np.ndarray, b: np.ndarray) -> complex:
    """Computes the complex dot product of two 3D vectors

    Args:
        a (np.ndarray): (3,) array
        b (np.ndarray): (3,) array

    Returns:
        complex: a · b
    """
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]

    return a0 * b0 + a1 * b1 + a2 * b2


@njit(f8[:](f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    crossv = np.empty((3,), dtype=np.float64)
    crossv[0] = a[1] * b[2] - a[2] * b[1]
    crossv[1] = a[2] * b[0] - a[0] * b[2]
    crossv[2] = a[0] * b[1] - a[1] * b[0]
    return crossv


@njit(c16[:](c16[:], c16[:]), cache=True, fastmath=True, nogil=True)
def cross_c(a: np.ndarray, b: np.ndarray):
    """Optimized complex single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    crossv = np.empty((3,), dtype=np.complex128)
    crossv[0] = a[1] * b[2] - a[2] * b[1]
    crossv[1] = a[2] * b[0] - a[0] * b[2]
    crossv[2] = a[0] * b[1] - a[1] * b[0]
    return crossv


@njit(f8[:](f8[:], f8[:], f8[:], f8[:]), cache=True, nogil=True)
def calc_inward_normal(
    n1: np.ndarray, n2: np.ndarray, n3: np.ndarray, inward_normal: np.ndarray
) -> np.ndarray:
    """Copmutes an inward normal vector of a triangle spanned by 3 points.

    Computes the triangle surface (n1, n2, n3) which have normal n.
    The normal is aligned with respect to an origin.

    Args:
        n1 (np.ndarray): Node 1 (3,) array
        n2 (np.ndarray): Node 2 (3,) array
        n3 (np.ndarray): Node 3 (3,) array
        o (np.ndarray): The alignment origin

    Returns:
        Node 1 (3,) array: The inward pointing normal
    """
    e1 = n2 - n1
    e2 = n3 - n1
    n = cross(e1, e2)
    n = n / np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    sgn = 1
    if dot(n, inward_normal) < 0:
        sgn = -1
    return n * sgn


@njit(f8(f8[:], f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def calc_area(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    """Computes the tirangle surface area of the traingle spun by three nodes.

    Args:
        x1 (np.ndarray): Node 1 (3,) array
        x2 (np.ndarray): Node 2 (3,) array
        x3 (np.ndarray): Node 3 (3,) array

    Returns:
        float: The surface area
    """
    e1 = x2 - x1
    e2 = x3 - x1
    av = cross(e1, e2)
    return np.sqrt(av[0] ** 2 + av[1] ** 2 + av[2] ** 2) / 2


@njit(f8(f8[:], f8[:], f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def calc_volume(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray
) -> float:
    """Computes the volume of the tetrahedron spun by four nodes.

    Args:
        x1 (np.ndarray): Node 1 (3,) array
        x2 (np.ndarray): Node 2 (3,) array
        x3 (np.ndarray): Node 3 (3,) array
        x4 (np.ndarray): Node 4 (3,) array

    Returns:
        float: The volume
    """
    a_x, a_y, a_z = x1[0] - x4[0], x1[1] - x4[1], x1[2] - x4[2]
    b_x, b_y, b_z = x2[0] - x4[0], x2[1] - x4[1], x2[2] - x4[2]
    c_x, c_y, c_z = x3[0] - x4[0], x3[1] - x4[1], x3[2] - x4[2]

    vol = (
        a_x * (b_y * c_z - b_z * c_y)
        - a_y * (b_x * c_z - b_z * c_x)
        + a_z * (b_x * c_y - b_y * c_x)
    )

    return abs(vol) / 6.0


@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex indices of one tetrahedron, in *its* order
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


@njit(f8[:, :](f8[:], f8[:], f8[:]), cache=True, nogil=True, fastmath=True)
def compute_distances(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    N = xs.shape[0]
    Ds = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i, N):
            Ds[i, j] = np.sqrt(
                (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 + (zs[i] - zs[j]) ** 2
            )
            Ds[j, i] = Ds[i, j]
    return Ds


ids = np.array([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=np.int64)


@njit(c16[:, :](c16[:, :]), cache=True, nogil=True)
def matinv(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.zeros((3, 3), dtype=np.complex128)

    if (
        M[0, 1] == 0
        and M[0, 2] == 0
        and M[1, 0] == 0
        and M[1, 2] == 0
        and M[2, 0] == 0
        and M[2, 1] == 0
    ):
        out[0, 0] = 1 / M[0, 0]
        out[1, 1] = 1 / M[1, 1]
        out[2, 2] = 1 / M[2, 2]
    else:
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


@njit(f8[:, :](f8[:, :]), cache=True, nogil=True)
def matinv_r(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.zeros((3, 3), dtype=np.float64)

    if (
        M[0, 1] == 0
        and M[0, 2] == 0
        and M[1, 0] == 0
        and M[1, 2] == 0
        and M[2, 0] == 0
        and M[2, 1] == 0
    ):
        out[0, 0] = 1 / M[0, 0]
        out[1, 1] = 1 / M[1, 1]
        out[2, 2] = 1 / M[2, 2]
    else:
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


@njit(cache=True, nogil=True)
def matmul(M: np.ndarray, vecs: np.ndarray):
    """Executes a basis transformation of vectors (3,N) with a basis matrix M

    Args:
        M (np.ndarray): A (3,3) basis matrix
        vec (np.ndarray): A (3,N) set of coordinates

    Returns:
        np.ndarray: The transformed (3,N) set of vectors
    """
    out = np.empty((3, vecs.shape[1]), dtype=vecs.dtype)
    out[0, :] = M[0, 0] * vecs[0, :] + M[0, 1] * vecs[1, :] + M[0, 2] * vecs[2, :]
    out[1, :] = M[1, 0] * vecs[0, :] + M[1, 1] * vecs[1, :] + M[1, 2] * vecs[2, :]
    out[2, :] = M[2, 0] * vecs[0, :] + M[2, 1] * vecs[1, :] + M[2, 2] * vecs[2, :]
    return out


@njit(cache=True, nogil=True)
def _subsample_coordinates(
    xs: np.ndarray, ys: np.ndarray, tolerance: float, xmin: float
) -> tuple[np.ndarray, np.ndarray]:
    """This function takes a set of x and y coordinates in a finely sampled set and returns a reduced
    set of numbers that traces the input curve within a provided tolerance.

    Args:
        xs (np.ndarray): The set of X-coordinates
        ys (np.ndarray): The set of Y-coordinates
        tolerance (float): The maximum deviation of the curve in meters
        xmin (float): The minimal distance to the next point.

    Returns:
        np.ndarray: The output X-coordinates
        np.ndarray: The output Y-coordinates
    """
    N = xs.shape[0]
    ids = np.zeros((N,), dtype=np.int32)
    store_index = 1
    start_index = 0
    final_index = 0
    for iteration in range(N):
        i1 = start_index
        done = 0
        for i2 in range(i1 + 1, N):
            x_true = xs[i1 : i2 + 1]
            y_true = ys[i1 : i2 + 1]

            x_f = np.linspace(xs[i1], xs[i2], i2 - i1 + 1)
            y_f = np.linspace(ys[i1], ys[i2], i2 - i1 + 1)
            error = np.max(np.sqrt((x_f - x_true) ** 2 + (y_f - y_true) ** 2))
            ds = np.sqrt((xs[i2] - xs[i1]) ** 2 + (ys[i2] - ys[i1]) ** 2)
            # If at the end
            if i2 == N - 1:
                ids[store_index] = i2 - 1
                final_index = store_index + 1
                done = 1
                break
            # If not yet past the minimum distance, accumulate more
            if ds < xmin:
                continue
            # If the end is less than a minimum distance
            if np.sqrt((ys[-1] - ys[i2]) ** 2 + (xs[-1] - xs[i2]) ** 2) < xmin:
                imid = i1 + (N - 1 - i1) // 2
                ids[store_index] = imid
                ids[store_index + 1] = N - 1
                final_index = store_index + 2
                done = 1
                break
            if error < tolerance:
                continue
            else:
                ids[store_index] = i2 - 1
                start_index = i2
                store_index = store_index + 1
                break
        if done == 1:
            break
    return xs[ids[0:final_index]], ys[ids[0:final_index]]


@njit(f8(f8[:], f8[:], f8[:]), cache=True, nogil=True)
def area(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    """Computes the area given three coordinates:

    Args:
        x1 (np.ndarray): (3,) vector
        x2 (np.ndarray): (3,) vector
        x3 (np.ndarray): (3,) vector

    Returns:
        area (float)
    """
    e1 = x2 - x1
    e2 = x3 - x1
    av = np.array(
        [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
    )
    return np.sqrt(av[0] ** 2 + av[1] ** 2 + av[2] ** 2) / 2


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8[:], i8[:]))(
        f8[:, :], i8[:, :], f8[:, :]
    ),
    cache=True,
    nogil=True,
)
def generate_int_data_tet(nodes: np.ndarray, tetrahedra: np.ndarray, PTS: np.ndarray):
    """Generate volumetric integration points

    Args:
        nodes (np.ndarray): _description_
        tetrahedra (np.ndarray): _description_
        PTS (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    nDPTs = PTS.shape[1]
    xall = np.zeros((nDPTs, tetrahedra.shape[1]))
    yall = np.zeros((nDPTs, tetrahedra.shape[1]))
    zall = np.zeros((nDPTs, tetrahedra.shape[1]))
    wall = np.zeros((nDPTs, tetrahedra.shape[1]))
    aall = np.zeros((nDPTs, tetrahedra.shape[1]))
    for it in range(tetrahedra.shape[1]):
        vertex_ids = tetrahedra[:, it]

        x1, x2, x3, x4 = nodes[0, vertex_ids]
        y1, y2, y3, y4 = nodes[1, vertex_ids]
        z1, z2, z3, z4 = nodes[2, vertex_ids]

        xspts = x1 * PTS[1, :] + x2 * PTS[2, :] + x3 * PTS[3, :] + x4 * PTS[4, :]
        yspts = y1 * PTS[1, :] + y2 * PTS[2, :] + y3 * PTS[3, :] + y4 * PTS[4, :]
        zspts = z1 * PTS[1, :] + z2 * PTS[2, :] + z3 * PTS[3, :] + z4 * PTS[4, :]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts
        wall[:, it] = PTS[0, :]
        aall[:, it] = calc_volume(
            nodes[:, vertex_ids[0]],
            nodes[:, vertex_ids[1]],
            nodes[:, vertex_ids[2]],
            nodes[:, vertex_ids[3]],
        )

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    wall_flat = wall.flatten()
    aall_flat = aall.flatten()

    shape = np.array((nDPTs, tetrahedra.shape[1]))

    return xall_flat, yall_flat, zall_flat, wall_flat, aall_flat, shape


@njit(
    types.Tuple((f8[:], f8[:], f8[:], f8[:], f8[:], i8[:]))(
        f8[:, :], i8[:, :], f8[:, :]
    ),
    cache=True,
    nogil=True,
)
def generate_int_data_tri(nodes: np.ndarray, triangles: np.ndarray, PTS: np.ndarray):

    nDPTs = PTS.shape[1]
    xall = np.zeros((nDPTs, triangles.shape[1]))
    yall = np.zeros((nDPTs, triangles.shape[1]))
    zall = np.zeros((nDPTs, triangles.shape[1]))
    wall = np.zeros((nDPTs, triangles.shape[1]))
    aall = np.zeros((nDPTs, triangles.shape[1]))
    for it in range(triangles.shape[1]):
        vertex_ids = triangles[:, it]

        x1, x2, x3 = nodes[0, vertex_ids]
        y1, y2, y3 = nodes[1, vertex_ids]
        z1, z2, z3 = nodes[2, vertex_ids]

        xspts = x1 * PTS[1, :] + x2 * PTS[2, :] + x3 * PTS[3, :]
        yspts = y1 * PTS[1, :] + y2 * PTS[2, :] + y3 * PTS[3, :]
        zspts = z1 * PTS[1, :] + z2 * PTS[2, :] + z3 * PTS[3, :]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts
        wall[:, it] = PTS[0, :]
        aall[:, it] = calc_area(
            nodes[:, vertex_ids[0]], nodes[:, vertex_ids[1]], nodes[:, vertex_ids[2]]
        )

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    wall_flat = wall.flatten()
    aall_flat = aall.flatten()

    shape = np.array((nDPTs, triangles.shape[1]))

    return xall_flat, yall_flat, zall_flat, wall_flat, aall_flat, shape


@njit(
    types.Tuple((f8[:], f8[:], f8[:], i8[:]))(f8[:, :], i8[:, :], f8[:, :]),
    cache=True,
    nogil=True,
)
def generate_int_points_tet(nodes: np.ndarray, tets: np.ndarray, PTS: np.ndarray):

    nPTS = PTS.shape[1]
    xall = np.zeros((nPTS, tets.shape[1]))
    yall = np.zeros((nPTS, tets.shape[1]))
    zall = np.zeros((nPTS, tets.shape[1]))

    for it in range(tets.shape[1]):
        vertex_ids = tets[:, it]

        x1, x2, x3, x4 = nodes[0, vertex_ids]
        y1, y2, y3, y4 = nodes[1, vertex_ids]
        z1, z2, z3, z4 = nodes[2, vertex_ids]

        xspts = x1 * PTS[1, :] + x2 * PTS[2, :] + x3 * PTS[3, :] + x4 * PTS[4, :]
        yspts = y1 * PTS[1, :] + y2 * PTS[2, :] + y3 * PTS[3, :] + y4 * PTS[4, :]
        zspts = z1 * PTS[1, :] + z2 * PTS[2, :] + z3 * PTS[3, :] + z4 * PTS[4, :]

        xall[:, it] = xspts
        yall[:, it] = yspts
        zall[:, it] = zspts

    xall_flat = xall.flatten()
    yall_flat = yall.flatten()
    zall_flat = zall.flatten()
    shape = np.array((nPTS, tets.shape[1]))

    return xall_flat, yall_flat, zall_flat, shape
