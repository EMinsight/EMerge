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
from numba import njit, f8, i8, prange  # type: ignore
from numba import get_thread_id as get_thread_id
import numpy as np

EPS = 1e-8
GRID_RES = 20


# fmt: off
@njit(i8[:](i8[:, :]), cache=True, nogil=True, parallel=True)
def matmax(A: np.ndarray) -> np.ndarray:
    ncols = A.shape[0]
    out = np.empty((ncols,), dtype=np.int64)
    for j in prange(ncols):
        out[j] = np.max(A[j, :])
    return out


@njit(cache=True, nogil=True)
def build_grid(
    coords: np.ndarray,         # (3, nNodes)
    res: int,
) -> tuple:
    """
    Build a CSR-style spatial grid over the query points (coords).

    Returns
    -------
    cell_offsets  : int32[res**3 + 1]  — start index of each cell in cell_contents
    cell_contents : int32[nNodes]      — point indices grouped by cell
    xmin, ymin, zmin, dx, dy, dz      — grid geometry scalars
    """
    nNodes = coords.shape[1]
    n_cells = res * res * res

    # ── bounding box ──
    xmin = coords[0, 0]
    xmax = coords[0, 0]
    ymin = coords[1, 0]
    ymax = coords[1, 0]
    zmin = coords[2, 0]
    zmax = coords[2, 0]

    for j in range(1, nNodes):
        x = coords[0, j]
        y = coords[1, j]
        z = coords[2, j]
        if x < xmin: 
            xmin = x
        if x > xmax: 
            xmax = x
        if y < ymin: 
            ymin = y
        if y > ymax: 
            ymax = y
        if z < zmin: 
            zmin = z
        if z > zmax: 
            zmax = z

    # small padding so no point lands exactly on the far edge
    xmin -= 1e-3
    xmax += 1e-3
    ymin -= 1e-3
    ymax += 1e-3
    zmin -= 1e-3
    zmax += 1e-3

    dx = (xmax - xmin) / res
    dy = (ymax - ymin) / res
    dz = (zmax - zmin) / res

    # ── count points per cell (first pass) ──
    counts = np.zeros(n_cells, dtype=np.int32)
    for j in range(nNodes):
        ix = int((coords[0, j] - xmin) / dx)
        iy = int((coords[1, j] - ymin) / dy)
        iz = int((coords[2, j] - zmin) / dz)
        if ix >= res: 
            ix = res - 1
        if iy >= res: 
            iy = res - 1
        if iz >= res: 
            iz = res - 1
        counts[ix + iy * res + iz * res * res] += 1

    # ── prefix sum → offsets ──
    cell_offsets = np.empty(n_cells + 1, dtype=np.int32)
    cell_offsets[0] = 0
    for c in range(n_cells):
        cell_offsets[c + 1] = cell_offsets[c] + counts[c]

    # ── fill contents (second pass) ──
    cell_contents = np.empty(nNodes, dtype=np.int32)
    fill_pos = counts.copy()   # reuse counts as a write-cursor per cell
    # reset to cell start positions first
    for c in range(n_cells):
        fill_pos[c] = cell_offsets[c]
    for j in range(nNodes):
        ix = int((coords[0, j] - xmin) / dx)
        iy = int((coords[1, j] - ymin) / dy)
        iz = int((coords[2, j] - zmin) / dz)
        if ix >= res: 
            ix = res - 1
        if iy >= res: 
            iy = res - 1
        if iz >= res: 
            iz = res - 1
        c = ix + iy * res + iz * res * res
        cell_contents[fill_pos[c]] = j
        fill_pos[c] += 1

    return cell_offsets, cell_contents, xmin, ymin, zmin, dx, dy, dz


@njit(cache=True, nogil=True)
def index_interp_inner(
    coords: np.ndarray,
    tets: np.ndarray,
    nodes: np.ndarray,
    tetids: np.ndarray,
    cell_offsets: np.ndarray,
    cell_contents: np.ndarray,
    xmin: float, ymin: float, zmin: float,
    dx: float,   dy: float,   dz: float,
    res: int,
    NThreads: int,
) -> np.ndarray:
    """
    Inner (parallelised) loop.  Grid arrays are passed in so that build_grid
    can be called once outside and re-used across multiple calls with the same
    query-point set.
    """
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]
    assigned = np.full((nNodes, NThreads), -1, dtype=np.int64)

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]

        iv1 = tets[0, itet]
        iv2 = tets[1, itet]
        iv3 = tets[2, itet]
        iv4 = tets[3, itet]

        v1x = nodes[0, iv1]
        v1y = nodes[1, iv1]
        v1z = nodes[2, iv1]

        m00 = nodes[0, iv2] - v1x
        m10 = nodes[1, iv2] - v1y
        m20 = nodes[2, iv2] - v1z
        m01 = nodes[0, iv3] - v1x
        m11 = nodes[1, iv3] - v1y
        m21 = nodes[2, iv3] - v1z
        m02 = nodes[0, iv4] - v1x
        m12 = nodes[1, iv4] - v1y
        m22 = nodes[2, iv4] - v1z

        det = (m00 * (m11 * m22 - m12 * m21)
             - m01 * (m10 * m22 - m12 * m20)
             + m02 * (m10 * m21 - m11 * m20))
        inv_det = 1.0 / det

        b00 = (m11 * m22 - m12 * m21) * inv_det
        b01 = (m02 * m21 - m01 * m22) * inv_det
        b02 = (m01 * m12 - m02 * m11) * inv_det
        b10 = (m12 * m20 - m10 * m22) * inv_det
        b11 = (m00 * m22 - m02 * m20) * inv_det
        b12 = (m10 * m02 - m00 * m12) * inv_det
        b20 = (m10 * m21 - m11 * m20) * inv_det
        b21 = (m20 * m01 - m00 * m21) * inv_det
        b22 = (m00 * m11 - m10 * m01) * inv_det

        # ── tet axis-aligned bounding box ──
        bx0 = v1x
        bx1 = v1x
        by0 = v1y
        by1 = v1y
        bz0 = v1z
        bz1 = v1z
        for k in range(1, 4):
            ivk = tets[k, itet]
            nx = nodes[0, ivk]
            ny = nodes[1, ivk]
            nz = nodes[2, ivk]
            if nx < bx0: 
                bx0 = nx
            if nx > bx1: 
                bx1 = nx
            if ny < by0: 
                by0 = ny
            if ny > by1: 
                by1 = ny
            if nz < bz0: 
                bz0 = nz
            if nz > bz1: 
                bz1 = nz

        # ── grid cells that overlap the bbox ──
        ix0 = int((bx0 - xmin) / dx) - 1
        ix1 = int((bx1 - xmin) / dx) + 1
        iy0 = int((by0 - ymin) / dy) - 1
        iy1 = int((by1 - ymin) / dy) + 1
        iz0 = int((bz0 - zmin) / dz) - 1
        iz1 = int((bz1 - zmin) / dz) + 1

        # clamp
        if ix0 < 0:
            ix0 = 0
        if iy0 < 0:
            iy0 = 0
        if iz0 < 0:
            iz0 = 0
        if ix1 >= res:
            ix1 = res - 1
        if iy1 >= res:
            iy1 = res - 1
        if iz1 >= res:
            iz1 = res - 1

        tid = get_thread_id()

        # ── only test points inside the bbox cells ──
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for iz in range(iz0, iz1 + 1):
                    cell_id = ix + iy * res + iz * res * res
                    start = cell_offsets[cell_id]
                    end   = cell_offsets[cell_id + 1]
                    for ci in range(start, end):
                        j = cell_contents[ci]
                        dx_ = coords[0, j] - v1x
                        dy_ = coords[1, j] - v1y
                        dz_ = coords[2, j] - v1z
                        u = b00 * dx_ + b01 * dy_ + b02 * dz_
                        v = b10 * dx_ + b11 * dy_ + b12 * dz_
                        w = b20 * dx_ + b21 * dy_ + b22 * dz_
                        if (u >= -EPS) and (v >= -EPS) and (w >= -EPS) and (u + v + w <= 1.0 + EPS):
                            assigned[j, tid] = itet

    return matmax(assigned)


def index_interp(
    coords: np.ndarray,
    tets: np.ndarray,
    nodes: np.ndarray,
    tetids: np.ndarray,
    NThreads: int = 6,
    res: int = GRID_RES,
) -> np.ndarray:
    """
    Public entry point.  Builds the spatial grid once, then dispatches to
    the JIT-compiled inner loop.

    Parameters
    ----------
    coords   : float64 (3, nNodes)   — query point coordinates
    tets     : int64   (4, nTets)    — vertex indices per tet
    nodes    : float64 (3, nMeshNodes) — mesh node coordinates
    tetids   : int64   (nTetIds,)    — subset of tet indices to search
    NThreads : int                   — number of Numba threads (default 6)
    res      : int                   — grid resolution per axis (default 20)

    Returns
    -------
    assigned : int64 (nNodes,)  — tet index for each query point, -1 if none
    """
    cell_offsets, cell_contents, xmin, ymin, zmin, dx, dy, dz = build_grid(coords, res)
    return index_interp_inner(
        coords, tets, nodes, tetids,
        cell_offsets, cell_contents,
        xmin, ymin, zmin, dx, dy, dz,
        res, NThreads,
    )
