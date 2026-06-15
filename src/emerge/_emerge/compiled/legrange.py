from numba import njit, c16, f8, i8, types
import numpy as np

SCALE_LENGTH = False


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nv(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    out = (ai + bi * xs + ci * ys + di * zs) * (
        2 * ai + 2 * bi * xs + 2 * ci * ys + 2 * di * zs - 1
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    out = 4 * (ai + bi * xs + ci * ys + di * zs) * (aj + bj * xs + cj * ys + dj * zs)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nv_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (ai + bi * xs + ci * ys) * (2 * ai + 2 * bi * xs + 2 * ci * ys - 1)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = 4 * (ai + bi * xs + ci * ys) * (aj + bj * xs + cj * ys)
    return out.astype(np.complex128)


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nv_grad(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = bi * (4 * ai + 4 * bi * xs + 4 * ci * ys + 4 * di * zs - 1)
    by = ci * (4 * ai + 4 * bi * xs + 4 * ci * ys + 4 * di * zs - 1)
    bz = di * (4 * ai + 4 * bi * xs + 4 * ci * ys + 4 * di * zs - 1)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne_grad(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = 4 * bi * (aj + bj * xs + cj * ys + dj * zs) + 4 * bj * (
        ai + bi * xs + ci * ys + di * zs
    )
    by = 4 * ci * (aj + bj * xs + cj * ys + dj * zs) + 4 * cj * (
        ai + bi * xs + ci * ys + di * zs
    )
    bz = 4 * di * (aj + bj * xs + cj * ys + dj * zs) + 4 * dj * (
        ai + bi * xs + ci * ys + di * zs
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nv_grad_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = bi * (4 * ai + 4 * bi * xs + 4 * ci * ys - 1)
    by = ci * (4 * ai + 4 * bi * xs + 4 * ci * ys - 1)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne_grad_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = 4 * bi * (aj + bj * xs + cj * ys) + 4 * bj * (ai + bi * xs + ci * ys)
    by = 4 * ci * (aj + bj * xs + cj * ys) + 4 * cj * (ai + bi * xs + ci * ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nv_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = ci * (4 * ai + 4 * bi * xs + 4 * ci * ys - 1)
    by = bi * (-4 * ai - 4 * bi * xs - 4 * ci * ys + 1)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = 4 * ci * (aj + bj * xs + cj * ys) + 4 * cj * (ai + bi * xs + ci * ys)
    by = -4 * bi * (aj + bj * xs + cj * ys) - 4 * bj * (ai + bi * xs + ci * ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out
