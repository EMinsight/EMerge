from numba import njit, c16, f8, i8, types
import numpy as np

SCALE_LENGTH = False


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = -bi * (aj + bj * xs + cj * ys + dj * zs) + bj * (
        ai + bi * xs + ci * ys + di * zs
    )
    by = -ci * (aj + bj * xs + cj * ys + dj * zs) + cj * (
        ai + bi * xs + ci * ys + di * zs
    )
    bz = -di * (aj + bj * xs + cj * ys + dj * zs) + dj * (
        ai + bi * xs + ci * ys + di * zs
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = bi * (aj + bj * xs + cj * ys + dj * zs) + bj * (
        ai + bi * xs + ci * ys + di * zs
    )
    by = ci * (aj + bj * xs + cj * ys + dj * zs) + cj * (
        ai + bi * xs + ci * ys + di * zs
    )
    bz = di * (aj + bj * xs + cj * ys + dj * zs) + dj * (
        ai + bi * xs + ci * ys + di * zs
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = bj * (ai + bi * xs + ci * ys + di * zs) * (ak + bk * xs + ck * ys + dk * zs)
    by = cj * (ai + bi * xs + ci * ys + di * zs) * (ak + bk * xs + ck * ys + dk * zs)
    bz = dj * (ai + bi * xs + ci * ys + di * zs) * (ak + bk * xs + ck * ys + dk * zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = bk * (ai + bi * xs + ci * ys + di * zs) * (aj + bj * xs + cj * ys + dj * zs)
    by = ck * (ai + bi * xs + ci * ys + di * zs) * (aj + bj * xs + cj * ys + dj * zs)
    bz = dk * (ai + bi * xs + ci * ys + di * zs) * (aj + bj * xs + cj * ys + dj * zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = 2 * ci * dj - 2 * cj * di
    by = -2 * bi * dj + 2 * bj * di
    bz = 2 * bi * cj - 2 * bj * ci
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_curl(coeff, coords, i, j, k):
    out = np.zeros((3, coords.shape[1]), dtype=np.complex128)
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = (
        ci * dj * (ak + bk * xs + ck * ys + dk * zs)
        - cj * di * (ak + bk * xs + ck * ys + dk * zs)
        - cj * dk * (ai + bi * xs + ci * ys + di * zs)
        + ck * dj * (ai + bi * xs + ci * ys + di * zs)
    )
    by = (
        -bi * dj * (ak + bk * xs + ck * ys + dk * zs)
        + bj * di * (ak + bk * xs + ck * ys + dk * zs)
        + bj * dk * (ai + bi * xs + ci * ys + di * zs)
        - bk * dj * (ai + bi * xs + ci * ys + di * zs)
    )
    bz = (
        bi * cj * (ak + bk * xs + ck * ys + dk * zs)
        - bj * ci * (ak + bk * xs + ck * ys + dk * zs)
        - bj * ck * (ai + bi * xs + ci * ys + di * zs)
        + bk * cj * (ai + bi * xs + ci * ys + di * zs)
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = (
        ci * dk * (aj + bj * xs + cj * ys + dj * zs)
        + cj * dk * (ai + bi * xs + ci * ys + di * zs)
        - ck * di * (aj + bj * xs + cj * ys + dj * zs)
        - ck * dj * (ai + bi * xs + ci * ys + di * zs)
    )
    by = (
        -bi * dk * (aj + bj * xs + cj * ys + dj * zs)
        - bj * dk * (ai + bi * xs + ci * ys + di * zs)
        + bk * di * (aj + bj * xs + cj * ys + dj * zs)
        + bk * dj * (ai + bi * xs + ci * ys + di * zs)
    )
    bz = (
        bi * ck * (aj + bj * xs + cj * ys + dj * zs)
        + bj * ck * (ai + bi * xs + ci * ys + di * zs)
        - bk * ci * (aj + bj * xs + cj * ys + dj * zs)
        - bk * cj * (ai + bi * xs + ci * ys + di * zs)
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = ai * bj - aj * bi - bi * cj * ys + bj * ci * ys
    by = ai * cj - aj * ci + bi * cj * xs - bj * ci * xs
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = bi * (aj + bj * xs + cj * ys) + bj * (ai + bi * xs + ci * ys)
    by = ci * (aj + bj * xs + cj * ys) + cj * (ai + bi * xs + ci * ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = bj * (ai + bi * xs + ci * ys) * (ak + bk * xs + ck * ys)
    by = cj * (ai + bi * xs + ci * ys) * (ak + bk * xs + ck * ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = bk * (ai + bi * xs + ci * ys) * (aj + bj * xs + cj * ys)
    by = ck * (ai + bi * xs + ci * ys) * (aj + bj * xs + cj * ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    out = (2 * bi * cj - 2 * bj * ci) * np.ones((coords.shape[1],), dtype=np.complex128)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_curl_tri(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        bi * cj * (ak + bk * xs + ck * ys)
        - bj * ci * (ak + bk * xs + ck * ys)
        - bj * ck * (ai + bi * xs + ci * ys)
        + bk * cj * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        bi * ck * (aj + bj * xs + cj * ys)
        + bj * ck * (ai + bi * xs + ci * ys)
        - bk * ci * (aj + bj * xs + cj * ys)
        - bk * cj * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_div_tri(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1]), dtype=np.complex128)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    out = (2 * bi * bj + 2 * ci * cj) * np.ones((coords.shape[1]), dtype=np.complex128)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        bi * bj * (ak + bk * xs + ck * ys)
        + bj * bk * (ai + bi * xs + ci * ys)
        + ci * cj * (ak + bk * xs + ck * ys)
        + cj * ck * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        bi * bk * (aj + bj * xs + cj * ys)
        + bj * bk * (ai + bi * xs + ci * ys)
        + ci * ck * (aj + bj * xs + cj * ys)
        + cj * ck * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)
