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
    bx = -(
        bj * (ak + bk * xs + ck * ys + dk * zs)
        - bk * (aj + bj * xs + cj * ys + dj * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
    by = -(
        cj * (ak + bk * xs + ck * ys + dk * zs)
        - ck * (aj + bj * xs + cj * ys + dj * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
    bz = -(
        dj * (ak + bk * xs + ck * ys + dk * zs)
        - dk * (aj + bj * xs + cj * ys + dj * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
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
    bx = -(
        bi * (ak + bk * xs + ck * ys + dk * zs)
        - bk * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
    by = -(
        ci * (ak + bk * xs + ck * ys + dk * zs)
        - ck * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
    bz = -(
        di * (ak + bk * xs + ck * ys + dk * zs)
        - dk * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
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
        -ci
        * (
            dj * (ak + bk * xs + ck * ys + dk * zs)
            - dk * (aj + bj * xs + cj * ys + dj * zs)
        )
        + di
        * (
            cj * (ak + bk * xs + ck * ys + dk * zs)
            - ck * (aj + bj * xs + cj * ys + dj * zs)
        )
        + 2 * (cj * dk - ck * dj) * (ai + bi * xs + ci * ys + di * zs)
    )
    by = (
        bi
        * (
            dj * (ak + bk * xs + ck * ys + dk * zs)
            - dk * (aj + bj * xs + cj * ys + dj * zs)
        )
        - di
        * (
            bj * (ak + bk * xs + ck * ys + dk * zs)
            - bk * (aj + bj * xs + cj * ys + dj * zs)
        )
        - 2 * (bj * dk - bk * dj) * (ai + bi * xs + ci * ys + di * zs)
    )
    bz = (
        -bi
        * (
            cj * (ak + bk * xs + ck * ys + dk * zs)
            - ck * (aj + bj * xs + cj * ys + dj * zs)
        )
        + ci
        * (
            bj * (ak + bk * xs + ck * ys + dk * zs)
            - bk * (aj + bj * xs + cj * ys + dj * zs)
        )
        + 2 * (bj * ck - bk * cj) * (ai + bi * xs + ci * ys + di * zs)
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
        -cj
        * (
            di * (ak + bk * xs + ck * ys + dk * zs)
            - dk * (ai + bi * xs + ci * ys + di * zs)
        )
        + dj
        * (
            ci * (ak + bk * xs + ck * ys + dk * zs)
            - ck * (ai + bi * xs + ci * ys + di * zs)
        )
        + 2 * (ci * dk - ck * di) * (aj + bj * xs + cj * ys + dj * zs)
    )
    by = (
        bj
        * (
            di * (ak + bk * xs + ck * ys + dk * zs)
            - dk * (ai + bi * xs + ci * ys + di * zs)
        )
        - dj
        * (
            bi * (ak + bk * xs + ck * ys + dk * zs)
            - bk * (ai + bi * xs + ci * ys + di * zs)
        )
        - 2 * (bi * dk - bk * di) * (aj + bj * xs + cj * ys + dj * zs)
    )
    bz = (
        -bj
        * (
            ci * (ak + bk * xs + ck * ys + dk * zs)
            - ck * (ai + bi * xs + ci * ys + di * zs)
        )
        + cj
        * (
            bi * (ak + bk * xs + ck * ys + dk * zs)
            - bk * (ai + bi * xs + ci * ys + di * zs)
        )
        + 2 * (bi * ck - bk * ci) * (aj + bj * xs + cj * ys + dj * zs)
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
    bx = -(bj * (ak + bk * xs + ck * ys) - bk * (aj + bj * xs + cj * ys)) * (
        ai + bi * xs + ci * ys
    )
    by = -(cj * (ak + bk * xs + ck * ys) - ck * (aj + bj * xs + cj * ys)) * (
        ai + bi * xs + ci * ys
    )
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
    bx = -(bi * (ak + bk * xs + ck * ys) - bk * (ai + bi * xs + ci * ys)) * (
        aj + bj * xs + cj * ys
    )
    by = -(ci * (ak + bk * xs + ck * ys) - ck * (ai + bi * xs + ci * ys)) * (
        aj + bj * xs + cj * ys
    )
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    return out


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (2 * bi * cj - 2 * bj * ci) * np.ones_like(xs)
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
        -bi * (cj * (ak + bk * xs + ck * ys) - ck * (aj + bj * xs + cj * ys))
        + ci * (bj * (ak + bk * xs + ck * ys) - bk * (aj + bj * xs + cj * ys))
        + 2 * (bj * ck - bk * cj) * (ai + bi * xs + ci * ys)
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
        -bj * (ci * (ak + bk * xs + ck * ys) - ck * (ai + bi * xs + ci * ys))
        + cj * (bi * (ak + bk * xs + ck * ys) - bk * (ai + bi * xs + ci * ys))
        + 2 * (bi * ck - bk * ci) * (aj + bj * xs + cj * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_div_tri(coeff, coords, i, j, k):
    out = np.zeros(coords.shape[1], dtype=np.complex128)
    return out


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    out = (2 * bi * bj + 2 * ci * cj) * np.ones_like(xs)
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = -bi * (bj * (ak + bk * xs + ck * ys) - bk * (aj + bj * xs + cj * ys)) - ci * (
        cj * (ak + bk * xs + ck * ys) - ck * (aj + bj * xs + cj * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = -bj * (bi * (ak + bk * xs + ck * ys) - bk * (ai + bi * xs + ci * ys)) - cj * (
        ci * (ak + bk * xs + ck * ys) - ck * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)
