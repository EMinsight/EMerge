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
    bx = (
        bi * (aj + bj * xs + cj * ys + dj * zs)
        - bj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
    by = (
        ci * (aj + bj * xs + cj * ys + dj * zs)
        - cj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
    bz = (
        di * (aj + bj * xs + cj * ys + dj * zs)
        - dj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ai + bi * xs + ci * ys + di * zs)
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
    bx = (
        bi * (aj + bj * xs + cj * ys + dj * zs)
        - bj * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
    by = (
        ci * (aj + bj * xs + cj * ys + dj * zs)
        - cj * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
    bz = (
        di * (aj + bj * xs + cj * ys + dj * zs)
        - dj * (ai + bi * xs + ci * ys + di * zs)
    ) * (aj + bj * xs + cj * ys + dj * zs)
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
def _nf2(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = (
        bi * (aj + bj * xs + cj * ys + dj * zs)
        - bj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ak + bk * xs + ck * ys + dk * zs)
    by = (
        ci * (aj + bj * xs + cj * ys + dj * zs)
        - cj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ak + bk * xs + ck * ys + dk * zs)
    bz = (
        di * (aj + bj * xs + cj * ys + dj * zs)
        - dj * (ai + bi * xs + ci * ys + di * zs)
    ) * (ak + bk * xs + ck * ys + dk * zs)
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
    bx = (
        -3 * ai * ci * dj
        + 3 * ai * cj * di
        - 3 * bi * ci * dj * xs
        + 3 * bi * cj * di * xs
        - 3 * ci**2 * dj * ys
        + 3 * ci * cj * di * ys
        - 3 * ci * di * dj * zs
        + 3 * cj * di**2 * zs
    )
    by = (
        3 * ai * bi * dj
        - 3 * ai * bj * di
        + 3 * bi**2 * dj * xs
        - 3 * bi * bj * di * xs
        + 3 * bi * ci * dj * ys
        + 3 * bi * di * dj * zs
        - 3 * bj * ci * di * ys
        - 3 * bj * di**2 * zs
    )
    bz = (
        -3 * ai * bi * cj
        + 3 * ai * bj * ci
        - 3 * bi**2 * cj * xs
        + 3 * bi * bj * ci * xs
        - 3 * bi * ci * cj * ys
        - 3 * bi * cj * di * zs
        + 3 * bj * ci**2 * ys
        + 3 * bj * ci * di * zs
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
    return out


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = (
        -3 * aj * ci * dj
        + 3 * aj * cj * di
        - 3 * bj * ci * dj * xs
        + 3 * bj * cj * di * xs
        - 3 * ci * cj * dj * ys
        - 3 * ci * dj**2 * zs
        + 3 * cj**2 * di * ys
        + 3 * cj * di * dj * zs
    )
    by = (
        3 * aj * bi * dj
        - 3 * aj * bj * di
        + 3 * bi * bj * dj * xs
        + 3 * bi * cj * dj * ys
        + 3 * bi * dj**2 * zs
        - 3 * bj**2 * di * xs
        - 3 * bj * cj * di * ys
        - 3 * bj * di * dj * zs
    )
    bz = (
        -3 * aj * bi * cj
        + 3 * aj * bj * ci
        - 3 * bi * bj * cj * xs
        - 3 * bi * cj**2 * ys
        - 3 * bi * cj * dj * zs
        + 3 * bj**2 * ci * xs
        + 3 * bj * ci * cj * ys
        + 3 * bj * ci * dj * zs
    )
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0, :] = bx
    out[1, :] = by
    out[2, :] = bz
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
def _nf2_curl(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:, i]
    aj, bj, cj, dj = coeff[:, j]
    ak, bk, ck, dk = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    zs = coords[2, :]
    bx = (
        ck
        * (
            di * (aj + bj * xs + cj * ys + dj * zs)
            - dj * (ai + bi * xs + ci * ys + di * zs)
        )
        - dk
        * (
            ci * (aj + bj * xs + cj * ys + dj * zs)
            - cj * (ai + bi * xs + ci * ys + di * zs)
        )
        - 2 * (ci * dj - cj * di) * (ak + bk * xs + ck * ys + dk * zs)
    )
    by = (
        -bk
        * (
            di * (aj + bj * xs + cj * ys + dj * zs)
            - dj * (ai + bi * xs + ci * ys + di * zs)
        )
        + dk
        * (
            bi * (aj + bj * xs + cj * ys + dj * zs)
            - bj * (ai + bi * xs + ci * ys + di * zs)
        )
        + 2 * (bi * dj - bj * di) * (ak + bk * xs + ck * ys + dk * zs)
    )
    bz = (
        bk
        * (
            ci * (aj + bj * xs + cj * ys + dj * zs)
            - cj * (ai + bi * xs + ci * ys + di * zs)
        )
        - ck
        * (
            bi * (aj + bj * xs + cj * ys + dj * zs)
            - bj * (ai + bi * xs + ci * ys + di * zs)
        )
        - 2 * (bi * cj - bj * ci) * (ak + bk * xs + ck * ys + dk * zs)
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
    bx = (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) * (
        ai + bi * xs + ci * ys
    )
    by = (ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)) * (
        ai + bi * xs + ci * ys
    )
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
    bx = (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) * (
        aj + bj * xs + cj * ys
    )
    by = (ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)) * (
        aj + bj * xs + cj * ys
    )
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


@njit(c16[:, :](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    bx = (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) * (
        ak + bk * xs + ck * ys
    )
    by = (ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)) * (
        ak + bk * xs + ck * ys
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
    out = (
        -3 * ai * bi * cj
        + 3 * ai * bj * ci
        - 3 * bi**2 * cj * xs
        + 3 * bi * bj * ci * xs
        - 3 * bi * ci * cj * ys
        + 3 * bj * ci**2 * ys
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        -3 * aj * bi * cj
        + 3 * aj * bj * ci
        - 3 * bi * bj * cj * xs
        - 3 * bi * cj**2 * ys
        + 3 * bj**2 * ci * xs
        + 3 * bj * ci * cj * ys
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_curl_tri(coeff, coords, i, j, k):
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
def _nf2_curl_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = (
        bk * (ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys))
        - ck * (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys))
        - 2 * (bi * cj - bj * ci) * (ak + bk * xs + ck * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne1_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = bi * (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) + ci * (
        ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _ne2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    xs = coords[0, :]
    ys = coords[1, :]
    out = bj * (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) + cj * (
        ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf1_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = -bj * (bi * (ak + bk * xs + ck * ys) - bk * (ai + bi * xs + ci * ys)) - cj * (
        ci * (ak + bk * xs + ck * ys) - ck * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)


@njit(c16[:](f8[:, :], f8[:, :], i8, i8, i8), cache=True, nogil=True)
def _nf2_div_tri(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:, i]
    aj, bj, cj = coeff[:, j]
    ak, bk, ck = coeff[:, k]
    xs = coords[0, :]
    ys = coords[1, :]
    out = bk * (bi * (aj + bj * xs + cj * ys) - bj * (ai + bi * xs + ci * ys)) + ck * (
        ci * (aj + bj * xs + cj * ys) - cj * (ai + bi * xs + ci * ys)
    )
    return out.astype(np.complex128)
