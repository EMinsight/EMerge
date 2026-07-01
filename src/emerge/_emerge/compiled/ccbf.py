
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

from numba import njit, c16, f8, i8, types
import numpy as np
from numba.typed import List 
# This will be an overview of the complete order basis functions
# A binary number will be used to represent each basis function. There wil be 4 codes(bits) for the type
# 000xxxxx = Nodal basis function
# 001xxxxx = Edge bassis function
# 010xxxxx = Face basis function
# 100xxxxx = Volume basis function

# This leaves 5 bits for the basis function number (16 total should be enough.)
NODE_TYPE = 0b00000000
EDGE_TYPE = 0b01000000
FACE_TYPE = 0b10000000
VOLU_TYPE = 0b11000000

MASK_TYPE = 0b11000000
MASK_INDEX = 0b00111111

@njit(types.Tuple((i8[:], i8[:], i8[:]))(i8[:]), cache=True)
def parse_dofcode(dofcodes: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    typearray = np.empty_like(dofcodes, dtype=np.int64)
    indexarray = np.empty_like(dofcodes, dtype=np.int64)
    idofarray = np.empty_like(dofcodes, dtype=np.int64)
    i = 0
    ne = np.zeros((2**6,), dtype=np.int64)
    nf = np.zeros((2**6,), dtype=np.int64)
    for code in dofcodes:
        idofcode = code & 0b00111111
        if code & 0b11000000==64:
            typearray[i] = 0
            indexarray[i] = ne[idofcode]
            ne[idofcode] += 1
        else:
            typearray[i] = 1
            indexarray[i] = nf[idofcode]
            nf[idofcode] += 1
        idofarray[i] = idofcode
        i += 1
    return typearray, indexarray, idofarray
        

@njit(cache=True)
def get_type_index(number: int):
    index = number & MASK_INDEX
    bftype = number & MASK_TYPE
    return bftype, index

# for i in range(256):
#     print(get_type_index(i))
@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne0_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = ai*bj - aj*bi - bi*cj*ys + bj*ci*ys
    by = ai*cj - aj*ci + bi*cj*xs - bj*ci*xs
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne1_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = bi*(aj + bj*xs + cj*ys) + bj*(ai + bi*xs + ci*ys)
    by = ci*(aj + bj*xs + cj*ys) + cj*(ai + bi*xs + ci*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = -(bi*(aj + bj*xs + cj*ys) - bj*(ai + bi*xs + ci*ys))*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys)
    by = -(ci*(aj + bj*xs + cj*ys) - cj*(ai + bi*xs + ci*ys))*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne3_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = bi*(aj + bj*xs + cj*ys)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + bj*(ai + bi*xs + ci*ys)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + (bi - bj)*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    by = ci*(aj + bj*xs + cj*ys)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + cj*(ai + bi*xs + ci*ys)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + (ci - cj)*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne0_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    out = 2*bi*cj - 2*bj*ci*np.ones_like(coords[0,:])
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne1_2d(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -(bi - bj)*(ci*(aj + bj*xs + cj*ys) - cj*(ai + bi*xs + ci*ys)) + (ci - cj)*(bi*(aj + bj*xs + cj*ys) - bj*(ai + bi*xs + ci*ys)) + 2*(bi*cj - bj*ci)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne3_2d(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne0_2d(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne1_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    out = 2*bi*bj + 2*ci*cj*np.ones_like(coords[0,:])
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -(bi - bj)*(bi*(aj + bj*xs + cj*ys) - bj*(ai + bi*xs + ci*ys)) - (ci - cj)*(ci*(aj + bj*xs + cj*ys) - cj*(ai + bi*xs + ci*ys))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne3_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    out = 2*bi*bj*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + 2*bi*(bi - bj)*(aj + bj*xs + cj*ys) + 2*bj*(bi - bj)*(ai + bi*xs + ci*ys) + 2*ci*cj*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys) + 2*ci*(ci - cj)*(aj + bj*xs + cj*ys) + 2*cj*(ci - cj)*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf0_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = -(bj*(ak + bk*xs + ck*ys) - bk*(aj + bj*xs + cj*ys))*(ai + bi*xs + ci*ys)
    by = -(cj*(ak + bk*xs + ck*ys) - ck*(aj + bj*xs + cj*ys))*(ai + bi*xs + ci*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf1_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = -(bi*(ak + bk*xs + ck*ys) - bk*(ai + bi*xs + ci*ys))*(aj + bj*xs + cj*ys)
    by = -(ci*(ak + bk*xs + ck*ys) - ck*(ai + bi*xs + ci*ys))*(aj + bj*xs + cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = bi*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + bj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) - 2*bk*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    by = ci*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + cj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) - 2*ck*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf3_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = -2*bi*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + bj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) + bk*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    by = -2*ci*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + cj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) + ck*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf4_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    bx = bi*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + bj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) + bk*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    by = ci*(aj + bj*xs + cj*ys)*(ak + bk*xs + ck*ys) + cj*(ai + bi*xs + ci*ys)*(ak + bk*xs + ck*ys) + ck*(ai + bi*xs + ci*ys)*(aj + bj*xs + cj*ys)
    out = np.empty((2, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf0_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -bi*(cj*(ak + bk*xs + ck*ys) - ck*(aj + bj*xs + cj*ys)) + ci*(bj*(ak + bk*xs + ck*ys) - bk*(aj + bj*xs + cj*ys)) + 2*(bj*ck - bk*cj)*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf1_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -bj*(ci*(ak + bk*xs + ck*ys) - ck*(ai + bi*xs + ci*ys)) + cj*(bi*(ak + bk*xs + ck*ys) - bk*(ai + bi*xs + ci*ys)) + 2*(bi*ck - bk*ci)*(aj + bj*xs + cj*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -3*bi*ck*(aj + bj*xs + cj*ys) - 3*bj*ck*(ai + bi*xs + ci*ys) + 3*bk*ci*(aj + bj*xs + cj*ys) + 3*bk*cj*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf3_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = 3*bi*cj*(ak + bk*xs + ck*ys) + 3*bi*ck*(aj + bj*xs + cj*ys) - 3*bj*ci*(ak + bk*xs + ck*ys) - 3*bk*ci*(aj + bj*xs + cj*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf4_2d(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf0_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -bi*(bj*(ak + bk*xs + ck*ys) - bk*(aj + bj*xs + cj*ys)) - ci*(cj*(ak + bk*xs + ck*ys) - ck*(aj + bj*xs + cj*ys))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf1_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -bj*(bi*(ak + bk*xs + ck*ys) - bk*(ai + bi*xs + ci*ys)) - cj*(ci*(ak + bk*xs + ck*ys) - ck*(ai + bi*xs + ci*ys))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf2_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = 2*bi*bj*(ak + bk*xs + ck*ys) - bi*bk*(aj + bj*xs + cj*ys) - bj*bk*(ai + bi*xs + ci*ys) + 2*ci*cj*(ak + bk*xs + ck*ys) - ci*ck*(aj + bj*xs + cj*ys) - cj*ck*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf3_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -bi*bj*(ak + bk*xs + ck*ys) - bi*bk*(aj + bj*xs + cj*ys) + 2*bj*bk*(ai + bi*xs + ci*ys) - ci*cj*(ak + bk*xs + ck*ys) - ci*ck*(aj + bj*xs + cj*ys) + 2*cj*ck*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf4_2d(coeff, coords, i, j, k):
    ai, bi, ci = coeff[:,i]
    aj, bj, cj = coeff[:,j]
    ak, bk, ck = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    out = 2*bi*bj*(ak + bk*xs + ck*ys) + 2*bi*bk*(aj + bj*xs + cj*ys) + 2*bj*bk*(ai + bi*xs + ci*ys) + 2*ci*cj*(ak + bk*xs + ck*ys) + 2*ci*ck*(aj + bj*xs + cj*ys) + 2*cj*ck*(ai + bi*xs + ci*ys)
    return out.astype(np.complex128)

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne0_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -bi*(aj + bj*xs + cj*ys + dj*zs) + bj*(ai + bi*xs + ci*ys + di*zs)
    by = -ci*(aj + bj*xs + cj*ys + dj*zs) + cj*(ai + bi*xs + ci*ys + di*zs)
    bz = -di*(aj + bj*xs + cj*ys + dj*zs) + dj*(ai + bi*xs + ci*ys + di*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne1_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = bi*(aj + bj*xs + cj*ys + dj*zs) + bj*(ai + bi*xs + ci*ys + di*zs)
    by = ci*(aj + bj*xs + cj*ys + dj*zs) + cj*(ai + bi*xs + ci*ys + di*zs)
    bz = di*(aj + bj*xs + cj*ys + dj*zs) + dj*(ai + bi*xs + ci*ys + di*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -(bi*(aj + bj*xs + cj*ys + dj*zs) - bj*(ai + bi*xs + ci*ys + di*zs))*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    by = -(ci*(aj + bj*xs + cj*ys + dj*zs) - cj*(ai + bi*xs + ci*ys + di*zs))*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    bz = -(di*(aj + bj*xs + cj*ys + dj*zs) - dj*(ai + bi*xs + ci*ys + di*zs))*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _ne3_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = bi*(aj + bj*xs + cj*ys + dj*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + bj*(ai + bi*xs + ci*ys + di*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + (bi - bj)*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    by = ci*(aj + bj*xs + cj*ys + dj*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + cj*(ai + bi*xs + ci*ys + di*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + (ci - cj)*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    bz = di*(aj + bj*xs + cj*ys + dj*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + dj*(ai + bi*xs + ci*ys + di*zs)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + (di - dj)*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne0_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    bx = 2*ci*dj - 2*cj*di
    by = -2*bi*dj + 2*bj*di
    bz = 2*bi*cj - 2*bj*ci
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne1_3d(coeff, coords, i, j, k):
    out = np.zeros((3, coords.shape[1]), dtype=np.complex128)
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -(ci - cj)*(di*(aj + bj*xs + cj*ys + dj*zs) - dj*(ai + bi*xs + ci*ys + di*zs)) + (di - dj)*(ci*(aj + bj*xs + cj*ys + dj*zs) - cj*(ai + bi*xs + ci*ys + di*zs)) + 2*(ci*dj - cj*di)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    by = (bi - bj)*(di*(aj + bj*xs + cj*ys + dj*zs) - dj*(ai + bi*xs + ci*ys + di*zs)) - (di - dj)*(bi*(aj + bj*xs + cj*ys + dj*zs) - bj*(ai + bi*xs + ci*ys + di*zs)) - 2*(bi*dj - bj*di)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    bz = -(bi - bj)*(ci*(aj + bj*xs + cj*ys + dj*zs) - cj*(ai + bi*xs + ci*ys + di*zs)) + (ci - cj)*(bi*(aj + bj*xs + cj*ys + dj*zs) - bj*(ai + bi*xs + ci*ys + di*zs)) + 2*(bi*cj - bj*ci)*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_ne3_3d(coeff, coords, i, j, k):
    out = np.zeros((3, coords.shape[1]), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne0_3d(coeff, coords, i, j, k):
    out = np.zeros((coords.shape[1],), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne1_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    out = 2*bi*bj + 2*ci*cj + 2*di*dj*np.ones_like(coords[0,:])
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = -(bi - bj)*(bi*(aj + bj*xs + cj*ys + dj*zs) - bj*(ai + bi*xs + ci*ys + di*zs)) - (ci - cj)*(ci*(aj + bj*xs + cj*ys + dj*zs) - cj*(ai + bi*xs + ci*ys + di*zs)) - (di - dj)*(di*(aj + bj*xs + cj*ys + dj*zs) - dj*(ai + bi*xs + ci*ys + di*zs))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_ne3_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = 2*bi*bj*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + 2*bi*(bi - bj)*(aj + bj*xs + cj*ys + dj*zs) + 2*bj*(bi - bj)*(ai + bi*xs + ci*ys + di*zs) + 2*ci*cj*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + 2*ci*(ci - cj)*(aj + bj*xs + cj*ys + dj*zs) + 2*cj*(ci - cj)*(ai + bi*xs + ci*ys + di*zs) + 2*di*dj*(ai - aj + bi*xs - bj*xs + ci*ys - cj*ys + di*zs - dj*zs) + 2*di*(di - dj)*(aj + bj*xs + cj*ys + dj*zs) + 2*dj*(di - dj)*(ai + bi*xs + ci*ys + di*zs)
    return out.astype(np.complex128)

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf0_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -(bj*(ak + bk*xs + ck*ys + dk*zs) - bk*(aj + bj*xs + cj*ys + dj*zs))*(ai + bi*xs + ci*ys + di*zs)
    by = -(cj*(ak + bk*xs + ck*ys + dk*zs) - ck*(aj + bj*xs + cj*ys + dj*zs))*(ai + bi*xs + ci*ys + di*zs)
    bz = -(dj*(ak + bk*xs + ck*ys + dk*zs) - dk*(aj + bj*xs + cj*ys + dj*zs))*(ai + bi*xs + ci*ys + di*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf1_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -(bi*(ak + bk*xs + ck*ys + dk*zs) - bk*(ai + bi*xs + ci*ys + di*zs))*(aj + bj*xs + cj*ys + dj*zs)
    by = -(ci*(ak + bk*xs + ck*ys + dk*zs) - ck*(ai + bi*xs + ci*ys + di*zs))*(aj + bj*xs + cj*ys + dj*zs)
    bz = -(di*(ak + bk*xs + ck*ys + dk*zs) - dk*(ai + bi*xs + ci*ys + di*zs))*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = bi*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + bj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) - 2*bk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    by = ci*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + cj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) - 2*ck*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    bz = di*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + dj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) - 2*dk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf3_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -2*bi*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + bj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + bk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    by = -2*ci*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + cj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + ck*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    bz = -2*di*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + dj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + dk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _nf4_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = bi*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + bj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + bk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    by = ci*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + cj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + ck*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    bz = di*(aj + bj*xs + cj*ys + dj*zs)*(ak + bk*xs + ck*ys + dk*zs) + dj*(ai + bi*xs + ci*ys + di*zs)*(ak + bk*xs + ck*ys + dk*zs) + dk*(ai + bi*xs + ci*ys + di*zs)*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf0_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -ci*(dj*(ak + bk*xs + ck*ys + dk*zs) - dk*(aj + bj*xs + cj*ys + dj*zs)) + di*(cj*(ak + bk*xs + ck*ys + dk*zs) - ck*(aj + bj*xs + cj*ys + dj*zs)) + 2*(cj*dk - ck*dj)*(ai + bi*xs + ci*ys + di*zs)
    by = bi*(dj*(ak + bk*xs + ck*ys + dk*zs) - dk*(aj + bj*xs + cj*ys + dj*zs)) - di*(bj*(ak + bk*xs + ck*ys + dk*zs) - bk*(aj + bj*xs + cj*ys + dj*zs)) - 2*(bj*dk - bk*dj)*(ai + bi*xs + ci*ys + di*zs)
    bz = -bi*(cj*(ak + bk*xs + ck*ys + dk*zs) - ck*(aj + bj*xs + cj*ys + dj*zs)) + ci*(bj*(ak + bk*xs + ck*ys + dk*zs) - bk*(aj + bj*xs + cj*ys + dj*zs)) + 2*(bj*ck - bk*cj)*(ai + bi*xs + ci*ys + di*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf1_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -cj*(di*(ak + bk*xs + ck*ys + dk*zs) - dk*(ai + bi*xs + ci*ys + di*zs)) + dj*(ci*(ak + bk*xs + ck*ys + dk*zs) - ck*(ai + bi*xs + ci*ys + di*zs)) + 2*(ci*dk - ck*di)*(aj + bj*xs + cj*ys + dj*zs)
    by = bj*(di*(ak + bk*xs + ck*ys + dk*zs) - dk*(ai + bi*xs + ci*ys + di*zs)) - dj*(bi*(ak + bk*xs + ck*ys + dk*zs) - bk*(ai + bi*xs + ci*ys + di*zs)) - 2*(bi*dk - bk*di)*(aj + bj*xs + cj*ys + dj*zs)
    bz = -bj*(ci*(ak + bk*xs + ck*ys + dk*zs) - ck*(ai + bi*xs + ci*ys + di*zs)) + cj*(bi*(ak + bk*xs + ck*ys + dk*zs) - bk*(ai + bi*xs + ci*ys + di*zs)) + 2*(bi*ck - bk*ci)*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = -3*ci*dk*(aj + bj*xs + cj*ys + dj*zs) - 3*cj*dk*(ai + bi*xs + ci*ys + di*zs) + 3*ck*di*(aj + bj*xs + cj*ys + dj*zs) + 3*ck*dj*(ai + bi*xs + ci*ys + di*zs)
    by = 3*bi*dk*(aj + bj*xs + cj*ys + dj*zs) + 3*bj*dk*(ai + bi*xs + ci*ys + di*zs) - 3*bk*di*(aj + bj*xs + cj*ys + dj*zs) - 3*bk*dj*(ai + bi*xs + ci*ys + di*zs)
    bz = -3*bi*ck*(aj + bj*xs + cj*ys + dj*zs) - 3*bj*ck*(ai + bi*xs + ci*ys + di*zs) + 3*bk*ci*(aj + bj*xs + cj*ys + dj*zs) + 3*bk*cj*(ai + bi*xs + ci*ys + di*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf3_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    bx = 3*ci*dj*(ak + bk*xs + ck*ys + dk*zs) + 3*ci*dk*(aj + bj*xs + cj*ys + dj*zs) - 3*cj*di*(ak + bk*xs + ck*ys + dk*zs) - 3*ck*di*(aj + bj*xs + cj*ys + dj*zs)
    by = -3*bi*dj*(ak + bk*xs + ck*ys + dk*zs) - 3*bi*dk*(aj + bj*xs + cj*ys + dj*zs) + 3*bj*di*(ak + bk*xs + ck*ys + dk*zs) + 3*bk*di*(aj + bj*xs + cj*ys + dj*zs)
    bz = 3*bi*cj*(ak + bk*xs + ck*ys + dk*zs) + 3*bi*ck*(aj + bj*xs + cj*ys + dj*zs) - 3*bj*ci*(ak + bk*xs + ck*ys + dk*zs) - 3*bk*ci*(aj + bj*xs + cj*ys + dj*zs)
    out = np.empty((3, coords.shape[1]), dtype=np.complex128)
    out[0,:] = bx
    out[1,:] = by
    out[2,:] = bz
    return out

@njit(c16[:,:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _curl_nf4_3d(coeff, coords, i, j, k):
    out = np.zeros((3, coords.shape[1]), dtype=np.complex128)
    return out

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf0_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = -bi*(bj*(ak + bk*xs + ck*ys + dk*zs) - bk*(aj + bj*xs + cj*ys + dj*zs)) - ci*(cj*(ak + bk*xs + ck*ys + dk*zs) - ck*(aj + bj*xs + cj*ys + dj*zs)) - di*(dj*(ak + bk*xs + ck*ys + dk*zs) - dk*(aj + bj*xs + cj*ys + dj*zs))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf1_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = -bj*(bi*(ak + bk*xs + ck*ys + dk*zs) - bk*(ai + bi*xs + ci*ys + di*zs)) - cj*(ci*(ak + bk*xs + ck*ys + dk*zs) - ck*(ai + bi*xs + ci*ys + di*zs)) - dj*(di*(ak + bk*xs + ck*ys + dk*zs) - dk*(ai + bi*xs + ci*ys + di*zs))
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf2_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = 2*bi*bj*(ak + bk*xs + ck*ys + dk*zs) - bi*bk*(aj + bj*xs + cj*ys + dj*zs) - bj*bk*(ai + bi*xs + ci*ys + di*zs) + 2*ci*cj*(ak + bk*xs + ck*ys + dk*zs) - ci*ck*(aj + bj*xs + cj*ys + dj*zs) - cj*ck*(ai + bi*xs + ci*ys + di*zs) + 2*di*dj*(ak + bk*xs + ck*ys + dk*zs) - di*dk*(aj + bj*xs + cj*ys + dj*zs) - dj*dk*(ai + bi*xs + ci*ys + di*zs)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf3_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = -bi*bj*(ak + bk*xs + ck*ys + dk*zs) - bi*bk*(aj + bj*xs + cj*ys + dj*zs) + 2*bj*bk*(ai + bi*xs + ci*ys + di*zs) - ci*cj*(ak + bk*xs + ck*ys + dk*zs) - ci*ck*(aj + bj*xs + cj*ys + dj*zs) + 2*cj*ck*(ai + bi*xs + ci*ys + di*zs) - di*dj*(ak + bk*xs + ck*ys + dk*zs) - di*dk*(aj + bj*xs + cj*ys + dj*zs) + 2*dj*dk*(ai + bi*xs + ci*ys + di*zs)
    return out.astype(np.complex128)

@njit(c16[:](f8[:,:], f8[:,:], i8, i8, i8), cache=True, nogil=True)
def _div_nf4_3d(coeff, coords, i, j, k):
    ai, bi, ci, di = coeff[:,i]
    aj, bj, cj, dj = coeff[:,j]
    ak, bk, ck, dk = coeff[:,k]
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    out = 2*bi*bj*(ak + bk*xs + ck*ys + dk*zs) + 2*bi*bk*(aj + bj*xs + cj*ys + dj*zs) + 2*bj*bk*(ai + bi*xs + ci*ys + di*zs) + 2*ci*cj*(ak + bk*xs + ck*ys + dk*zs) + 2*ci*ck*(aj + bj*xs + cj*ys + dj*zs) + 2*cj*ck*(ai + bi*xs + ci*ys + di*zs) + 2*di*dj*(ak + bk*xs + ck*ys + dk*zs) + 2*di*dk*(aj + bj*xs + cj*ys + dj*zs) + 2*dj*dk*(ai + bi*xs + ci*ys + di*zs)
    return out.astype(np.complex128)

@njit(cache=True)
def _eval_f_2d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)
    
    if bftype == 64:
        if index == 0:
            return _ne0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _ne1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _ne2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _ne3_2d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _nf0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _nf1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _nf2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _nf3_2d(coeff, coords, i,j,k) 
        elif index == 4:
            return _nf4_2d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords, dtype=np.complex128)

@njit(cache=True)
def _eval_f_3d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)
    if bftype == 64:
        if index == 0:
            return _ne0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _ne1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _ne2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _ne3_3d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _nf0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _nf1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _nf2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _nf3_3d(coeff, coords, i,j,k) 
        elif index == 4:
            return _nf4_3d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords, dtype=np.complex128)


@njit(cache=True)
def _eval_curl_f_2d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)
    if bftype == 64:
        if index == 0:
            return _curl_ne0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _curl_ne1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _curl_ne2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _curl_ne3_2d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _curl_nf0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _curl_nf1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _curl_nf2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _curl_nf3_2d(coeff, coords, i,j,k) 
        elif index == 4:
            return _curl_nf4_2d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords[0,:], dtype=np.complex128)


@njit(cache=True)
def _eval_curl_f_3d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            return _curl_ne0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _curl_ne1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _curl_ne2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _curl_ne3_3d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _curl_nf0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _curl_nf1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _curl_nf2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _curl_nf3_3d(coeff, coords, i,j,k) 
        elif index == 4:
            return _curl_nf4_3d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords, dtype=np.complex128)


@njit(cache=True)
def _eval_div_f_2d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            return _div_ne0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _div_ne1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _div_ne2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _div_ne3_2d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _div_nf0_2d(coeff, coords, i,j,k)
        elif index == 1:
            return _div_nf1_2d(coeff, coords, i,j,k) 
        elif index == 2:
            return _div_nf2_2d(coeff, coords, i,j,k) 
        elif index == 3:
            return _div_nf3_2d(coeff, coords, i,j,k) 
        elif index == 4:
            return _div_nf4_2d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords[0,:], dtype=np.complex128)


@njit(cache=True)
def _eval_div_f_3d(coeff, coords, i, j, k, code):
    bftype, index = get_type_index(code)

    if bftype == 64:
        if index == 0:
            return _div_ne0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _div_ne1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _div_ne2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _div_ne3_3d(coeff, coords, i,j,k) 
    if bftype == 128:
        if index == 0:
            return _div_nf0_3d(coeff, coords, i,j,k)
        elif index == 1:
            return _div_nf1_3d(coeff, coords, i,j,k) 
        elif index == 2:
            return _div_nf2_3d(coeff, coords, i,j,k) 
        elif index == 3:
            return _div_nf3_3d(coeff, coords, i,j,k) 
        elif index == 4:
            return _div_nf4_3d(coeff, coords, i,j,k) 
    raise ValueError('Unrecognized basis function type.')
    return np.zeros_like(coords[0,:], dtype=np.complex128)