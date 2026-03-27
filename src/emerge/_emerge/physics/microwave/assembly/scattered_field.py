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

from __future__ import annotations
import numpy as np
from ....elements import Nedelec2
from numba import njit, f8, c16, i8, types, prange, void
from ....mth.optimized import compute_distances, local_mapping, matinv, generate_int_points_tet, gaus_quad_tet
from ....mth.csr_cast import CSRMapping
from typing import Callable


@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8, i8), cache=True, nogil=True)
def local_tet_to_triid(tet_to_field, tets, tris, itet, nedges) -> np.ndarray:
    """Returns the triangle node indices in the right order given a tet-index"""
    tri_ids = tet_to_field[6:10, itet] - nedges
    global_tri_map = tris[:, tri_ids]
    return local_mapping(tets[:, itet], global_tri_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tet_to_edgeid(tets, edges, tet_to_field, itet) -> np.ndarray:
    """Returns the edge node indices in the right order given a tet-index"""
    global_edge_map = edges[:, tet_to_field[:6,itet]]
    return local_mapping(tets[:, itet], global_edge_map)

@njit(i8[:,:](i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def local_tri_to_edgeid(tris, edges, tri_to_field, itri: int) -> np.ndarray:
    """Returns the edge node indices in the right order given a triangle-index"""
    global_edge_map = edges[:, tri_to_field[:3,itri]]
    return local_mapping(tris[:, itri], global_edge_map)


njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients_bcd_opt(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes the a,b,c and d coefficients of a tet barycentric coordinate functions and the volume

    Args:
        xs (np.ndarray): The tetrahedron X-coordinates
        ys (np.ndarray): The tetrahedron Y-coordinates
        zs (np.ndrray): The tetrahedron Z-coordinates

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: The a, b, c, d coefficients and volume
    """
    

    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    x1y2 = x1*y2
    x1y4 = x1*y4
    x2y1 = x2*y1
    x1y3 = x1*y3 
    x2y3 = x2*y3
    x3y1 = x3*y1
    x2y4 = x2*y4
    x3y4 = x3*y4
    x4y3 = x4*y3
    x4y1 = x4*y1
    x4y2 = x4*y2
    x3y2 = x3*y2

    P1 = z1*(-x2y3 + x2y4 +x3y2 - x3y4 - x4y2 + x4y3)
    P2 = z2*(x1y3 - x1y4 - x3y1 + x3y4 + x4y1 - x4y3)
    P3 = z3*(-x1y2 + x1y4 + x2y1 - x2y4 - x4y1 + x4y2)
    P4 = z4*(x1y2 - x1y3 - x2y1  + x2y3  + x3y1 - x3y2)
    V = np.abs(P1+P2+P3+P4)/6

    # X differences
    dx13 = x1 - x3
    dx14 = x1 - x4
    dx23 = x2 - x3
    dx24 = x2 - x4
    dx34 = x3 - x4

    # Y differences
    dy13 = y1 - y3
    dy14 = y1 - y4
    dy23 = y2 - y3
    dy24 = y2 - y4
    dy34 = y3 - y4

    # Z differences
    dz13 = z1 - z3
    dz14 = z1 - z4
    dz23 = z2 - z3
    dz24 = z2 - z4
    dz34 = z3 - z4

    aas[0] = x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2 
    aas[1] = -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1 
    aas[2] = x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1 
    aas[3] = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1 

    bbs[0] = -dy24 * dz34 + dy34 * dz24
    bbs[1] =  dy14 * dz34 - dy34 * dz14
    bbs[2] = -dy14 * dz24 + dy24 * dz14
    bbs[3] =  dy13 * dz23 - dy23 * dz13

    ccs[0] =  dx24 * dz34 - dx34 * dz24
    ccs[1] = -dx14 * dz34 + dx34 * dz14
    ccs[2] =  dx14 * dz24 - dx24 * dz14
    ccs[3] = -dx13 * dz23 + dx23 * dz13

    dds[0] = -dx24 * dy34 + dx34 * dy24
    dds[1] =  dx14 * dy34 - dx34 * dy14
    dds[2] = -dx14 * dy24 + dx24 * dy14
    dds[3] =  dx13 * dy23 - dx23 * dy13

    return bbs, ccs, dds, V


@njit(c16[:,:](c16[:,:], c16[:,:]), cache=True, nogil=True)
def matmul(a, b):
    out = np.empty((3,b.shape[1]), dtype=np.complex128)
    out[0,:] = a[0,0]*b[0,:] + a[0,1]*b[1,:] + a[0,2]*b[2,:]
    out[1,:] = a[1,0]*b[0,:] + a[1,1]*b[1,:] + a[1,2]*b[2,:]
    out[2,:] = a[2,0]*b[0,:] + a[2,1]*b[1,:] + a[2,2]*b[2,:]
    return out

@njit(c16[:](c16[:,:], c16[:,:]), cache=True, nogil=True)
def matdot(a, b):
    return a[0,:]*b[0,:] + a[1,:]*b[1,:] + a[2,:]*b[2,:]


# Coefficients are 4x4 matrix (abcd x i1, i2, i3, i4)
njit(void(c16[:,:], f8[:,:], f8[:], f8[:], f8[:]), cache=True, nogil=True)
def bary_inplace(bary: np.ndarray, coeff: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    bary[0,:] = coeff[0,0] + coeff[1,0]*x + coeff[2,0]*y + coeff[3,0]*z
    bary[1,:] = coeff[0,1] + coeff[1,1]*x + coeff[2,1]*y + coeff[3,1]*z
    bary[2,:] = coeff[0,2] + coeff[1,2]*x + coeff[2,2]*y + coeff[3,2]*z
    bary[3,:] = coeff[0,3] + coeff[1,3]*x + coeff[2,3]*y + coeff[3,3]*z

def func_edge(Lij, coef1, coef2, V, lami, lamj):
    ai, bi, ci, di = coef1
    aj, bj, cj, dj = coef2
    
    Fx = Lij*(bi*lamj - bj*lami)/(6*V)
    Fy = Lij*(ci*lamj - cj*lami)/(6*V)
    Fz = Lij*(di*lamj - dj*lami)/(6*V)
    
    N1x = Fx*lami
    N1y = Fy*lami
    N1z = Fz*lami

    N2x = Fx*lamj
    N2y = Fy*lamj
    N2z = Fz*lamj

    N1 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    N2 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    N1[0,:] = N1x
    N1[1,:] = N1y
    N1[2,:] = N1z

    N2[0,:] = N2x
    N2[1,:] = N2y
    N2[2,:] = N2z

    return N1, N2

def func_face(Lij, Lik, coef1, coef2, coef3, V, lami, lamj, lamk):
    ai, bi, ci, di = coef1
    aj, bj, cj, dj = coef2
    ak, bk, ck, dk = coef3
    
    N1x = Lik*lamj*(bk*lami - bi*lamk)/(6*V)
    N1y = Lik*lamj*(ck*lami - ci*lamk)/(6*V)
    N1z = Lik*lamj*(dk*lami - di*lamk)/(6*V)

    N2x = Lij*lamk*(bi*lamj - bj*lami)/(6*V)
    N2y = Lij*lamk*(ci*lamj - cj*lami)/(6*V)
    N2z = Lij*lamk*(di*lamj - dj*lami)/(6*V)

    N1 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    N2 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    N1[0,:] = N1x
    N1[1,:] = N1y
    N1[2,:] = N1z

    N2[0,:] = N2x
    N2[1,:] = N2y
    N2[2,:] = N2z

    return N1, N2

#@njit(cache=True)
def curl_edge(Lij, coef1, coef2, V, lam1, lam2):
    ai, bi, ci, di = coef1
    aj, bj, cj, dj = coef2
    Fx = Lij*(cj*di - ci*dj)/(12*V**2)
    Fy = Lij*(bi*dj - bj*di)/(12*V**2)
    Fz = Lij*(bj*ci - bi*cj)/(12*V**2)
    
    C1x = Fx*lam1
    C1y = Fy*lam1
    C1z = Fz*lam1

    C2x = Fx*lam2
    C2y = Fy*lam2
    C2z = Fz*lam2

    Curl1 = np.empty((3, lam1.shape[0]), dtype=np.complex128)
    Curl2 = np.empty((3, lam1.shape[0]), dtype=np.complex128)
    Curl1[0,:] = C1x
    Curl1[1,:] = C1y
    Curl1[2,:] = C1z

    Curl2[0,:] = C2x
    Curl2[1,:] = C2y
    Curl2[2,:] = C2z

    return Curl1, Curl2

def curl_face(Lij,Lik, coef1, coef2, coef3, V, lami, lamj, lamk):
    ai, bi, ci, di = coef1
    aj, bj, cj, dj = coef2
    ak, bk, ck, dk = coef3

    F1 = Lik / (36*V**2)
    F2 = Lij / (36*V**2)

    C1x = F1*(dj*(ci*lamk - ck*lami) - cj*(di*lamk - dk*lami) + 2*lamj*(ci*dk- ck*di))
    C1y = F1*(bj*(di*lamk - dk*lami) - dj*(bi*lamk - bk*lami) - 2*lamj*(bi*dk - bk*di))
    C1z = F1*(cj*(bi*lamk - bk*lamk) - bj*(ci*lamk - ck*lami) + 2*lamj*(bi*ck - bk*ci))

    C2x = F2*(ck*(di*lamj - dj*lami) - dk*(ci*lamj - cj*lami) - 2*lamk*(ci*dj - cj*di))
    C2y = F2*(dk*(bi*lamj - bj*lami) - bk*(di*lamj - dj*lami) + 2*lamk*(bi*dj - bj*di))
    C2z = F2*(bk*(ci*lamj - cj*lami) - ck*(bi*lamj - bj*lami) - 2*lamk*(bi*cj - bj*ci))

    Curl1 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    Curl2 = np.empty((3, lami.shape[0]), dtype=np.complex128)
    Curl1[0,:] = C1x
    Curl1[1,:] = C1y
    Curl1[2,:] = C1z

    Curl2[0,:] = C2x
    Curl2[1,:] = C2y
    Curl2[2,:] = C2z

    return Curl1, Curl2


@njit(cache=True)
def assemble_force(tet_vertices, Einc, CurlEinc, DPTs, k0Epsr, MuInv, local_edge_map, local_tri_map ):


    bout = np.zeros((20,), dtype=np.complex128)

    xs = tet_vertices[0,:]
    ys = tet_vertices[1,:]
    zs = tet_vertices[2,:]

    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    edge_lengths = compute_distances(xs, ys, zs)

    aas, bbs, ccs, dds, V = tet_coefficients_bcd_opt(xs, ys, zs)

    coeffs = np.empty((4,4), dtype=np.float64)
    coeffs[0,:] = aas
    coeffs[1,:] = bbs
    coeffs[2,:] = ccs
    coeffs[3,:] = dds

    Ws = DPTs[0,:]
    xf = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:] + z3*DPTs[4,:]
    yf = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:] + z3*DPTs[4,:]
    zf = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:] + z3*DPTs[4,:]

    nf = Ws.shape[0]

    Lb = np.empty((4,nf), dtype=np.complex128)

    bary_inplace(Lb, coeffs, xf, yf, zf)

    # Lb contains the barycentrid functions.

    for ei in range(6):
        ei1 = local_edge_map[0, ei]
        ei2 = local_edge_map[1, ei]
        L1 = edge_lengths[ei1, ei2]

        N1, N2 = func_edge(L1, coeffs[:,ei1], coeffs[:,ei2], V, Lb[ei1,:], Lb[ei2,:])
        CurlE1, CurlE2 = curl_edge(L1, coeffs[:,ei1], coeffs[:,ei2], V, Lb[ei1,:], Lb[ei2,:])

        bout[ei] = matdot(matmul(MuInv, CurlE1), CurlEinc) - matdot(matmul(k0Epsr, N1), Einc)
        bout[ei+10] = matdot(matmul(MuInv, CurlE2), CurlEinc) - matdot(matmul(k0Epsr, N2), Einc)

    
    for fi in range(6):
        v1, v2, v3 = local_tri_map[:, fi]

        L1 = edge_lengths[v1, v2]
        L2 = edge_lengths[v1, v3]

        N1, N2 = func_face(L1,L2, coeffs[:,v1], coeffs[:,v2], coeffs[:,v3], V, Lb[v1,:], Lb[v2,:], Lb[v3,:])

        CurlE1, CurlE2 = curl_face(L1,L2, coeffs[:,v1], coeffs[:,v2], coeffs[:,v3], V, Lb[v1,:], Lb[v2,:], Lb[v3,:])

        bout[ei+6] = matdot(matmul(MuInv, CurlE1), CurlEinc) - matdot(matmul(k0Epsr, N1), Einc)
        bout[ei+16] = matdot(matmul(MuInv, CurlE2), CurlEinc) - matdot(matmul(k0Epsr, N2), Einc)

    return bout


# @njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:], 
#                                                 i8[:,:], 
#                                                 i8[:,:], 
#                                                 i8[:,:], 
#                                                 f8[:], 
#                                                 i8[:,:], 
#                                                 i8[:,:], 
#                                                 c16[:,:,:], 
#                                                 c16[:,:,:]), cache=True, nogil=True, parallel=True)
def _vector_builder(data_out, nodes, tets, tris, edges, tet_to_field, Einc, CurlEinc,DPTs, ur, er, nedges):
    ntets = tets.shape[1]
    for itet in range(ntets):

        urt = ur[:,:,itet]
        ert = er[:,:,itet]

        # Construct a local mapping to global triangle orientations
        
        local_tri_map = local_tet_to_triid(tet_to_field, tets, tris, itet, nedges)
        local_edge_map = local_tet_to_edgeid(tets, edges, tet_to_field, itet)

        bsub = assemble_force(nodes[:,tets[:,itet]],
                            Einc,
                            CurlEinc,
                            DPTs,
                            ert,
                            matinv(urt),
                            local_edge_map, 
                            local_tri_map, 
                            )
        
        indices = tet_to_field[:, itet]
        data_out[indices] = bsub
    return data_out


def scattered_tet_force(field: Nedelec2,
                        er: np.ndarray, 
                        ur: np.ndarray, 
                        func: Callable,
                        curl_func: Callable,
                        csrmap: CSRMapping | None = None) -> tuple[np.ndarray, np.ndarray, CSRMapping]:
    """Computes the curl-curl Nedelec-2 mass and stiffness matrices

    Args:
        field (Nedelec2): The Nedelec2 Field object
        er (np.ndarray): a 3x3xN array with permittivity tensors
        ur (np.ndarray): a 3x3xN array with permeability tensors

    Returns:
        tuple[csr_matrix, csr_matrix]: The stiffness and mass matrix.
    """
    tets = field.mesh.tets
    tris = field.mesh.tris
    edges = field.mesh.edges
    nodes = field.mesh.nodes

    tet_to_field = field.tet_to_field
    data_out = np.zeros((field.nfield,), dtype=np.complex128)
    
    DPTs = gaus_quad_tet(4)

    xflat, yflat, zflat = generate_int_points_tet(nodes, tets, DPTs)
    
    Einc = func(xflat, yflat, zflat)
    CurlEinc = curl_func(xflat, yflat, zflat)

    data_out = _vector_builder(data_out, nodes, tets, tris, edges, tet_to_field, Einc, CurlEinc, DPTs, ur, er, field.mesh.n_edges)
    return data_out

