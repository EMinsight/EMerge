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


import numpy as np
from ....elements import Nedelec2
from ....mth.optimized import local_mapping, compute_distances
from numba import c16, types, f8, i8, njit, prange


############################################################
#                      FIELD MAPPING                      #
############################################################

@njit(i8[:,:](i8, i8[:,:], i8[:,:], i8[:,:]), cache=True, nogil=True)
def local_tri_to_edgeid(itri: int, tris, edges, tri_to_edge) -> np.ndarray:
    global_edge_map = edges[:, tri_to_edge[:,itri]]
    return local_mapping(tris[:, itri], global_edge_map)


@njit(cache=True, fastmath=True, nogil=True)
def optim_matmul(B: np.ndarray, data: np.ndarray):
    dnew = np.zeros_like(data)
    dnew[0,:] = B[0,0]*data[0,:] + B[0,1]*data[1,:] + B[0,2]*data[2,:]
    dnew[1,:] = B[1,0]*data[0,:] + B[1,1]*data[1,:] + B[1,2]*data[2,:]
    dnew[2,:] = B[2,0]*data[0,:] + B[2,1]*data[1,:] + B[2,2]*data[2,:]
    return dnew

@njit(f8[:](f8[:], f8[:]), cache=True, fastmath=True, nogil=True)
def cross(a: np.ndarray, b: np.ndarray):
    crossv = np.empty((3,), dtype=np.float64)
    crossv[0] = a[1]*b[2] - a[2]*b[1]
    crossv[1] = a[2]*b[0] - a[0]*b[2]
    crossv[2] = a[0]*b[1] - a[1]*b[0]
    return crossv

@njit(cache=True, nogil=True)
def normalize(a: np.ndarray):
    return a/((a[0]**2 + a[1]**2 + a[2]**2)**0.5)


############################################################
#              GAUSS QUADRATURE IMPLEMENTATION             #
############################################################

@njit(c16(c16[:], c16[:], types.Array(types.float64, 1, 'A', readonly=True)), cache=True, nogil=True)
def _gqi(v1, v2, W):
    return np.sum(v1*v2*W)


############################################################
#                BASIS FUNCTION DERIVATIVES               #
############################################################


@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _curl_edge_1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -3*a1*b1*c2 + 3*a1*b2*c1 - 3*b1**2*c2*xs + 3*b1*b2*c1*xs - 3*b1*c1*c2*ys + 3*b2*c1**2*ys + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _curl_edge_2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -3*a2*b1*c2 + 3*a2*b2*c1 - 3*b1*b2*c2*xs - 3*b1*c2**2*ys + 3*b2**2*c1*xs + 3*b2*c1*c2*ys + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _curl_face_1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -b2*(c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys)) + c2*(b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys)) + 2*(b1*c3 - b3*c1)*(a2 + b2*xs + c2*ys) + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _curl_face_2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out = b3*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) - c3*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) - 2*(b1*c2 - b2*c1)*(a3 + b3*xs + c3*ys) + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _divergence_edge_1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out = b1*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) + c1*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _divergence_edge_2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    xs = coords[0,:]
    ys = coords[1,:]
    out = b2*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) + c2*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _divergence_face_1(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out = -b2*(b1*(a3 + b3*xs + c3*ys) - b3*(a1 + b1*xs + c1*ys)) - c2*(c1*(a3 + b3*xs + c3*ys) - c3*(a1 + b1*xs + c1*ys)) + 0j
    return out

@njit(c16[:](f8[:,:], f8[:,:]), cache=True)
def _divergence_face_2(coeff, coords):
    a1, b1, c1 = coeff[:,0]
    a2, b2, c2 = coeff[:,1]
    a3, b3, c3 = coeff[:,2]
    xs = coords[0,:]
    ys = coords[1,:]
    out = b3*(b1*(a2 + b2*xs + c2*ys) - b2*(a1 + b1*xs + c1*ys)) + c3*(c1*(a2 + b2*xs + c2*ys) - c2*(a1 + b1*xs + c1*ys)) + 0j
    return out

############################################################
#     TRIANGLE BARYCENTRIC COORDINATE LIN. COEFFICIENTS    #
############################################################


@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:]), cache = True, nogil=True)
def tri_coefficients(vxs, vys):

    x1, x2, x3 = vxs
    y1, y2, y3 = vys

    a1 = x2*y3-y2*x3
    a2 = x3*y1-y3*x1
    a3 = x1*y2-y1*x2
    b1 = y2-y3
    b2 = y3-y1
    b3 = y1-y2
    c1 = x3-x2
    c2 = x1-x3
    c3 = x2-x1

    sA = 0.5*(((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3])*sign
    Bs = np.array([b1, b2, b3])*sign
    Cs = np.array([c1, c2, c3])*sign
    return As, Bs, Cs, A


############################################################
#                  GAUSS QUADRATURE POINTS                 #
############################################################


DPTS = np.array([[0.22338159, 0.22338159, 0.22338159, 0.10995174, 0.10995174, 0.10995174],
                 [0.10810302, 0.44594849, 0.44594849, 0.81684757, 0.09157621, 0.09157621],
                 [0.44594849, 0.44594849, 0.10810302, 0.09157621, 0.09157621, 0.81684757],
                 [0.44594849, 0.10810302, 0.44594849, 0.09157621, 0.81684757, 0.09157621]], dtype=np.float64)


############################################################
#                 NUMBA OPTIMIZED ASSEMBLER                #
############################################################


@njit(c16[:,:](f8[:,:], i8[:,:], c16), cache=True, nogil=True)
def _abc_order_2_terms(tri_vertices, local_edge_map, cf):
    '''ABC order 2 tangent gradient term'''
    
    origin = tri_vertices[:,0]
    vertex_2 = tri_vertices[:,1]
    vertex_3 = tri_vertices[:,2]
    
    edge_1 = vertex_2-origin
    edge_2 = vertex_3-origin
    
    zhat = normalize(cross(edge_1, edge_2))
    xhat = normalize(edge_1)
    yhat = normalize(cross(zhat, xhat))
    
    basis = np.zeros((3,3), dtype=np.float64)
    basis[0,:] = xhat
    basis[1,:] = yhat
    basis[2,:] = zhat
    
    local_vertices = optim_matmul(basis, tri_vertices - origin[:,np.newaxis])
    
    CurlMatrix = np.zeros((8,8), dtype=np.complex128)
    DivMatrix = np.zeros((8,8), dtype=np.complex128)
    
    Lengths = np.ones((8,8), dtype=np.float64)

    WEIGHTS = DPTS[0,:]
    DPTS1 = DPTS[1,:]
    DPTS2 = DPTS[2,:]
    DPTS3 = DPTS[3,:]

    xpts = local_vertices[0,:]
    ypts = local_vertices[1,:]

    distances = compute_distances(xpts, ypts, 0*xpts)

    xs = xpts[0]*DPTS1 + xpts[1]*DPTS2 + xpts[2]*DPTS3
    ys = ypts[0]*DPTS1 + ypts[1]*DPTS2 + ypts[2]*DPTS3
    
    int_coords = np.empty((2,xs.shape[0]), dtype=np.float64)
    int_coords[0,:] = xs
    int_coords[1,:] = ys

    aas, bbs, ccs, Area = tri_coefficients(xpts, ypts)
    
    bary_coeff = np.empty((3,3), dtype=np.float64)
    bary_coeff[0,:] = aas/(2*Area)
    bary_coeff[1,:] = bbs/(2*Area)
    bary_coeff[2,:] = ccs/(2*Area)

    Lengths[3,:] *= distances[0,2]
    Lengths[7,:] *= distances[0,1]
    Lengths[:,3] *= distances[0,2]
    Lengths[:,7] *= distances[0,1]
    
    FF1C = _curl_face_1(bary_coeff, int_coords)
    FF2C = _curl_face_2(bary_coeff, int_coords)
    FF1D = _divergence_face_1(bary_coeff, int_coords)
    FF2D = _divergence_face_2(bary_coeff, int_coords)
    
    for iv1 in range(3):
        ie1 = local_edge_map[:, iv1]
        
        Le = distances[ie1[0], ie1[1]]
        Lengths[iv1,:] *= Le
        Lengths[iv1+4,:] *= Le
        Lengths[:,iv1] *= Le
        Lengths[:,iv1+4] *= Le
        
        FE1C_1 = _curl_edge_1(bary_coeff[:,ie1], int_coords)
        FE2C_1 = _curl_edge_2(bary_coeff[:,ie1], int_coords)
        FE1D_1 = _divergence_edge_1(bary_coeff[:,ie1], int_coords)
        FE2D_1 = _divergence_edge_2(bary_coeff[:,ie1], int_coords)
        
        for iv2 in range(3):
            ie2 = local_edge_map[:, iv2]
            
            FE1C_2 = _curl_edge_1(bary_coeff[:,ie2], int_coords)
            FE2C_2 = _curl_edge_2(bary_coeff[:,ie2], int_coords)
            FE1D_2 = _divergence_edge_1(bary_coeff[:,ie2], int_coords)
            FE2D_2 = _divergence_edge_2(bary_coeff[:,ie2], int_coords)
            
            CurlMatrix[iv1, iv2]     = _gqi(FE1C_1, FE1C_2, WEIGHTS)
            CurlMatrix[iv1, iv2+4]   = _gqi(FE1C_1, FE2C_2, WEIGHTS)
            CurlMatrix[iv1+4, iv2]   = _gqi(FE2C_1, FE1C_2, WEIGHTS)
            CurlMatrix[iv1+4, iv2+4] = _gqi(FE2C_1, FE2C_2, WEIGHTS)
            
            DivMatrix[iv1, iv2]     = _gqi(FE1D_1, FE1D_2, WEIGHTS)
            DivMatrix[iv1, iv2+4]   = _gqi(FE1D_1, FE2D_2, WEIGHTS)
            DivMatrix[iv1+4, iv2]   = _gqi(FE2D_1, FE1D_2, WEIGHTS)
            DivMatrix[iv1+4, iv2+4] = _gqi(FE2D_1, FE2D_2, WEIGHTS)
        
        CurlMatrix[iv1,  3]      = _gqi(FE1C_1, FF1C, WEIGHTS)
        CurlMatrix[iv1+4,3]      = _gqi(FE2C_1, FF1C, WEIGHTS)
        CurlMatrix[iv1,  7]      = _gqi(FE1C_1, FF2C, WEIGHTS)
        CurlMatrix[iv1+4,7]      = _gqi(FE2C_1, FF2C, WEIGHTS)
        
        CurlMatrix[3, iv1]   = CurlMatrix[iv1, 3]
        CurlMatrix[3, iv1+4] = CurlMatrix[iv1+4, 3]
        CurlMatrix[7, iv1]   = CurlMatrix[iv1, 7]
        CurlMatrix[7, iv1+4] = CurlMatrix[iv1+4, 7]
        
        DivMatrix[iv1,  3]      = _gqi(FE1D_1, FF1D, WEIGHTS)
        DivMatrix[iv1+4,3]      = _gqi(FE2D_1, FF1D, WEIGHTS)
        DivMatrix[iv1,  7]      = _gqi(FE1D_1, FF2D, WEIGHTS)
        DivMatrix[iv1+4,7]      = _gqi(FE2D_1, FF2D, WEIGHTS)
        
        DivMatrix[3, iv1]   = DivMatrix[iv1, 3]
        DivMatrix[3, iv1+4] = DivMatrix[iv1+4, 3]
        DivMatrix[7, iv1]   = DivMatrix[iv1, 7]
        DivMatrix[7, iv1+4] = DivMatrix[iv1+4, 7]
    
    CurlMatrix[3, 3] = _gqi(FF1C, FF1C, WEIGHTS)
    CurlMatrix[3, 7] = _gqi(FF1C, FF2C, WEIGHTS)
    CurlMatrix[7, 3] = _gqi(FF2C, FF1C, WEIGHTS)
    CurlMatrix[7, 7] = _gqi(FF2C, FF2C, WEIGHTS)
    
    DivMatrix[3, 3] = _gqi(FF1D, FF1D, WEIGHTS)
    DivMatrix[3, 7] = _gqi(FF1D, FF2D, WEIGHTS)
    DivMatrix[7, 3] = _gqi(FF2D, FF1D, WEIGHTS)
    DivMatrix[7, 7] = _gqi(FF2D, FF2D, WEIGHTS)
    
    Mat = cf*Lengths*(CurlMatrix-DivMatrix)*np.abs(Area)
    return Mat


############################################################
#            NUMBA OPTIMIZED INTEGRAL OVER TETS            #
############################################################

@njit((c16[:])(f8[:,:],
            i8[:,:], 
            i8[:,:], 
            i8[:,:],
            i8[:],
            c16), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tris, edges, tri_to_field, tri_ids, coeff):
    """ Numba optimized loop over each face triangle."""
    ntritot = tris.shape[1]
    nnz = ntritot*64
    
    Mat = np.zeros(nnz, dtype=np.complex128)

    tri_to_edge = tri_to_field[:3,:]
    
    Ntris = tri_ids.shape[0]
    for itri_sub in prange(Ntris): # type: ignore
        
        itri = tri_ids[itri_sub]
        p = itri*64

        # Construct a local mapping to global triangle orientations
        local_tri_map = local_tri_to_edgeid(itri, tris, edges, tri_to_edge)

        # Construct the local edge map
        tri_nodes = nodes[:, tris[:,itri]]
        subMat = _abc_order_2_terms(tri_nodes, local_tri_map, coeff)

        Mat[p:p+64] += subMat.ravel()
        
    return Mat

############################################################
#                     PYTHON INTERFACE                     #
############################################################

    
def abc_order_2_matrix(field: Nedelec2,
                       surf_triangle_indices: np.ndarray,
                       coeff: complex) -> np.ndarray:
    """Computes the second order absorbing boundary condition correction terms.

    Args:
        field (Nedelec2): The Basis function object
        surf_triangle_indices (np.ndarray): The surface triangle indices to add
        coeff (complex): The integral coefficient jp2/k0

    Returns:
        np.ndarray: The resultant matrix items
    """
    Mat = _matrix_builder(field.mesh.nodes, field.mesh.tris, field.mesh.edges, field.tri_to_field, surf_triangle_indices, coeff)
    return Mat