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
from ....mth.optimized import local_mapping, matinv, dot_c, cross_c, compute_distances
from ....mth.csc_cast import CSCMapping
from ....mth.csr_cast import CSRMapping
from numba import c16, types, f8, i8, njit, prange, void


############################################################
#                  CACHED FACTORIAL VALUES                 #
############################################################

_FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880], dtype=np.int64)

############################################################
#                  INDEX MAPPING FUNCTIONS                 #
############################################################

# These mapping functions return edge and face coordinates in the appropriate order.

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

@njit(c16[:](c16[:,:], c16[:]), cache=True, nogil=True)
def matmul(Mat, Vec):
    ## Matrix multiplication of a 3D vector
    Vout = np.empty((3,), dtype=np.complex128)
    Vout[0] = Mat[0,0]*Vec[0] + Mat[0,1]*Vec[1] + Mat[0,2]*Vec[2]
    Vout[1] = Mat[1,0]*Vec[0] + Mat[1,1]*Vec[1] + Mat[1,2]*Vec[2]
    Vout[2] = Mat[2,0]*Vec[0] + Mat[2,1]*Vec[1] + Mat[2,2]*Vec[2]
    return Vout

@njit(void(c16[:,:], c16[:], c16[:]), cache=True, nogil=True)
def matmul_inplace(Mat, Vec, Vout):
    v0, v1, v2 = Vec[0], Vec[1], Vec[2]
    Vout[0] = Mat[0,0]*v0 + Mat[0,1]*v1 + Mat[0,2]*v2
    Vout[1] = Mat[1,0]*v0 + Mat[1,1]*v1 + Mat[1,2]*v2
    Vout[2] = Mat[2,0]*v0 + Mat[2,1]*v1 + Mat[2,2]*v2

@njit(void(c16[:], c16[:], c16[:]), cache=True, fastmath=True, nogil=True)
def cross_c_inplace(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Optimized complex single vector cross product

    Args:
        a (np.ndarray): (3,) vector a
        b (np.ndarray): (3,) vector b

    Returns:
        np.ndarray: a ⨉ b
    """
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]

@njit(f8(i8, i8, i8, i8), cache=True, fastmath=True, nogil=True)
def volume_coeff(a: int, b: int, c: int, d: int):
    """ Computes the appropriate matrix coefficients given a list of
    barycentric coordinate functions mentioned.
    Example:
      - L1^2 * L2 - volume_coeff(1,1,2,0) """
    klmn = np.array([0,0,0,0,0,0,0])
    klmn[a] += 1
    klmn[b] += 1
    klmn[c] += 1
    klmn[d] += 1
    output = (_FACTORIALS[klmn[1]]*_FACTORIALS[klmn[2]]*_FACTORIALS[klmn[3]]
                  *_FACTORIALS[klmn[4]]*_FACTORIALS[klmn[5]]*_FACTORIALS[klmn[6]])/_FACTORIALS[(np.sum(klmn[1:])+3)]
    return output


############################################################
#        PRECOMPUTATION OF INTEGRATION COEFFICIENTS       #
############################################################

NFILL = 5
VOLUME_COEFF_CACHE_BASE = np.zeros((NFILL,NFILL,NFILL,NFILL), dtype=np.float64)
for I in range(NFILL):
    for J in range(NFILL):
        for K in range(NFILL):
            for L in range(NFILL):
                VOLUME_COEFF_CACHE_BASE[I,J,K,L] = volume_coeff(I,J,K,L)

VOLUME_COEFF_CACHE = VOLUME_COEFF_CACHE_BASE


############################################################
#  COMPUTATION OF THE BARYCENTRIC COORDINATE COEFFICIENTS #
############################################################

@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients_bcd(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + \
                x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + \
                x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - \
                x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - \
                x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
    bbs[0] = -y2*z3 + y2*z4 + y3*z2 - y3*z4 - y4*z2 + y4*z3
    bbs[1] = y1*z3 - y1*z4 - y3*z1 + y3*z4 + y4*z1 - y4*z3
    bbs[2] = -y1*z2 + y1*z4 + y2*z1 - y2*z4 - y4*z1 + y4*z2
    bbs[3] = y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2
    ccs[0] = x2*z3 - x2*z4 - x3*z2 + x3*z4 + x4*z2 - x4*z3
    ccs[1] = -x1*z3 + x1*z4 + x3*z1 - x3*z4 - x4*z1 + x4*z3
    ccs[2] = x1*z2 - x1*z4 - x2*z1 + x2*z4 + x4*z1 - x4*z2
    ccs[3] = -x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2
    dds[0] = -x2*y3 + x2*y4 + x3*y2 - x3*y4 - x4*y2 + x4*y3
    dds[1] = x1*y3 - x1*y4 - x3*y1 + x3*y4 + x4*y1 - x4*y3
    dds[2] = -x1*y2 + x1*y4 + x2*y1 - x2*y4 - x4*y1 + x4*y2
    dds[3] = x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2

    return bbs, ccs, dds, V


@njit(types.Tuple((f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
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


############################################################
#              MAIN CURL-CURL MATRIX ASSEMBLY             #
############################################################

def tet_mass_stiffness_matrices(field: Nedelec2,
                                er: np.ndarray, 
                                ur: np.ndarray, 
                                cscmap: CSRMapping | None = None) -> tuple[np.ndarray, np.ndarray, CSRMapping]:
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
    tet_to_edge = field.mesh.tet_to_edge


    dataE, dataB, rows, cols = _matrix_builder(nodes, tets, tris, edges, field.mesh.edge_lengths, tet_to_field, tet_to_edge, ur, er)
    if cscmap is None:
        cscmap = CSCMapping.from_rowcol(rows, cols, field.n_field)
    return dataE, dataB, cscmap


############################################################
#           NUMBA ACCELLERATE SUB-MATRIX ASSEMBLY          #
############################################################

@njit(types.Tuple((c16[:,:],c16[:,:]))(f8[:,:], f8[:], i8[:,:], i8[:,:], c16[:,:], c16[:,:]), nogil=True, cache=True, parallel=False, fastmath=True)
def ned2_tet_stiff_mass(tet_vertices, edge_lengths, local_edge_map, local_tri_map, Ms, Mm):
    ''' Nedelec 2 tetrahedral stiffness and mass matrix submatrix Calculation

    '''
    
    Dmat = np.empty((20,20), dtype=np.complex128)
    Fmat = np.empty((20,20), dtype=np.complex128)

    xs, ys, zs = tet_vertices

    bbs, ccs, dds, V = tet_coefficients_bcd_opt(xs, ys, zs)
    b1, b2, b3, b4 = bbs
    c1, c2, c3, c4 = ccs
    d1, d2, d3, d4 = dds
    
    Ds = compute_distances(xs, ys, zs)

    GL1 = np.array([b1, c1, d1]).astype(np.complex128)
    GL2 = np.array([b2, c2, d2]).astype(np.complex128)
    GL3 = np.array([b3, c3, d3]).astype(np.complex128)
    GL4 = np.array([b4, c4, d4]).astype(np.complex128)

    GLs = (GL1, GL2, GL3, GL4)

    letters = [1,2,3,4,5,6]

    KA = 1/(6*V)**4
    KB = 1/(6*V)**2

    V6 = 6*V

    VOLUME_COEFF_CACHE = VOLUME_COEFF_CACHE_BASE*V6
    
    # Preallocation
    erGF = np.empty((3,), dtype=np.complex128)
    erGC = np.empty((3,), dtype=np.complex128)
    erGD = np.empty((3,), dtype=np.complex128)
    CROSS_CF = np.empty((3,), dtype=np.complex128)
    CROSS_DF = np.empty((3,), dtype=np.complex128)
    CROSS_CD = np.empty((3,), dtype=np.complex128)
    
    Ms_MUL_GCxGF = np.empty((3,), dtype=np.complex128)
    Ms_MUL_GDxGF = np.empty((3,), dtype=np.complex128)
    Ms_MUL_GCxGD = np.empty((3,), dtype=np.complex128)
    Ms_MUL_GCxGD = np.empty((3,), dtype=np.complex128)
    GAxGB = np.empty((3,), dtype=np.complex128)
    GCxGD = np.empty((3,), dtype=np.complex128)
    GCxGF = np.empty((3,), dtype=np.complex128)
    GDxGF = np.empty((3,), dtype=np.complex128)
    GCxGD = np.empty((3,), dtype=np.complex128)
    GAxGE = np.empty((3,), dtype=np.complex128)
    GBxGE = np.empty((3,), dtype=np.complex128)
    
    for ei in range(6):
        ei1 = local_edge_map[0, ei]
        ei2 = local_edge_map[1, ei]
        GA = GLs[ei1]
        GB = GLs[ei2]
        A, B = letters[ei1], letters[ei2]
        L1 = edge_lengths[ei]
        
        VAA_mat = VOLUME_COEFF_CACHE[A,A]
        VAB_mat = VOLUME_COEFF_CACHE[A,B]
        VBB_mat = VOLUME_COEFF_CACHE[B,B]
        
        for ej in range(6):
            ej1 = local_edge_map[0, ej]
            ej2 = local_edge_map[1, ej]
            
            C,D = letters[ej1], letters[ej2]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VABCD = VAB_mat[C,D]
            VABCC = VAB_mat[C,C]
            VABDD = VAB_mat[D,D]
            VBBCD = VBB_mat[C,D]
            VAACD = VAA_mat[C,D]
            VAADD = VAA_mat[D,D]
            VBBCC = VBB_mat[C,C]
            VBBDD = VBB_mat[D,D]
            VAACC = VAA_mat[C,C]

            L2 = edge_lengths[ej]

            matmul_inplace(Mm,GC, erGF)
            matmul_inplace(Mm,GD, erGC)
            GA_MUL_erGD = dot_c(GA,erGF)
            GE_MUL_erGF = dot_c(GA,erGC)
            GE_MUL_erGC = dot_c(GB,erGF)
            GA_MUL_erGF = dot_c(GB,erGC)

            L12 = L1*L2
            cross_c_inplace(GA,GB,GAxGB)
            cross_c_inplace(GC,GD,GCxGD)
            matmul_inplace(Ms,GCxGD,Ms_MUL_GCxGD)
            Factor = L12*9*dot_c(GAxGB,Ms_MUL_GCxGD)
            Dmat[ei+0,ej+0] = Factor*VAC
            Dmat[ei+0,ej+10] = Factor*VAD
            Dmat[ei+10,ej+0] = Factor*VBC
            Dmat[ei+10,ej+10] = Factor*VBD
            
            Fmat[ei+0,ej+0] = L12*(VABCD*GA_MUL_erGD-VABCC*GE_MUL_erGF-VAACD*GE_MUL_erGC+VAACC*GA_MUL_erGF)
            Fmat[ei+0,ej+10] = L12*(VABDD*GA_MUL_erGD-VABCD*GE_MUL_erGF-VAADD*GE_MUL_erGC+VAACD*GA_MUL_erGF)
            Fmat[ei+10,ej+0] = L12*(VBBCD*GA_MUL_erGD-VBBCC*GE_MUL_erGF-VABCD*GE_MUL_erGC+VABCC*GA_MUL_erGF)
            Fmat[ei+10,ej+10] = L12*(VBBDD*GA_MUL_erGD-VBBCD*GE_MUL_erGF-VABDD*GE_MUL_erGC+VABCD*GA_MUL_erGF)       


        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]

            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            VABCD = VAB_mat[C,D]
            VBBCD = VBB_mat[C,D]
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VABDF = VAB_mat[D,F]
            VABCF = VAB_mat[F,C]
            VAADF = VAA_mat[D,F]
            VAACD = VAA_mat[C,D]
            VBBDF = VBB_mat[D,F]
            VBBCF = VBB_mat[F,C]
            VAACF = VAA_mat[C,F]

            Lab2 = Ds[ej1, ej2]
            Lac2 = Ds[ej1, fj]
            
            cross_c_inplace(GA,GB,GAxGB)
            cross_c_inplace(GC,GF,GCxGF)
            cross_c_inplace(GD,GF,GDxGF)
            cross_c_inplace(GC,GD,GCxGD)
            matmul_inplace(Ms,GCxGF,Ms_MUL_GCxGF)
            matmul_inplace(Ms,GDxGF,Ms_MUL_GDxGF)
            matmul_inplace(Ms,GCxGD,Ms_MUL_GCxGD)
            AE_MUL_DF = dot_c(GAxGB,Ms_MUL_GCxGF)
            AE_MUL_CD = dot_c(GAxGB,Ms_MUL_GDxGF)
            AE_MUL_CF = dot_c(GAxGB,Ms_MUL_GCxGD)
            matmul_inplace(Mm,GF,erGF)
            matmul_inplace(Mm,GC,erGC)
            matmul_inplace(Mm,GD,erGD)
            GE_MUL_erGF = dot_c(GA,erGF)
            GE_MUL_erGC = dot_c(GA,erGC)
            GA_MUL_erGF = dot_c(GB,erGF)
            GA_MUL_erGC = dot_c(GB,erGC)
            GE_MUL_erGD = dot_c(GA,erGD)
            GA_MUL_erGD = dot_c(GB,erGD)
            
            L1Lac2 = L1*Lac2
            L1Lab2 = L1*Lab2
            Dmat[ei+0,ej+6] = L1Lac2*(-6*VAD*AE_MUL_DF-3*VAC*AE_MUL_CD-3*VAF*AE_MUL_CF)
            Dmat[ei+0,ej+16] = L1Lab2*(6*VAF*AE_MUL_CF+3*VAD*AE_MUL_DF-3*VAC*AE_MUL_CD)
            Dmat[ei+10,ej+6] = L1Lac2*(-6*VBD*AE_MUL_DF-3*VBC*AE_MUL_CD-3*VBF*AE_MUL_CF)
            Dmat[ei+10,ej+16] = L1Lab2*(6*VBF*AE_MUL_CF+3*VBD*AE_MUL_DF-3*VBC*AE_MUL_CD)

            Fmat[ei+0,ej+6] = L1Lac2*(VABCD*GE_MUL_erGF-VABDF*GE_MUL_erGC-VAACD*GA_MUL_erGF+VAADF*GA_MUL_erGC)
            Fmat[ei+0,ej+16] = L1Lab2*(VABDF*GE_MUL_erGC-VABCF*GE_MUL_erGD-VAADF*GA_MUL_erGC+VAACF*GA_MUL_erGD)
            Fmat[ei+10,ej+6] = L1Lac2*(VBBCD*GE_MUL_erGF-VBBDF*GE_MUL_erGC-VABCD*GA_MUL_erGF+VABDF*GA_MUL_erGC)
            Fmat[ei+10,ej+16] = L1Lab2*(VBBDF*GE_MUL_erGC-VBBCF*GE_MUL_erGD-VABDF*GA_MUL_erGC+VABCF*GA_MUL_erGD)
    
    ## Mirror the transpose part of the previous iteration as its symmetrical

    Dmat[6:10, :6] = Dmat[:6, 6:10].T
    Fmat[6:10, :6] = Fmat[:6, 6:10].T
    Dmat[16:20, :6] = Dmat[:6, 16:20].T
    Fmat[16:20, :6] = Fmat[:6, 16:20].T
    Dmat[6:10, 10:16] = Dmat[10:16, 6:10].T
    Fmat[6:10, 10:16] = Fmat[10:16, 6:10].T
    Dmat[16:20, 10:16] = Dmat[10:16, 16:20].T
    Fmat[16:20, 10:16] = Fmat[10:16, 16:20].T
    
    for ei in range(4):
        ei1, ei2, fi = local_tri_map[:, ei]
        A, B, E = letters[ei1], letters[ei2], letters[fi]
        VAA_mat = VOLUME_COEFF_CACHE[A,A]
        VAB_mat = VOLUME_COEFF_CACHE[A,B]
        VBB_mat = VOLUME_COEFF_CACHE[B,B]
        
        GA = GLs[ei1]
        GB = GLs[ei2]
        GE = GLs[fi]
        Lac1 = Ds[ei1, fi]
        Lab1 = Ds[ei1, ei2]
        
        cross_c_inplace(GA,GE,GAxGE)
        cross_c_inplace(GB,GE,GBxGE)
        cross_c_inplace(GA,GB,GAxGB)
        
        for ej in range(4):
            ej1, ej2, fj = local_tri_map[:, ej]
            
            C,D,F = letters[ej1], letters[ej2], letters[fj]
            
            GC = GLs[ej1]
            GD = GLs[ej2]
            GF = GLs[fj]

            VABCD = VAB_mat[C,D]
            VAD = VOLUME_COEFF_CACHE[A,D,0,0]
            VAC = VOLUME_COEFF_CACHE[A,C,0,0]
            VAF = VOLUME_COEFF_CACHE[A,F,0,0]
            VBF = VOLUME_COEFF_CACHE[B,F,0,0]
            VBC = VOLUME_COEFF_CACHE[B,C,0,0]
            VBD = VOLUME_COEFF_CACHE[B,D,0,0]
            VDE = VOLUME_COEFF_CACHE[E,D,0,0]
            VEF = VOLUME_COEFF_CACHE[E,F,0,0]
            VCE = VOLUME_COEFF_CACHE[E,C,0,0]
            VABDF = VAB_mat[D,F]
            VACEF = VOLUME_COEFF_CACHE[A,C,E,F]
            VABCF = VAB_mat[F,C]
            VBCDE = VOLUME_COEFF_CACHE[B,C,D,F]
            VBDEF = VOLUME_COEFF_CACHE[B,E,D,F]
            VACDE = VOLUME_COEFF_CACHE[E,A,C,D]
            VBCEF = VOLUME_COEFF_CACHE[B,E,F,C]
            VADEF = VOLUME_COEFF_CACHE[E,A,D,F]

            Lac2 = Ds[ej1, fj]
            Lab2 = Ds[ej1, ej2]
            
            matmul_inplace(Ms, cross_c(GC,GF), CROSS_CF)
            matmul_inplace(Ms, cross_c(GD,GF), CROSS_DF)
            matmul_inplace(Ms, cross_c(GC,GD), CROSS_CD)
            AE_MUL_CF = dot_c(GAxGE,CROSS_CF)
            AE_MUL_DF = dot_c(GAxGE,CROSS_DF)
            AE_MUL_CD = dot_c(GAxGE,CROSS_CD)
            BE_MUL_CF = dot_c(GBxGE,CROSS_CF)
            BE_MUL_DF = dot_c(GBxGE,CROSS_DF)
            BE_MUL_CD = dot_c(GBxGE,CROSS_CD)
            AB_MUL_CF = dot_c(GAxGB,CROSS_CF)
            AB_MUL_DF = dot_c(GAxGB,CROSS_DF)
            AB_MUL_CD = dot_c(GAxGB,CROSS_CD)
            matmul_inplace(Mm,GF,erGF)
            matmul_inplace(Mm,GC,erGC)
            matmul_inplace(Mm,GD,erGD)
            GE_MUL_erGF = dot_c(GE,erGF)
            GE_MUL_erGC = dot_c(GE,erGC)
            GA_MUL_erGF = dot_c(GA,erGF)
            GA_MUL_erGC = dot_c(GA,erGC)
            GE_MUL_erGD = dot_c(GE,erGD)
            GA_MUL_erGD = dot_c(GA,erGD)
            GB_MUL_erGF = dot_c(GB,erGF)
            GB_MUL_erGC = dot_c(GB,erGC)
            GB_MUL_erGD = dot_c(GB,erGD)

            Q1 = 2*VAD*BE_MUL_CF+VAC*BE_MUL_DF+VAF*BE_MUL_CD
            L12 = -2*VAF*BE_MUL_CD-VAD*BE_MUL_CF+VAC*BE_MUL_DF
            Dmat[ei+6,ej+6] = Lac1*Lac2*(4*VBD*AE_MUL_CF+2*VBC*AE_MUL_DF+2*VBF*AE_MUL_CD+Q1+2*VDE*AB_MUL_CF+VCE*AB_MUL_DF+VEF*AB_MUL_CD)
            Dmat[ei+6,ej+16] = Lac1*Lab2*(-4*VBF*AE_MUL_CD-2*VBD*AE_MUL_CF+2*VBC*AE_MUL_DF+L12-2*VEF*AB_MUL_CD-VDE*AB_MUL_CF+VCE*AB_MUL_DF)
            Dmat[ei+16,ej+6] = Lab1*Lac2*(-4*VDE*AB_MUL_CF-2*VCE*AB_MUL_DF-2*VEF*AB_MUL_CD-2*VBD*AE_MUL_CF-VBC*AE_MUL_DF-VBF*AE_MUL_CD+Q1)
            Dmat[ei+16,ej+16] = Lab1*Lab2*(4*VEF*AB_MUL_CD+2*VDE*AB_MUL_CF-2*VCE*AB_MUL_DF+2*VBF*AE_MUL_CD+VBD*AE_MUL_CF-VBC*AE_MUL_DF+L12)
            Fmat[ei+6,ej+6] = Lac1*Lac2*(VABCD*GE_MUL_erGF-VABDF*GE_MUL_erGC-VBCDE*GA_MUL_erGF+VBDEF*GA_MUL_erGC)
            Fmat[ei+6,ej+16] = Lac1*Lab2*(VABDF*GE_MUL_erGC-VABCF*GE_MUL_erGD-VBDEF*GA_MUL_erGC+VBCEF*GA_MUL_erGD)
            Fmat[ei+16,ej+6] = Lab1*Lac2*(VBCDE*GA_MUL_erGF-VBDEF*GA_MUL_erGC-VACDE*GB_MUL_erGF+VADEF*GB_MUL_erGC)
            Fmat[ei+16,ej+16] = Lab1*Lab2*(VBDEF*GA_MUL_erGC-VBCEF*GA_MUL_erGD-VADEF*GB_MUL_erGC+VACEF*GB_MUL_erGD)

    Dmat = Dmat*KA
    Fmat = Fmat*KB

    return Dmat, Fmat


############################################################
#             NUMBA ACCELLERATED MATRIX BUILDER            #
############################################################

@njit(types.Tuple((c16[:], c16[:], i8[:], i8[:]))(f8[:,:], 
                                                i8[:,:], 
                                                i8[:,:], 
                                                i8[:,:], 
                                                f8[:], 
                                                i8[:,:], 
                                                i8[:,:], 
                                                c16[:,:,:], 
                                                c16[:,:,:]), cache=True, nogil=True, parallel=True)
def _matrix_builder(nodes, tets, tris, edges, all_edge_lengths, tet_to_field, tet_to_edge, ur, er):
    nT = tets.shape[1]
    nedges = edges.shape[1]

    nnz = nT*400

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    dataE = np.empty_like(rows, dtype=np.complex128)
    dataB = np.empty_like(rows, dtype=np.complex128)

    
    for itet in prange(nT): # ty: ignore
        p = itet*400
        urt = ur[:,:,itet]
        ert = er[:,:,itet]

        # Construct a local mapping to global triangle orientations
        
        local_tri_map = local_tet_to_triid(tet_to_field, tets, tris, itet, nedges)
        local_edge_map = local_tet_to_edgeid(tets, edges, tet_to_field, itet)
        edge_lengths = all_edge_lengths[tet_to_edge[:,itet]]

        # Construct the local edge map

        Esub, Bsub = ned2_tet_stiff_mass(nodes[:,tets[:,itet]], 
                                                edge_lengths, 
                                                local_edge_map, 
                                                local_tri_map, 
                                                matinv(urt), ert)
        
        indices = tet_to_field[:, itet]
        for ii in range(20):
            rows[p+20*ii:p+20*(ii+1)] = indices[ii]
            cols[p+ii:p+400+ii:20] = indices[ii]

        dataE[p:p+400] = Esub.ravel()
        dataB[p:p+400] = Bsub.ravel()
    return dataE, dataB, rows, cols


