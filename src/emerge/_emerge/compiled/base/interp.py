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
from numba import njit, f8, c16, i8, types, prange # type: ignore
from numba import get_thread_id as get_thread_id

import numpy as np
from ...mth.optimized import compute_distances

EPS = 0.00000001

@njit(f8[:,:](f8[:,:]), cache=True, nogil=True)
def matinv(M: np.ndarray) -> np.ndarray:
    """Optimized matrix inverse of 3x3 matrix

    Args:
        M (np.ndarray): Input matrix M of shape (3,3)

    Returns:
        np.ndarray: The matrix inverse inv(M)
    """
    out = np.empty((3,3), dtype=np.float64)
   
    det = M[0,0]*M[1,1]*M[2,2] - M[0,0]*M[1,2]*M[2,1] - M[0,1]*M[1,0]*M[2,2] + M[0,1]*M[1,2]*M[2,0] + M[0,2]*M[1,0]*M[2,1] - M[0,2]*M[1,1]*M[2,0]
    out[0,0] = M[1,1]*M[2,2] - M[1,2]*M[2,1]
    out[0,1] = -M[0,1]*M[2,2] + M[0,2]*M[2,1]
    out[0,2] = M[0,1]*M[1,2] - M[0,2]*M[1,1]
    out[1,0] = -M[1,0]*M[2,2] + M[1,2]*M[2,0]
    out[1,1] = M[0,0]*M[2,2] - M[0,2]*M[2,0]
    out[1,2] = -M[0,0]*M[1,2] + M[0,2]*M[1,0]
    out[2,0] = M[1,0]*M[2,1] - M[1,1]*M[2,0]
    out[2,1] = -M[0,0]*M[2,1] + M[0,1]*M[2,0]
    out[2,2] = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    out = out/det
    return out

@njit(types.Tuple((f8[:], f8[:], f8[:], f8[:], f8))(f8[:], f8[:], f8[:]), cache = True, nogil=True)
def tet_coefficients(xs, ys, zs):
    ## THIS FUNCTION WORKS
    x1, x2, x3, x4 = xs
    y1, y2, y3, y4 = ys
    z1, z2, z3, z4 = zs

    aas = np.empty((4,), dtype=np.float64)
    bbs = np.empty((4,), dtype=np.float64)
    ccs = np.empty((4,), dtype=np.float64)
    dds = np.empty((4,), dtype=np.float64)

    V = np.abs(-x1*y2*z3/6 + x1*y2*z4/6 + x1*y3*z2/6 - x1*y3*z4/6 - x1*y4*z2/6 + x1*y4*z3/6 + x2*y1*z3/6 - x2*y1*z4/6 - x2*y3*z1/6 + x2*y3*z4/6 + x2*y4*z1/6 - x2*y4*z3/6 - x3*y1*z2/6 + x3*y1*z4/6 + x3*y2*z1/6 - x3*y2*z4/6 - x3*y4*z1/6 + x3*y4*z2/6 + x4*y1*z2/6 - x4*y1*z3/6 - x4*y2*z1/6 + x4*y2*z3/6 + x4*y3*z1/6 - x4*y3*z2/6)
    
    aas[0] = x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2
    aas[1] = -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1
    aas[2] = x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1
    aas[3] = -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1
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

    return aas, bbs, ccs, dds, V

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

    #A = 0.5*(b1*c2 - b2*c1)
    sA = 0.5*(((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)))
    sign = np.sign(sA)
    A = np.abs(sA)
    As = np.array([a1, a2, a3])*sign
    Bs = np.array([b1, b2, b3])*sign
    Cs = np.array([c1, c2, c3])*sign
    return As, Bs, Cs, A

@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def local_mapping(vertex_ids, triangle_ids):
    """
    Parameters
    ----------
    vertex_ids   : 1-D int64 array (length 4)
        Global vertex 0.1005964238ers of one tetrahedron, in *its* order
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
    out  = np.zeros(triangle_ids.shape, dtype=np.int64)

    for t in range(ntri):                 # each triangle
        for j in range(ndim):                # each vertex in that triangle
            gid = triangle_ids[j, t]      # global ID to look up

            # linear search over the four tet vertices
            for k in range(4):
                if vertex_ids[k] == gid:
                    out[j, t] = k         # store local index 0-3
                    break                 # stop the k-loop

    return out

@njit(i8[:](i8[:,:]), cache=True, nogil=True, parallel=True)
def matmax(A: np.ndarray) -> np.ndarray:
    ncols = A.shape[0]
    out = np.empty((ncols,), dtype=np.int64)
    for j in prange(ncols):
        out[j] = np.max(A[j,:])
    return out

@njit(types.Tuple((i8[:], i8[:], i8[:]))(i8[:]), cache=True, nogil=True)
def get_group_indices(assigned_sorted):
    # Count how many unique tets we have
    if len(assigned_sorted) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    
    # Pre-calculate unique count to allocate arrays
    n_unique = 1
    for i in range(1, len(assigned_sorted)):
        if assigned_sorted[i] != assigned_sorted[i-1]:
            n_unique += 1
            
    unique_tets = np.empty(n_unique, dtype=assigned_sorted.dtype)
    first_indices = np.empty(n_unique, dtype=np.int64)
    last_indices = np.empty(n_unique, dtype=np.int64)
    
    # Fill the arrays
    curr_idx = 0
    unique_tets[0] = assigned_sorted[0]
    first_indices[0] = 0
    
    for i in range(1, len(assigned_sorted)):
        if assigned_sorted[i] != assigned_sorted[i-1]:
            last_indices[curr_idx] = i
            curr_idx += 1
            unique_tets[curr_idx] = assigned_sorted[i]
            first_indices[curr_idx] = i
            
    last_indices[curr_idx] = len(assigned_sorted)
    
    return unique_tets, first_indices, last_indices

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:], i8[:]), cache=True, nogil=True, parallel=True)
def ned2_tet_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tets: np.ndarray, 
                    tris: np.ndarray,
                    edges: np.ndarray,
                    nodes: np.ndarray,
                    tet_to_field: np.ndarray,
                    tetids: np.ndarray):
    
    ''' Nedelec 2 tetrahedral interpolation'''
    NThreads = 6
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    nEdges = edges.shape[1]
    nTetIds = tetids.shape[0]
    
    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    
    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)
    setnan = np.zeros((nNodes, ), dtype=np.int64)
    assigned = np.full((nNodes,NThreads), -1, dtype=np.int64)

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        iv1, iv2, iv3, iv4 = tets[0, itet], tets[1, itet], tets[2, itet], tets[3, itet]
        
        v1x, v1y, v1z = nodes[0, iv1], nodes[1, iv1], nodes[2, iv1]

        m00 = nodes[0, iv2] - v1x
        m10 = nodes[1, iv2] - v1y
        m20 = nodes[2, iv2] - v1z
        
        m01 = nodes[0, iv3] - v1x
        m11 = nodes[1, iv3] - v1y
        m21 = nodes[2, iv3] - v1z
        
        m02 = nodes[0, iv4] - v1x
        m12 = nodes[1, iv4] - v1y
        m22 = nodes[2, iv4] - v1z

        det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)
        inv_det = 1.0 / det

        b00 = (m11*m22 - m12*m21) * inv_det
        b01 = (m02*m21 - m01*m22) * inv_det
        b02 = (m01*m12 - m02*m11) * inv_det
        
        b10 = (m12*m20 - m10*m22) * inv_det
        b11 = (m00*m22 - m02*m20) * inv_det
        b12 = (m10*m02 - m00*m12) * inv_det
        
        b20 = (m10*m21 - m11*m20) * inv_det
        b21 = (m20*m01 - m00*m21) * inv_det
        b22 = (m00*m11 - m10*m01) * inv_det
        tid = get_thread_id()

        for j in range(nNodes):

            dx = coords[0, j] - v1x
            dy = coords[1, j] - v1y
            dz = coords[2, j] - v1z

            u = b00*dx + b01*dy + b02*dz
            v = b10*dx + b11*dy + b12*dz
            w = b20*dx + b21*dy + b22*dz
            
            if (u >= -EPS) and (v >= -EPS) and (w >= -EPS) and (u + v + w <= 1.0 + EPS):
                assigned[j,tid] = itet
    
    assigned = matmax(assigned)
    sort_idx = np.argsort(assigned)
    xs_s = xs[sort_idx]
    ys_s = ys[sort_idx]
    zs_s = zs[sort_idx]
    assigned_sorted = assigned[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side='right')
    
    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]
        
        if start == end:
            continue

        xvs = nodes[0, tets[:,itet]]
        yvs = nodes[1, tets[:,itet]]
        zvs = nodes[2, tets[:,itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        Ds = compute_distances(xvs, yvs, zvs)
        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_field[:6, itet]]
        g_tri_ids = tris[:, tet_to_field[6:10, itet]-nEdges]

        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)
        
        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]
        
        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]

        V1 = 1/(216*V**3)

        for ie in range(6):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]

            LV = Ds[edgeids[0], edgeids[1]]*V1

            for i in range(start,end):
                x = xs_s[i]
                y = ys_s[i]
                z = zs_s[i]
                idx = sort_idx[i]
                F1 = (a1 + b1*x + c1*y + d1*z)
                F2 = (a2 + b2*x + c2*y + d2*z)
                F3 = (Em1*F1 + Em2*F2)
                Ex[idx] += LV*F3*(b1*F2 - b2*F1)
                Ey[idx] += LV*F3*(c1*F2 - c2*F1)
                Ez[idx] += LV*F3*(d1*F2 - d2*F1)
            
        for ie in range(4):
            Em1, Em2 = Ef1s[ie], Ef2s[ie]
            triids = l_tri_ids[:, ie]
            a1, a2, a3 = a_s[triids]
            b1, b2, b3 = b_s[triids]
            c1, c2, c3 = c_s[triids]
            d1, d2, d3 = d_s[triids]

            L1 = Ds[l_tri_ids[2, ie], l_tri_ids[0,ie]]
            L2 = Ds[l_tri_ids[1, ie], l_tri_ids[0,ie]]

            for i in range(start,end):
                x = xs_s[i]
                y = ys_s[i]
                z = zs_s[i]
                idx = sort_idx[i]

                F1 = (a1 + b1*x + c1*y + d1*z)
                F2 = (a2 + b2*x + c2*y + d2*z)
                F3 = (a3 + b3*x + c3*y + d3*z)
                
                Q1 = Em1*L1*F2
                Q2 = Em2*L2*F3

                Ex[idx] += (-Q1*(b1*F3 - b3*F1) + Q2*(b1*F2 - b2*F1))*V1
                Ey[idx] += (-Q1*(c1*F3 - c3*F1) + Q2*(c1*F2 - c2*F1))*V1
                Ez[idx] += (-Q1*(d1*F3 - d3*F1) + Q2*(d1*F2 - d2*F1))*V1
                
        inside = sort_idx[start:end]
        setnan[inside] = 1
    
    Ex[setnan==0] = np.nan
    Ey[setnan==0] = np.nan
    Ez[setnan==0] = np.nan
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:], i8[:,:], c16[:], i8[:]), cache=True, nogil=True, parallel=True)
def ned2_tet_interp_curl(coords: np.ndarray,
                         solutions: np.ndarray, 
                         tets: np.ndarray, 
                         tris: np.ndarray,
                         edges: np.ndarray,
                         nodes: np.ndarray,
                         tet_to_field: np.ndarray,
                         c: np.ndarray,
                         tetids: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation of the analytic curl'''
    # Solution has shape (nEdges, nsols)
    NThreads = 6
    nNodes = coords.shape[1]
    nEdges = edges.shape[1]
    nTetIds = tetids.shape[0]

    xs = coords[0,:]
    ys = coords[1,:]
    zs = coords[2,:]
    
    Ex = np.zeros((nNodes, ), dtype=np.complex128)
    Ey = np.zeros((nNodes, ), dtype=np.complex128)
    Ez = np.zeros((nNodes, ), dtype=np.complex128)
    setnan = np.zeros((nNodes, ), dtype=np.int64)
    assigned = np.full((nNodes,NThreads), -1, dtype=np.int64)

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]

        iv1, iv2, iv3, iv4 = tets[0, itet], tets[1, itet], tets[2, itet], tets[3, itet]
        
        v1x, v1y, v1z = nodes[0, iv1], nodes[1, iv1], nodes[2, iv1]
        
        m00 = nodes[0, iv2] - v1x
        m10 = nodes[1, iv2] - v1y
        m20 = nodes[2, iv2] - v1z
        
        m01 = nodes[0, iv3] - v1x
        m11 = nodes[1, iv3] - v1y
        m21 = nodes[2, iv3] - v1z
        
        m02 = nodes[0, iv4] - v1x
        m12 = nodes[1, iv4] - v1y
        m22 = nodes[2, iv4] - v1z

        det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)
        inv_det = 1.0 / det

        b00 = (m11*m22 - m12*m21) * inv_det
        b01 = (m02*m21 - m01*m22) * inv_det
        b02 = (m01*m12 - m02*m11) * inv_det
        
        b10 = (m12*m20 - m10*m22) * inv_det
        b11 = (m00*m22 - m02*m20) * inv_det
        b12 = (m10*m02 - m00*m12) * inv_det
        
        b20 = (m10*m21 - m11*m20) * inv_det
        b21 = (m20*m01 - m00*m21) * inv_det
        b22 = (m00*m11 - m10*m01) * inv_det

        tid = get_thread_id()

        for j in range(nNodes):
            dx = coords[0, j] - v1x
            dy = coords[1, j] - v1y
            dz = coords[2, j] - v1z
            
            u = b00*dx + b01*dy + b02*dz
            v = b10*dx + b11*dy + b12*dz
            w = b20*dx + b21*dy + b22*dz
            
            if (u >= -EPS) and (v >= -EPS) and (w >= -EPS) and (u + v + w <= 1.0 + EPS):
                assigned[j, tid] = itet

    assigned = matmax(assigned)
    sort_idx = np.argsort(assigned)
    xs_s = xs[sort_idx]
    ys_s = ys[sort_idx]
    zs_s = zs[sort_idx]
    assigned_sorted = assigned[sort_idx]
    offsets = np.searchsorted(assigned_sorted, tetids)
    offsets_end = np.searchsorted(assigned_sorted, tetids, side='right')
    
    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]
        start = offsets[i_iter]
        end = offsets_end[i_iter]
        
        if start == end:
            continue
            
        xvs = nodes[0, tets[:,itet]]
        yvs = nodes[1, tets[:,itet]]
        zvs = nodes[2, tets[:,itet]]

        a_s, b_s, c_s, d_s, V = tet_coefficients(xvs, yvs, zvs)
        Ds = compute_distances(xvs, yvs, zvs)
        
        g_node_ids = tets[:, itet]
        g_edge_ids = edges[:, tet_to_field[:6, itet]]
        g_tri_ids = tris[:, tet_to_field[6:10, itet]-nEdges]

        l_edge_ids = local_mapping(g_node_ids, g_edge_ids)
        l_tri_ids = local_mapping(g_node_ids, g_tri_ids)
        
        field_ids = tet_to_field[:, itet]
        Etet = solutions[field_ids]
        
        const = c[itet]

        ######### INSIDE THE TETRAHEDRON #########

        Em1s = Etet[0:6]
        Ef1s = Etet[6:10]
        Em2s = Etet[10:16]
        Ef2s = Etet[16:20]
        
        V1 = const/(216*V**3)
        V2 = const/(72*V**3)
        
        for ie in range(6):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            d1, d2 = d_s[edgeids]
            x1, x2 = xvs[edgeids]
            y1, y2 = yvs[edgeids]
            z1, z2 = zvs[edgeids]

            L = Ds[edgeids[0], edgeids[1]]
            C1 = Em1*a1
            C2 = Em1*b1
            C3 = Em1*c1
            C4 = Em1*c2
            C5 = Em2*a2
            C6 = Em2*b2
            C7 = Em2*c1
            C8 = Em2*c2
            C9 = Em1*b2
            C10 = Em2*b1
            D1 = c1*d2
            D2 = c2*d1
            D3 = d1*d2
            D4 = d1*d1
            D5 = c2*d2
            D6 = d2*d2
            D7 = b1*d2
            D8 = b2*d1
            D9 = c1*d1
            D10 = b2*d2
            D11 = b1*c2
            D12 = b2*c1
            D13 = c1*c2
            D14 = c1*c1
            D15 = b2*c2
            KXc = -C1*D1 + C1*D2 - C5*D1 + C5*D2
            KXx = (-C2*D1 + C2*D2 - C6*D1 + C6*D2)
            KXy = (-C3*D1 + C3*D2 - C7*D5 + C8*D2)
            KXz = (-C3*D3 + C4*D4 - C7*D6 + C8*D3)
            
            KYc = (C1*D7 - C1*D8 + C5*D7 - C5*D8)
            KYx = (C2*D7 - C2*D8 + C10*D10 - C6*D8)
            KYy = (C2*D1 - C9*D9 + C10*D5 - C6*D2)
            KYz = (C2*D3 - C9*D4 + C10*D6 - C6*D3)

            KZc = -C1*D11 + C1*D12 - C5*D11 + C5*D12
            KZx = (-C2*D11 + C2*D12 - C10*D15 + C6*D12)
            KZy = (-C2*D13 + C9*D14 - C10*c2*c2 + C6*D13)
            KZz = (-C2*D2 + C9*D9 - C10*D5 + C6*D1)
            VL = V2*L
            for i in range(start,end):
                x = xs_s[i]
                y = ys_s[i]
                z = zs_s[i]
                idx = sort_idx[i]
                Ex[idx] +=  VL*(KXc + KXx*x + KXy*y + KXz*z)
                Ey[idx] +=  VL*(KYc + KYx*x + KYy*y + KYz*z)
                Ez[idx] +=  VL*(KZc + KZx*x + KZy*y + KZz*z)
        
        for ie in range(4):
            Em1, Em2 = Ef1s[ie], Ef2s[ie]
            triids = l_tri_ids[:, ie]
            a1, a2, a3 = a_s[triids]
            b1, b2, b3 = b_s[triids]
            c1, c2, c3 = c_s[triids]
            d1, d2, d3 = d_s[triids]

            x1, x2, x3 = xvs[l_tri_ids[:, ie]]
            y1, y2, y3 = yvs[l_tri_ids[:, ie]]
            z1, z2, z3 = zvs[l_tri_ids[:, ie]]

            N6 = ((c1*d2) - (c2*d1))
            N10 = ((b1*d2) - (b2*d1))
            N11 = ((b1*c2) - (b2*c1))
            N12 = (b1*c3 - b3*c1)
            N3 = (c1*d3 - c3*d1)
            N8 = (b1*d3 - b3*d1)
            Em1L1 = Em1*Ds[l_tri_ids[0, ie], l_tri_ids[2, ie]]
            Em2L2 = Em2*Ds[l_tri_ids[0, ie], l_tri_ids[1, ie]]
            for i in range(start,end):
                x = xs_s[i]
                y = ys_s[i]
                z = zs_s[i]
                idx = sort_idx[i]
                F1 = (a3 + b3*x + c3*y + d3*z)
                F2 = (a1 + b1*x + c1*y + d1*z)
                F3 = (a2 + b2*x + c2*y + d2*z)
                N1 = (d1*F1 - d3*F2)
                N2 = (c1*F1 - c3*F2)
                N4 = (d1*F3 - d2*F2)
                N5 = (c1*F3 - c2*F2)
                N7 = (b1*F1 - b3*F2)
                N9 = (b1*F3 - b2*F2)
                Ex[idx] += (Em1L1*(-c2*N1 + d2*N2 + 2*N3*F3) - Em2L2*(-c3*N4 + d3*N5 + 2*N6*F1))*V1
                Ey[idx] += -(Em1L1*(-b2*N1 + d2*N7 + 2*N8*F3) - Em2L2*(-b3*N4 + d3*N9 + 2*N10*F1))*V1
                Ez[idx] += (Em1L1*(-b2*N2 + c2*N7 + 2*N12*F3) - Em2L2*(-b3*N5 + c3*N9 + 2*N11*F1))*V1

        inside = sort_idx[start:end]
        setnan[inside] = 1
    
    Ex[setnan==0] = np.nan
    Ey[setnan==0] = np.nan
    Ez[setnan==0] = np.nan
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
def ned2_tri_interp(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]

    Ex = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes, ), np.nan, dtype=np.complex128)

    nodes = nodes[:2,:]

    l_edge_ids = np.array([[0,1,0],[1,2,2]])

    for itri in range(tris.shape[1]):

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1+EPS) & (coords_local[0,:] >= -EPS) & (coords_local[1,:] >= -EPS)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        Ds = compute_distances(xvs, yvs, 0*xvs)

        L1 = Ds[0,1]
        L2 = Ds[1,2]
        L3 = Ds[0,2]

        mult = np.array([L1,L2,L3,L3,L1,L2,L3,L1])

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        
        Etri = Etri*mult
        
        Em1s = Etri[:3]
        Ef1s = Etri[3]
        Em2s = Etri[4:7]
        Ef2s = Etri[7]
        
        Exl = np.zeros(x.shape, dtype=np.complex128)
        Eyl = np.zeros(x.shape, dtype=np.complex128)
        
        for ie in range(3):
            Em1, Em2 = Em1s[ie], Em2s[ie]
            edgeids = l_edge_ids[:, ie]
            a1, a2 = a_s[edgeids]
            b1, b2 = b_s[edgeids]
            c1, c2 = c_s[edgeids]
            
            ex =  (Em1*(a1 + b1*x + c1*y) + Em2*(a2 + b2*x + c2*y))*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))/(8*A**3)
            ey =  (Em1*(a1 + b1*x + c1*y) + Em2*(a2 + b2*x + c2*y))*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))/(8*A**3)
            
            Exl += ex
            Eyl += ey
        
    
        Em1, Em2 = Ef1s, Ef2s
        triids = np.array([0,1,2])

        a1, a2, a3 = a_s[triids]
        b1, b2, b3 = b_s[triids]
        c1, c2, c3 = c_s[triids]

        ex =  (-Em1*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + Em2*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        ey =  (-Em1*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + Em2*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y))/(8*A**3)
        
        Exl += ex
        Eyl += ey

        Ex[inside] = Exl
        Ey[inside] = Eyl
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:]), cache=True, nogil=True)
def ned2_tri_interp_full(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]

    Ex = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes, ), np.nan, dtype=np.complex128)

    nodes = nodes[:2,:]

    for itri in range(tris.shape[1]):

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.0 + EPS) & (coords_local[0,:] >= -EPS) & (coords_local[1,:] >= -EPS)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TRIANGLE #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s
        
        # New Nedelec-1 order 2 formulation
        ex = (-2*A*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))) - e4*((b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) - e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*((b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y)))/(8*A**3)
        ey = (-2*A*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))) - e4*((c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) - e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*((c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y)))/(8*A**3)
        ez = (-e10*(a2 + b2*x + c2*y)*(A - a2 - b2*x - c2*y)/2 - e11*(a3 + b3*x + c3*y)*(A - a3 - b3*x - c3*y)/2 + e12*(a1 + b1*x + c1*y)*(a2 + b2*x + c2*y) + e13*(a2 + b2*x + c2*y)*(a3 + b3*x + c3*y) + e14*(a1 + b1*x + c1*y)*(a3 + b3*x + c3*y) - e9*(a1 + b1*x + c1*y)*(A - a1 - b1*x - c1*y)/2)/A**2
        Ex[inside] = ex
        Ey[inside] = ey
        Ez[inside] = ez
    return Ex, Ey, Ez

@njit(types.Tuple((c16[:], c16[:], c16[:]))(f8[:,:], c16[:], i8[:,:], f8[:,:], i8[:,:], c16[:,:,:], c16), cache=True, nogil=True)
def ned2_tri_interp_curl(coords: np.ndarray,
                    solutions: np.ndarray, 
                    tris: np.ndarray,
                    nodes: np.ndarray,
                    tri_to_field: np.ndarray,
                    diadic: np.ndarray,
                    beta: float):
    ''' Nedelec 2 tetrahedral interpolation'''
    ### THIS IS VERIFIED TO WORK
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    xs = coords[0,:]
    ys = coords[1,:]
    jB = 1j*beta
    Ex = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ey = np.full((nNodes, ), np.nan, dtype=np.complex128)
    Ez = np.full((nNodes, ), np.nan, dtype=np.complex128)

    nodes = nodes[:2,:]

    for itri in range(tris.shape[1]):
        
        dc = diadic[:,:,itri]

        iv1, iv2, iv3 = tris[:, itri]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]

        bv1 = v2 - v1
        bv2 = v3 - v1

        blocal = np.zeros((2,2))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))

        field_ids = tri_to_field[:, itri]

        Etri = solutions[field_ids]

        inside = ((coords_local[0,:] + coords_local[1,:]) <= 1.0 + EPS) & (coords_local[0,:] >= -EPS) & (coords_local[1,:] >= -EPS)

        if inside.sum() == 0:
            continue
        
        ######### INSIDE THE TETRAHEDRON #########
        
        x = xs[inside==1]
        y = ys[inside==1]

        xvs = nodes[0, tris[:,itri]]
        yvs = nodes[1, tris[:,itri]]

        a_s, b_s, c_s, A = tri_coefficients(xvs, yvs)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = Etri

        a1, a2, a3 = a_s
        b1, b2, b3 = b_s
        c1, c2, c3 = c_s
        
        # New Nedelec-1 order 2 formulation
        hx = (4*A*(2*c1*e12*(a2 + b2*x + c2*y) + 2*c1*e14*(a3 + b3*x + c3*y) + c1*e9*(a1 + b1*x + c1*y) - c1*e9*(A - a1 - b1*x - c1*y) + c2*e10*(a2 + b2*x + c2*y) - c2*e10*(A - a2 - b2*x - c2*y) + 2*c2*e12*(a1 + b1*x + c1*y) + 2*c2*e13*(a3 + b3*x + c3*y) + c3*e11*(a3 + b3*x + c3*y) - c3*e11*(A - a3 - b3*x - c3*y) + 2*c3*e13*(a2 + b2*x + c2*y) + 2*c3*e14*(a1 + b1*x + c1*y)) + jB*(2*A*(e1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e2*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e3*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))) + e4*((c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) + e5*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) + e6*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) + e7*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) - e8*((c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y))))/(8*A**3)
        hy = (4*A*(-2*b1*e12*(a2 + b2*x + c2*y) - 2*b1*e14*(a3 + b3*x + c3*y) - b1*e9*(a1 + b1*x + c1*y) + b1*e9*(A - a1 - b1*x - c1*y) - b2*e10*(a2 + b2*x + c2*y) + b2*e10*(A - a2 - b2*x - c2*y) - 2*b2*e12*(a1 + b1*x + c1*y) - 2*b2*e13*(a3 + b3*x + c3*y) - b3*e11*(a3 + b3*x + c3*y) + b3*e11*(A - a3 - b3*x - c3*y) - 2*b3*e13*(a2 + b2*x + c2*y) - 2*b3*e14*(a1 + b1*x + c1*y)) - jB*(2*A*(e1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + e2*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + e3*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))) + e4*((b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y) + (b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a1 + b1*x + c1*y)) + e5*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) + e6*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y))*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) + e7*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) - e8*((b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))*(a3 + b3*x + c3*y) + (b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))*(a2 + b2*x + c2*y))))/(8*A**3)
        hz = (4*A*(e1*(b1*c2 - b2*c1) + e2*(b2*c3 - b3*c2) + e3*(b1*c3 - b3*c1)) - e4*(b1*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + b2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) - (b1*c3 - b3*c1)*(a2 + b2*x + c2*y) - (b2*c3 - b3*c2)*(a1 + b1*x + c1*y)) + e4*(c1*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + c2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + (b1*c3 - b3*c1)*(a2 + b2*x + c2*y) + (b2*c3 - b3*c2)*(a1 + b1*x + c1*y)) - e5*(b1 - b2)*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) + e5*(c1 - c2)*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + 2*e5*(b1*c2 - b2*c1)*(a1 - a2 + b1*x - b2*x + c1*y - c2*y) - e6*(b2 - b3)*(c2*(a3 + b3*x + c3*y) - c3*(a2 + b2*x + c2*y)) + e6*(c2 - c3)*(b2*(a3 + b3*x + c3*y) - b3*(a2 + b2*x + c2*y)) + 2*e6*(b2*c3 - b3*c2)*(a2 - a3 + b2*x - b3*x + c2*y - c3*y) - e7*(b1 - b3)*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) + e7*(c1 - c3)*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + 2*e7*(b1*c3 - b3*c1)*(a1 - a3 + b1*x - b3*x + c1*y - c3*y) + e8*(b2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y)) + b3*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y)) - (b1*c2 - b2*c1)*(a3 + b3*x + c3*y) - (b1*c3 - b3*c1)*(a2 + b2*x + c2*y)) - e8*(c2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y)) + c3*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y)) + (b1*c2 - b2*c1)*(a3 + b3*x + c3*y) + (b1*c3 - b3*c1)*(a2 + b2*x + c2*y)))/(8*A**3)     
        
        Ex[inside] = hx*dc[0,0]
        Ey[inside] = hy*dc[1,1]
        Ez[inside] = hz*dc[2,2]
    return Ex, Ey, Ez



@njit(i8[:](f8[:,:], i8[:,:], f8[:,:], i8[:]), cache=True, nogil=True, parallel=True)
def index_interp(coords: np.ndarray,
                tets: np.ndarray, 
                nodes: np.ndarray,
                tetids: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation of the analytic curl'''
    NThreads = 6
    nNodes = coords.shape[1]
    nTetIds = tetids.shape[0]
    assigned = np.full((nNodes,NThreads), -1, dtype=np.int64)

    for i_iter in prange(nTetIds):
        itet = tetids[i_iter]

        # 1. Direct access - avoid slicing where possible
        iv1, iv2, iv3, iv4 = tets[0, itet], tets[1, itet], tets[2, itet], tets[3, itet]
        
        v1x, v1y, v1z = nodes[0, iv1], nodes[1, iv1], nodes[2, iv1]
        
        # 2. Manual 3x3 Jacobian (b-local)
        m00 = nodes[0, iv2] - v1x
        m10 = nodes[1, iv2] - v1y
        m20 = nodes[2, iv2] - v1z
        
        m01 = nodes[0, iv3] - v1x
        m11 = nodes[1, iv3] - v1y
        m21 = nodes[2, iv3] - v1z
        
        m02 = nodes[0, iv4] - v1x
        m12 = nodes[1, iv4] - v1y
        m22 = nodes[2, iv4] - v1z

        # 3. Determinant for manual inversion
        det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)
        inv_det = 1.0 / det

        # 4. Manual Inverse Basis (Cramer's Rule)
        b00 = (m11*m22 - m12*m21) * inv_det
        b01 = (m02*m21 - m01*m22) * inv_det
        b02 = (m01*m12 - m02*m11) * inv_det
        
        b10 = (m12*m20 - m10*m22) * inv_det
        b11 = (m00*m22 - m02*m20) * inv_det
        b12 = (m10*m02 - m00*m12) * inv_det
        
        b20 = (m10*m21 - m11*m20) * inv_det
        b21 = (m20*m01 - m00*m21) * inv_det
        b22 = (m00*m11 - m10*m01) * inv_det

        tid = get_thread_id()
        # 5. Point-check loop (Avoids all temporary arrays!)
        for j in range(nNodes):
            # Translate point
            dx = coords[0, j] - v1x
            dy = coords[1, j] - v1y
            dz = coords[2, j] - v1z
            
            # Matmul: basis @ dx
            u = b00*dx + b01*dy + b02*dz
            v = b10*dx + b11*dy + b12*dz
            w = b20*dx + b21*dy + b22*dz
            
            # Barycentric coordinate check
            if (u >= -EPS) and (v >= -EPS) and (w >= -EPS) and (u + v + w <= 1.0 + EPS):
                assigned[j, tid] = itet

    assigned = matmax(assigned)
    return assigned


@njit(f8[:](f8[:,:], i8[:,:], f8[:,:], i8[:], f8[:]), cache=True, nogil=True)
def constant_interp(coords: np.ndarray,
                tets: np.ndarray, 
                nodes: np.ndarray,
                tetids: np.ndarray,
                value: np.ndarray):
    ''' Nedelec 2 tetrahedral interpolation of the analytic curl'''
    # Solution has shape (nEdges, nsols)
    nNodes = coords.shape[1]
    
    prop = np.full((nNodes, ), 0, dtype=np.float64)
    
    for i_iter in range(tetids.shape[0]):
        itet = tetids[i_iter]
        
        iv1, iv2, iv3, iv4 = tets[:, itet]

        v1 = nodes[:,iv1]
        v2 = nodes[:,iv2]
        v3 = nodes[:,iv3]
        v4 = nodes[:,iv4]

        bv1 = v2 - v1
        bv2 = v3 - v1
        bv3 = v4 - v1

        blocal = np.zeros((3,3))
        blocal[:,0] = bv1
        blocal[:,1] = bv2
        blocal[:,2] = bv3
        basis = np.linalg.pinv(blocal)

        coords_offset = coords - v1[:,np.newaxis]
        coords_local = (basis @ (coords_offset))


        inside = ((coords_local[0,:] + coords_local[1,:] + coords_local[2,:]) <= 1.00000001) & (coords_local[0,:] >= -1e-6) & (coords_local[1,:] >= -1e-6) & (coords_local[2,:] >= -1e-6)

        if inside.sum() == 0:
            continue
        
        prop[inside] = value[itet]

    return prop
