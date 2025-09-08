# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the TERMS of the GNU General Public License
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
from numba import njit, types, i8, c16
from scipy.sparse import csr_matrix

############################################################
#                      NUMBA COMPILED                     #
############################################################

@njit(types.Tuple((i8[:], i8[:], c16[:], c16[:]))(i8[:,:], i8[:,:], i8[:,:], i8[:,:], i8), cache=True, nogil=True)
def _fill_periodic_matrix(tris: np.ndarray, edges: np.ndarray, tri_to_field: np.ndarray, edge_to_field: np.ndarray, Nfield: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates sparse matrix row, column ids and ones plus the matrix diagonal.

    Args:
        tris (: np.ndarray): The triangle ids
        edges (: np.ndarray): The edge ids
        tri_to_field (: np.ndarray): The triangle to field index mapping
        edge_to_field (: np.ndarray): The edge to field index mapping
        Nfield (int): The number of field points

    Returns:
        np.ndarray: The row ids
        np.ndarray: The column ids
        np.ndarray: The data
        np.ndarray: The diagonal array
    """

    # NUMBERS
    N = tris.shape[1] + edges.shape[1]
    NT = tris.shape[1]
    
    DIAGONAL = np.ones((Nfield,), dtype=np.complex128)
    ROWS = np.zeros((N*2,), dtype=np.int64)
    COLS = np.zeros((N*2,), dtype=np.int64)
    TERMS = np.zeros((N*2,), dtype=np.complex128)
    
    i = 0
    for it in range(NT):
        t1 = tris[0,it]
        t2 = tris[1,it]
        f11 = tri_to_field[3, t1]
        f21 = tri_to_field[7, t1]
        f12 = tri_to_field[3, t2]
        f22 = tri_to_field[7, t2]
        DIAGONAL[f12] = 0.
        DIAGONAL[f22] = 0.
        ROWS[i] = f12
        ROWS[i+1] = f22
        COLS[i] = f11
        COLS[i+1] = f21
        TERMS[i] = 1.0
        TERMS[i+1] = 1.0
        i += 2
    NE = edges.shape[1]
    for ie in range(NE):
        e1 = edges[0,ie]
        e2 = edges[1,ie]
        f11 = edge_to_field[0, e1]
        f21 = edge_to_field[1, e1]
        f12 = edge_to_field[0, e2]
        f22 = edge_to_field[1, e2]
        DIAGONAL[f12] = 0.
        DIAGONAL[f22] = 0.
        ROWS[i] = f12
        ROWS[i+1] = f22
        COLS[i] = f11
        COLS[i+1] = f21
        TERMS[i] = 1.0
        TERMS[i+1] = 1.0
        i += 2
    ROWS = ROWS[:i]
    COLS = COLS[:i]
    TERMS = TERMS[:i]
    return ROWS, COLS, TERMS, DIAGONAL


############################################################
#                     PYTHON INTERFACE                    #
############################################################

def gen_periodic_matrix(tris: np.ndarray, 
                        edges: np.ndarray, 
                        tri_to_field: np.ndarray, 
                        edge_to_field: np.ndarray, 
                        linked_tris: dict[int, int], 
                        linked_edges: dict[int, int], 
                        Nfield: int, 
                        phi: complex) -> tuple[csr_matrix, np.ndarray]:
    """This function constructs the periodic boundary matrix

    Args:
        tris (np.ndarray): _description_
        edges (np.ndarray): _description_
        tri_to_field (np.ndarray): _description_
        edge_to_field (np.ndarray): _description_
        linked_tris (dict[int, int]): _description_
        linked_edges (dict[int, int]): _description_
        Nfield (int): _description_
        phi (complex): _description_

    Returns:
        tuple[csr_matrix, np.ndarray]: _description_
    """

    tris_array = np.array([(tri, linked_tris[tri]) for tri in tris]).T
    edges_array = np.array([(edge, linked_edges[edge]) for edge in edges]).T
    ROWS, COLS, TERMS, diagonal = _fill_periodic_matrix(tris_array, edges_array, tri_to_field, edge_to_field, Nfield)
    matrix = csr_matrix((TERMS, (ROWS, COLS)), [Nfield, Nfield], dtype=np.complex128)
    matrix.data.fill(phi)
    matrix.setdiag(diagonal)
    
    return matrix, ROWS