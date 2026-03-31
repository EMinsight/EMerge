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

# This specific function is written by Claude Code and optimized manually for memory reduction.

from __future__ import annotations
import numpy as np
from numba import njit, prange, c16, i8, types
from scipy.sparse import csc_matrix
from dataclasses import dataclass



@dataclass
class CSCMapping:
    indptr: np.ndarray
    indices: np.ndarray
    csc_map: np.ndarray
    N: int
    nnz: int = 0
    

    def __post_init__(self):
        self.nnz = self.indices.shape[0]

    @staticmethod
    def from_rowcol(rows, cols, N) -> CSCMapping:
        return CSCMapping(*precompute_csc_pattern(cols, rows, N), N)
    
    def to_csc(self, data: np.ndarray) -> csc_matrix:
        return csc_matrix((scatter_to_csc(data, self.csc_map, self.nnz), self.indices, self.indptr), shape=(self.N,self.N))
    

@njit(types.Tuple((i8[:], i8[:], i8[:]))(i8[:], i8[:], i8), nogil=True, cache=True, parallel=False)
def precompute_csc_pattern(cols, rows, N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-time precomputation: builds CSR structure and a mapping
    from each COO entry to its position in the CSR data array.
    
    Args:
        rows: int64 array of row indices (COO)
        cols: int64 array of col indices (COO)
        N: matrix dimension
        
    Returns:
        indptr: CSR row pointer array (length N+1)
        indices: CSR column index array (length nnz, deduplicated)
        csc_map: for each COO entry k, csc_map[k] is the index into
                 the CSR data array where that entry accumulates.
    """
    nnz_coo = cols.shape[0]
    
    # --- Pass 1: count unique (row, col) pairs per row ---
    # Sort COO entries by (row, col) using radix-style approach:
    # first, count entries per row
    col_offsets = np.zeros(N + 1, dtype=np.int64)
    for k in range(nnz_coo):
        col_offsets[cols[k]+1] += 1
    
    max_counts = 0
    # Build row offsets for a row-sorted permutation
    for i in range(1, N+1):
        col_offsets[i] = col_offsets[i-1] + col_offsets[i]
        diff = col_offsets[i]-col_offsets[i-1]
        if diff > max_counts:
            max_counts = diff
    
    # Scatter into row-sorted order
    # Perm tells where to place the next entry in the COO matrix
    perm = np.empty(nnz_coo, dtype=np.int32)
    cursor = col_offsets.copy()
    for k in range(nnz_coo):
        r = cols[k]
        perm[cursor[r]] = k
        cursor[r] += 1
    
    # --- Pass 2: for each row, sort columns and find unique entries ---
    # First, count unique nnz per row to build indptr
    indptr = np.zeros(N+1, dtype=np.int64)
    
    local_cols = np.empty(max_counts, dtype=np.int64)
    local_perm = np.empty(max_counts, dtype=np.int64)
    for i in range(N):
        start = col_offsets[i]
        end = col_offsets[i + 1]
        if start == end:
            continue
        count = end - start
        
        for j in range(count):
            local_cols[j] = rows[perm[start + j]]
            local_perm[j] = j

        # Insertion sort on local_cols (and track the permutation)
        for j in range(1, count):
            key_col = local_cols[j]
            key_p = local_perm[j]
            m = j - 1
            while m >= 0 and local_cols[m] > key_col:
                local_cols[m + 1] = local_cols[m]
                local_perm[m + 1] = local_perm[m]
                m -= 1
            local_cols[m + 1] = key_col
            local_perm[m + 1] = key_p
        
        # Count unique columns
        n_unique = 1
        for j in range(1, count):
            if local_cols[j] != local_cols[j - 1]:
                n_unique += 1
        indptr[i+1] = n_unique
    
    # Build indptr by cumulative sum
    for i in range(1,N+1):
        indptr[i] = indptr[i-1] + indptr[i]
    
    total_nnz = indptr[N]
    indices = np.empty(total_nnz, dtype=np.int64)
    csc_map = np.empty(nnz_coo, dtype=np.int64)
    
    # --- Pass 3: fill indices and csr_map ---
    for i in range(N):
        start = col_offsets[i]
        end = col_offsets[i + 1]
        if start == end:
            continue
        count = end - start

        for j in range(count):
            local_cols[j] = rows[perm[start + j]]
            local_perm[j] = j
        
        # Same insertion sort
        for j in range(1, count):
            key_col = local_cols[j]
            key_p = local_perm[j]
            m = j - 1
            while m >= 0 and local_cols[m] > key_col:
                local_cols[m + 1] = local_cols[m]
                local_perm[m + 1] = local_perm[m]
                m -= 1
            local_cols[m + 1] = key_col
            local_perm[m + 1] = key_p
        
        # Walk sorted entries, assign CSR positions
        csc_pos = indptr[i]
        indices[csc_pos] = local_cols[0]
        csc_map[perm[start + local_perm[0]]] = csc_pos
        
        for j in range(1, count):
            if local_cols[j] != local_cols[j - 1]:
                csc_pos += 1
                indices[csc_pos] = local_cols[j]
            csc_map[perm[start + local_perm[j]]] = csc_pos
    
    return indptr, indices, csc_map


@njit(c16[:](c16[:], i8[:], i8), nogil=True, cache=True, parallel=True)
def scatter_to_csc(data_coo, csr_map, nnz):
    """Scatter COO values into CSR data array, summing duplicates.
    
    Args:
        data_coo: complex128 array of COO values
        csr_map: precomputed mapping from COO index -> CSR data index
        nnz: number of unique nonzeros (length of CSR data array)
        
    Returns:
        data: CSR data array with duplicates summed

    This function is written by Claude Code and checked by Robert Fennis
    """
    data = np.zeros(nnz, dtype=data_coo.dtype)
    
    # Parallel chunked scatter — each thread accumulates into a private
    # array, then we reduce. This avoids write conflicts.
    n = data_coo.shape[0]
    n_threads = 8  # numba prange will clamp to available cores
    chunk = (n + n_threads - 1) // n_threads
    
    # Allocate per-thread buffers
    thread_data = np.zeros((n_threads, nnz), dtype=data_coo.dtype)
    
    for t in prange(n_threads):
        start = t * chunk
        end = min(start + chunk, n)
        for k in range(start, end):
            thread_data[t, csr_map[k]] += data_coo[k]
    
    # Reduce across threads
    for t in range(n_threads):
        for j in prange(nnz):
            data[j] += thread_data[t, j]
    
    return data