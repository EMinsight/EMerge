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
import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz # type: ignore
from ...solver import SolveReport

class SimJob:

    def __init__(self, 
                 A: csr_matrix,
                 b: np.ndarray | None,
                 freq: float,
                 cache_factorization: bool,
                 B: csr_matrix | None = None
                 ):

        self.A: csr_matrix = A
        self.B: csr_matrix | None = B
        self.b: np.ndarray | None = b
        self.P: csr_matrix | None= None
        self.Pd: csr_matrix | None = None
        self.has_periodic: bool = False

        self.freq: float = freq
        self.k0: float = 2*np.pi*freq/299792458
        self.cache_factorization: bool = cache_factorization
        self._fields: dict[int, np.ndarray] = dict()
        self.excitations: dict | None = None
        self.solve_ids: np.ndarray | None = None

        self.store_limit: int | None = None
        self.relative_path: str | None  = None
        self._store_location: dict = {}
        self._stored: bool = False
        
        self._sorter: np.ndarray | None = None
        self._isorter: np.ndarray | None = None
        
        self._active_port: int = -1
        self.reports: list[SolveReport] = []
        self.id: int = -1
        self.store_if_needed()

    def maybe_store(self, matrix, name):
        if self.store_limit is None:
            return matrix
        
        if matrix is not None and matrix.nnz > self.store_limit:
            # Create temp directory if needed
            os.makedirs(self.relative_path, exist_ok=True)
            path = os.path.join(self.relative_path, f"csr_{str(self.freq).replace('.','_')}_{name}.npz")
            save_npz(path, matrix, compressed=False)
            self._store_location[name] = path
            self._stored = True
            return None  # Unload from memory
        return matrix
    
    def set_sorter(self, order: np.ndarray):
        self._sorter = order
        self._isorter = np.argsort(order)
        
    def store_if_needed(self):
        self.A = self.maybe_store(self.A, 'A')
        if self.has_periodic:
            self.P = self.maybe_store(self.P, 'P')
            self.Pd = self.maybe_store(self.Pd, 'Pd')

    def load_if_needed(self, name):
        if name in self._store_location:
            return load_npz(self._store_location[name])
        return getattr(self, name)

    def iter_Ab(self):
        reuse_factorization = False

        for key, mode in self.excitations.items():
            # Set port as active and add the port mode to the forcing vector
            self._active_port = key

            b_active = self.b + mode
            A = self.load_if_needed('A')

            aux = {
                'Active Excitation': str(key),
            }

            if self.has_periodic:
                P = self.load_if_needed('P')
                Pd = self.load_if_needed('Pd')
                b_active = Pd @ b_active
                A = Pd @ A @ P
                aux['Periodic reduction'] = str(P.shape)

            A, b_active, solve_ids = self.sort(A, b_active, self.solve_ids)
            yield A, b_active, solve_ids, reuse_factorization, aux

            reuse_factorization = True
        
        self.cleanup()

    def yield_AC(self):
        A = self.A
        B = self.B
        
        if self.has_periodic:
            P = self.P
            Pd = self.Pd
            A = Pd @ A @ P
            B = Pd @ B @ P

        return A, B, self.solve_ids
        
    def fix_solutions(self, solution: np.ndarray) -> np.ndarray:
        if self.has_periodic:
            solution = self.P @ solution
        return solution
    
    def submit_solution(self, solution: np.ndarray, report: SolveReport):
        # Solve the Ax=b problem
        solution = self.unsort(solution)
        if self.has_periodic:
            solution = self.P @ solution
        # From now reuse the factorization

        self._fields[self._active_port] = solution
        self.reports.append(report)
        self.routine = None
    
    def cleanup(self):
        if not self._stored:
            return
        
        if not os.path.isdir(self.relative_path):
            return

        # Remove only the files we saved
        for path in self._store_location.values():
            if os.path.isfile(path):
                os.remove(path)

        # If the directory is now empty, remove it
        if not os.listdir(self.relative_path):
            os.rmdir(self.relative_path)
            
    def sort(self, A, b, solve_ids) -> tuple[csr_matrix, np.ndarray]:
        if self._sorter is None:
            return A, b, solve_ids
        Asorted = A[self._sorter, :][:, self._sorter]
        bsorted = b[self._sorter]
        counts = np.zeros((A.shape[0],), dtype=np.int64)
        counts[solve_ids] = 1
        counts = counts[self._sorter]
        
        return Asorted, bsorted, np.argwhere(counts==1).flatten()
    
    def unsort(self, x: np.ndarray) -> np.ndarray:
        if self._isorter is None:
            return x
        return  x[self._isorter]