import numpy as np
import os
from typing import Hashable
from scipy.sparse import csc_matrix, save_npz, load_npz, issparse # type: ignore
from ...solver import SolveReport
from loguru import logger

class SimJob:
    
    def __init__(self, 
                 A: csc_matrix,
                 b_vec: np.ndarray,
                 i_prescribed: list[int],
                 t_prescribed: list[float]):
        
        self.A: csc_matrix = A
        self.b_vec: np.ndarray = b_vec
        self.i_presribed: list[int] = i_prescribed
        self.i_free: list[int] = [i for i in range(self.A.shape[0]) if i not in self.i_presribed]
        self.n_dof: int = len(self.i_free)
        self.t_prescribed: np.ndarray = np.array(t_prescribed)
        self.solution: np.ndarray | None = None
        self.reports: list[SolveReport] = []
        self.id: int = -1


    def get_Ab(self) -> tuple[csc_matrix, np.ndarray, list[int], dict]:
        Afd = self.A[self.i_free,:][:,self.i_presribed]
        

        bvec = self.b_vec[self.i_free] - Afd @ self.t_prescribed
        solve_ids = np.array([i for i in range(self.n_dof)])
        bvec = bvec.reshape((self.n_dof, 1))
        return self.A[self.i_free,:][:,self.i_free], bvec, solve_ids, dict()
    
    def submit_solution(self, solution: np.ndarray, report: dict):
        self.reports.append(report)
        self.solution = np.zeros((self.A.shape[0],), dtype=np.float64)
        self.solution[self.i_presribed] = self.t_prescribed
        self.solution[self.i_free] = solution[:,0]
