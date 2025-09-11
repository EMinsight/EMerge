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

from __future__ import annotations
from ..mesh3d import Mesh3D
import numpy as np
from typing import Callable


class FEMBasis:

    def __init__(self, mesh: Mesh3D):
        self.mesh: Mesh3D = mesh
        self.n_edges: int = self.mesh.n_edges
        self.n_tris: int = self.mesh.n_tris
        self.n_tets: int = self.mesh.n_tets

        self.n_tet_dofs: int = -1
        self.n_tri_dofs: int = -1

        self.n_field: int = 2*self.n_edges + 2*self.n_tris

        self.tet_to_field: np.ndarray = np.array([])
        
        self.edge_to_field: np.ndarray = np.array([])

        self.tri_to_field: np.ndarray = np.array([])

        self._rows: np.ndarray = np.array([])
        self._cols: np.ndarray = np.array([])
        
    
    def interpolate_Ef(self, field: np.ndarray, basis: np.ndarray | None = None, origin: np.ndarray | None = None, tetids: np.ndarray | None = None) -> Callable:
        '''Generates the Interpolation function as a function object for a given coordiante basis and origin.'''
        from ..mth.optimized import matmul
        
        if basis is None:
            basis = np.eye(3)

        if origin is None:
            origin = np.zeros(3)
        
        ibasis = np.linalg.pinv(basis)
        def func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
            xyz = np.array([xs, ys, zs]) + origin[:, np.newaxis]
            xyzg = matmul(basis, xyz)
            return matmul(ibasis, np.array(self.interpolate(field, xyzg[0,:], xyzg[1,:], xyzg[2,:], tetids)))
        return func
    
    def interpolate(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, tetids: np.ndarray | None = None, usenan: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def interpolate_curl(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, constants: np.ndarray, tetids: np.ndarray | None = None, usenan: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates the curl of the field at the given points.
        """
        raise NotImplementedError()
    
    def interpolate_error(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, tetids: np.ndarray | None = None, cs: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    
    def empty_tet_matrix(self) -> np.ndarray:
        nnz = self.n_tets*self.n_tet_dofs**2
        matrix = np.empty((nnz,), dtype=np.complex128)
        return matrix
    
    def empty_tri_matrix(self) -> np.ndarray:
        nnz = self.n_tris*self.n_tri_dofs**2
        matrix = np.zeros((nnz,), dtype=np.complex128)
        return matrix
     
    def empty_tet_rowcol(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.n_tet_dofs
        N2 = N**2
        nnz = self.n_tets*N2
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)

        for itet in range(self.n_tets):
            p = itet*N2
            
            indices = self.tet_to_field[:, itet]
            for ii in range(self.n_tet_dofs):
                rows[p+N*ii:p+N*(ii+1)] = indices[ii]
                cols[p+ii:p+N2:N] = indices[ii]
        self._rows = rows
        self._cols = cols 
        return rows, cols
    
    def empty_tri_rowcol(self) -> tuple[np.ndarray, np.ndarray]:
        N = self.n_tri_dofs
        N2 = N**2
        nnz = self.n_tris*N2
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)

        for itri in range(self.n_tris):
            p = itri*N2
            indices = self.tri_to_field[:, itri]
            for ii in range(N):
                rows[p+N*ii:p+N*(ii+1)] = indices[ii]
                cols[p+ii:p+N2:N] = indices[ii]

        self._rows = rows
        self._cols = cols
        return rows, cols

    def tetslice(self, itet: int) -> slice:
        N = self.n_tet_dofs**2
        return slice(itet*N,(itet+1)*N)
    
    def trislice(self, itri: int) -> slice:
        N = self.n_tri_dofs**2
        return slice(itri*N,(itri+1)*N)
    
    def generate_csr(self, data: np.ndarray):
        from scipy.sparse import csr_matrix # type: ignore
        ids = np.argwhere(data!=0)[:,0]
        return csr_matrix((data[ids], (self._rows[ids], self._cols[ids])), shape=(self.n_field, self.n_field))
    ### QUANTITIES

    def tet_to_edge_lengths(self, itet: int) -> np.ndarray:
        """
        Returns the edge lengths of the tetrahedron itet.
        """
        return self.mesh.edge_lengths[self.mesh.tet_to_edge[:,itet]]
    
    def tri_to_edge_lengths(self, itri: int) -> np.ndarray:
        """
        Returns the edge lengths of the triangle itri.
        """
        return self.mesh.edge_lengths[self.mesh.tri_to_edge[:,itri]]
    ####### INDEX MAPPINGS

    def local_tet_to_triid(self, itet: int) -> np.ndarray:
        raise NotImplementedError("local_tet_to_triid not implemented")
    
    def local_tet_to_edgeid(self, itet: int) -> np.ndarray:
        raise NotImplementedError("local_tet_to_edgeid not implemented")

    def interpolate_index(self, xs: np.ndarray,
                    ys: np.ndarray,
                    zs: np.ndarray,
                    tetids: np.ndarray | None = None,
                    usenan: bool = True) -> np.ndarray:
        raise NotImplementedError()
    
    def map_edge_to_field(self, edge_ids: np.ndarray) -> np.ndarray:
        """
        Returns the field ids for the edges.
        """
        return edge_ids

    # ##### CUTS

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float],tuple[float, float]]:
        xmin = np.min(self.mesh.nodes[0,:])
        xmax = np.max(self.mesh.nodes[0,:])
        ymin = np.min(self.mesh.nodes[1,:])
        ymax = np.max(self.mesh.nodes[1,:])
        zmin = np.min(self.mesh.nodes[2,:])
        zmax = np.max(self.mesh.nodes[2,:])
        return (xmin, xmax), (ymin, ymax), (zmin, zmax)

    # @staticmethod
    # def tet_stiff_mass_submatrix(tet_vertices: np.ndarray, 
    #                              edge_lengths: np.ndarray, 
    #                              local_edge_map: np.ndarray, 
    #                              local_tri_map: np.ndarray, 
    #                              C_stiffness: float, 
    #                              C_mass: float) -> tuple[np.ndarray, np.ndarray]:
    #     pass
    
    # @staticmethod
    # def tri_stiff_mass_submatrix(tri_vertices: np.ndarray, 
    #                              edge_lengths: np.ndarray,
    #                              local_edge_map: np.ndarray,
    #                              C_stiffness: float, 
    #                              C_mass: float) -> tuple[np.ndarray, np.ndarray]:
    #     pass
    
    # @staticmethod
    # def tri_stiff_vec_matrix(lcs_vertices: np.ndarray, 
    #                          edge_lengths: np.ndarray, 
    #                          gamma: complex, 
    #                          lcs_Uinc: np.ndarray, 
    #                          DPTs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     pass

    # @staticmethod
    # def tri_stiff_matrix(lcs_vertices: np.ndarray, 
    #                          edge_lengths: np.ndarray, 
    #                          gamma: complex) -> tuple[np.ndarray, np.ndarray]:
    #     pass
    
    
    # @staticmethod
    # def tri_surf_integral(lcs_vertices: np.ndarray, 
    #                       edge_lengths: np.ndarray, 
    #                       lcs_Uinc: np.ndarray, 
    #                       DPTs: np.ndarray) -> complex:
    #     pass
