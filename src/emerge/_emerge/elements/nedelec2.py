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
from ..mesh3d import Mesh3D
from .femdata import FEMBasis
from emsutil import Saveable
from loguru import logger
from ..compiled import MATHLIB

############### Nedelec2 Class
USE_NUMBA = False

class Nedelec2(FEMBasis, Saveable):


    def __init__(self, mesh: Mesh3D):
        super().__init__(mesh)
        
        self.nedges: int = self.mesh.n_edges
        self.ntris: int = self.mesh.n_tris
        self.ntets: int = self.mesh.n_tets

        self.nfield: int = 2*self.nedges + 2*self.ntris
        
        ######## MESH Derived

        nedges = self.mesh.n_edges
        ntris = self.mesh.n_tris

        self.tet_to_field: np.ndarray = np.zeros((20, self.mesh.tets.shape[1]), dtype=int)
        self.tet_to_field[:6,:] = self.mesh.tet_to_edge
        self.tet_to_field[6:10,:] = self.mesh.tet_to_tri + nedges
        self.tet_to_field[10:16,:] = self.mesh.tet_to_edge + (ntris+nedges)
        self.tet_to_field[16:20,:] = self.mesh.tet_to_tri + (ntris+2*nedges)
    
        self.edge_to_field: np.ndarray = np.zeros((2,nedges), dtype=int)

        self.edge_to_field[0,:] = np.arange(nedges)
        self.edge_to_field[1,:] = np.arange(nedges) + ntris + nedges

        self.tri_to_field: np.ndarray = np.zeros((8,ntris), dtype=int)

        self.tri_to_field[:3,:] = self.mesh.tri_to_edge
        self.tri_to_field[3,:] = np.arange(ntris) + nedges
        self.tri_to_field[4:7,:] = self.mesh.tri_to_edge + nedges + ntris
        self.tri_to_field[7,:] = np.arange(ntris) + 2*nedges + ntris

        ##
        self._field: np.ndarray | None = None
        self.n_tet_dofs = 20
        self.n_tri_dofs = 8
        self._all_tet_ids = np.arange(self.ntets)

        self.diagnose()

        self.empty_tri_rowcol()
    
    def diagnose(self):
        visited_field = np.zeros((self.n_field,), dtype=np.bool)
        for i in range(self.mesh.n_tets):
            visited_field[self.tet_to_field[:,i]] = True
        assert np.all(visited_field)
        
    def interpolate(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, tetids: np.ndarray | None = None, usenan: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' 
        Interpolate the provided field data array at the given xs, ys and zs coordinates
        '''
        
        tetids = self._all_tet_ids
        vals = MATHLIB.ned2_tet_interp(np.array([xs, ys, zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, tetids)
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f'Converted {n_zeros} to zeros.')
            vals = np.nan_to_num(vals)
        return vals
    
    def interpolate_curl(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs:np.ndarray, c: np.ndarray, tetids: np.ndarray | None = None, usenan: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates the curl of the field at the given points.
        """
        tetids = self._all_tet_ids
        vals = vals = MATHLIB.ned2_tet_interp_curl(np.array([xs, ys, zs]), field, self.mesh.tets, self.mesh.tris, self.mesh.edges, self.mesh.nodes, self.tet_to_field, c, tetids)
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f'Converted {n_zeros} to zeros.')
            vals = np.nan_to_num(vals)
        return vals
    
    def interpolate_index(self, xs: np.ndarray,
                        ys: np.ndarray,
                        zs: np.ndarray,
                        tetids: np.ndarray | None = None,
                        usenan: bool = True) -> np.ndarray:
        if tetids is None:
            tetids = self._all_tet_ids
        vals = MATHLIB.index_interp(np.array([xs, ys, zs]), self.mesh.tets, self.mesh.nodes, tetids)
        n_zeros = np.isnan(vals).sum()
        if not usenan and n_zeros > 0:
            logger.debug(f'Converted {n_zeros} to zeros.')
            vals[vals==-1]==0
        return vals