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

class DoFSplitException(Exception):
    pass

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

        self.tri_to_field_os: np.ndarray | None = None
        self.edge_to_field_os: np.ndarray | None = None
        ##
        self._field: np.ndarray | None = None
        self.n_tet_dofs = 20
        self.n_tri_dofs = 8
        self._all_tet_ids = np.arange(self.ntets)

        self._dof_mapping: dict[int,int] = dict()
        self._partitioned: bool = False
        self.diagnose()

        rows, cols = self.empty_tri_rowcol()
        self._rows = rows
        self._cols = cols
    
    def diagnose(self):
        visited_field = np.zeros((self.n_field,), dtype=np.bool)
        for i in range(self.mesh.n_tets):
            visited_field[self.tet_to_field[:,i]] = True
        assert np.all(visited_field)
    
    def partition_dof(self, list_tags: list[list[int]]) -> None:
        """Partition_DOF splits the degrees of freedom for an input list of face tags."""
        
        # If no tri_to_field mapping for side 2 is managed, do it now
        if self._partitioned:
            return
        
        if self.tri_to_field_os is None:
            self.tri_to_field_os = self.tri_to_field.copy()
            self.edge_to_field_os = self.edge_to_field.copy()
            
        tags: list[int] = []
        for sublist in list_tags:
            tags.extend(sublist)
        for tag in tags:
            self._partition_dof_face(tag)
            
        self._partitioned = True

    def _partition_dof_face(self, tag: int) -> None:
        """Splits DOFs on a specific face tag between two volumes.

        Splitting means that one domain gets one set of degrees of freedom
        and the other a copied set of degrees of freedom.

        the mapping between them is stored in self._dof_mapping

        """

        tris = self.mesh.get_triangles([tag])
        linked_tets = self.mesh.tri_to_tet[:, tris]

        # Check exterior
        if np.any(linked_tets == self.mesh._MISSING_ID):
            raise DoFSplitException(
                f"Cannot split DoF for boundary {tag} because its on the exterior "
                f"of the simulation domain. Check the ThermalContact boundary assignment."
            )

        # Check same-domain
        # Build tet -> vol mapping as array (faster than dict)
        tet_to_volume = np.full(self.ntets, -1, dtype=np.int64)
        for vtag, tet_ids in self.mesh.vtag_to_tet.items():
            tet_to_volume[tet_ids] = vtag

        vol_a_of_tri = tet_to_volume[linked_tets[0, :]]
        vol_b_of_tri = tet_to_volume[linked_tets[1, :]]

        connected_volumes = set(vol_a_of_tri) | set(vol_b_of_tri)
        connected_volumes = sorted(list(connected_volumes))
        
        # Boundary DOFs
        boundary_tris = self.mesh.get_triangles(tag)
        boundary_edges = self.mesh.get_edges(tag)
        boundary_nodes = self.mesh.get_nodes(tag)
        
        # figure out which edges are counted only ones
        edge_counter = {ie: 0 for ie in boundary_edges}
        
        # base normal:
        normal = self.mesh.ftag_to_normal[tag]
        
        for itri in tris:
            e1, e2, e3 = self.mesh.tri_to_edge[:,itri]
            edge_counter[e1] += 1
            edge_counter[e2] += 1
            edge_counter[e3] += 1
        
        split_edges = set()
        
        for itri in tris:
            
            trinorm = self.mesh.get_normal(itri)
            normdot = np.abs(np.sum(trinorm * normal))
                
            tet1 = self.mesh.tri_to_tet[0,itri]
            tet2 = self.mesh.tri_to_tet[1,itri]
            vol1 = tet_to_volume[tet1] 
            vol2 = tet_to_volume[tet2]
            
            e1, e2, e3 = self.mesh.tri_to_edge[:,itri]
            for ie in (e1, e2, e3):
                if edge_counter[ie] == 2 and ie not in split_edges:
                    NDOF = self.n_field
                    self.n_field += 2
                    idof1, idof2 = self.edge_to_field[:, ie]
                    self._dof_mapping[idof1] = NDOF
                    self._dof_mapping[idof2] = NDOF + 1
                    self.edge_to_field_os[0,ie] = NDOF
                    self.edge_to_field_os[1,ie] = NDOF + 1
            
            _,_,_,idof1, _,_,_,idof2 = self.tri_to_field[:, itri]
            NDOF = self.n_field
            self.n_field += 2
            self._dof_mapping[idof1] = NDOF
            self._dof_mapping[idof2] = NDOF + 1
            
            pick = tet1
            if normdot > 0:
                pick = tet2
            
            for i, idof in enumerate(self.tet_to_field[:,pick]):
                self.tet_to_field[i,pick] = self._dof_mapping.get(idof, idof)

            for i, idof in enumerate(self.tri_to_field[:,itri]):
                self.tri_to_field_os[i,itri] = self._dof_mapping.get(idof, idof)        
            
            
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