
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

from functools import reduce
from enum import Enum
from emsutil import Saveable

def merge_lists(lists):
    if not lists:
        return []
    return reduce(lambda a,b: a+b, lists)

class _DoFSet(Saveable):

    def __init__(self, edge_dof_ids: list[int], face_dof_ids: list[int], dim: int):
        if dim == 2:
            ne = 3
            nf = 1
        elif dim == 3:
            ne = 6
            nf = 4
        self.codes = merge_lists([[64+idof,]*ne for idof in edge_dof_ids]) + merge_lists([[128+idof,]*nf for idof in face_dof_ids])
        self.n_edge_dofs: int = len(edge_dof_ids)
        self.n_face_dofs: int = len(face_dof_ids)
        self.n_edge_dofs_tot: int = len(edge_dof_ids)*ne
        self.n_face_dofs_tot: int = len(face_dof_ids)*nf
        self.n_dof_tot: int = self.n_edge_dofs_tot + self.n_face_dofs_tot
        self.n_node_dofs = 0
        self.n_vol_dofs = 0

    def __str__(self):
        line = ''
        for key,value in self.__dict__.items():
            line = line + f',{key} = {value}'
        return line
    
class DoFSet(Saveable):

    def __init__(self, edge_dof_ids: list[int], face_dof_ids: list[int], name: str = "UnnamedSet"):
        self.set2d: _DoFSet = _DoFSet(edge_dof_ids, face_dof_ids, 2)
        self.set3d: _DoFSet = _DoFSet(edge_dof_ids, face_dof_ids, 3)
        self.name: str = name

    def __str__(self):
        return self.name

class ElementSpace(Enum):
    FIRST_ORDER_MIXED = 0
    FIRST_ORDER_COMPLETE = 1
    SECOND_MIXED_SAVAGE = 2
    SECOND_MIXED_VOLAKIS = 3
    SECOND_MIXED_WEBB = 4
    SECOND_COMPLETE_WEBB = 5
    SECOND_COMPLETE_VOLAKIS = 6
    SECOND_MIXED_TEST = -1
    SECOND_COMPLETE_TEST = -2

    def get_set(self) -> DoFSet:
        match self:
            case ElementSpace.FIRST_ORDER_MIXED:
                return DoFSet([0], [], '1st Order Mixed')
            case ElementSpace.FIRST_ORDER_COMPLETE:
                return DoFSet([0, 1], [], '1st Order Complete')
            case ElementSpace.SECOND_MIXED_SAVAGE:
                return DoFSet([0, 1], [0, 1], '2nd Order Mixed (Savage)')
            case ElementSpace.SECOND_MIXED_VOLAKIS:
                return DoFSet([0, 2], [0, 1], '2nd Order Mixed (Volakis)')
            case ElementSpace.SECOND_MIXED_WEBB:
                return DoFSet([0, 1], [2, 3], '2nd Order Mixed (Webb)')
            case ElementSpace.SECOND_COMPLETE_WEBB:
                return DoFSet([0, 1, 3], [2, 3, 4], '2nd Order Complete (Webb)')
            case ElementSpace.SECOND_COMPLETE_VOLAKIS:
                return DoFSet([0, 2, 3], [0, 1, 4], '2nd Order Complete (Volakis)')
            case ElementSpace.SECOND_MIXED_TEST:
                return DoFSet([0, 1, 2, 3], [0,1,2,3,4], '2nd Order Mixed (TEST)')
            case ElementSpace.SECOND_COMPLETE_TEST:
                return DoFSet([0, 1, 3], [5, 6, 4], '2nd Order Complete (TEST)')
            case _:
                raise ValueError(f'No DoFSet defined for {self!r}')