from functools import reduce

def merge_lists(lists):
    if not lists:
        return []
    return reduce(lambda a,b: a+b, lists)

class _DoFSet:

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
    
class DoFSet:

    def __init__(self, edge_dof_ids: list[int], face_dof_ids: list[int]):
        self.set2d = _DoFSet(edge_dof_ids, face_dof_ids, 2)
        self.set3d = _DoFSet(edge_dof_ids, face_dof_ids, 3)

    def __str__(self):
        return str(self.set2d) + '\n' + str(self.set3d)
    
DoF_FIRST = DoFSet([0,],[])
DoF_SAVAGE = DoFSet([0,1],[0,1])
DoF_VOLAKIS = DoFSet([0,2],[0,1])
DoF_COMPLETE = DoFSet([0,1,3],[2,3,4])
DoF_WEBB = DoFSet([0,1],[2,3])
if __name__=="__main__":
    print(DoF_SAVAGE)