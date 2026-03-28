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
from loguru import logger
from typing import Callable, Literal, Generator
from dataclasses import dataclass
from collections import defaultdict
from ...selection import Selection, FaceSelection
from ...cs import CoordinateSystem, Axis, GCS, _parse_axis
from ...coord import Line
from ...geometry import GeoSurface, GeoObject
from ...bc import BoundaryCondition, BoundaryConditionSet, Periodic
from ...periodic import PeriodicCell, HexCell, RectCell
from emsutil import Material, AIR, Saveable
from ...const import Z0, C0, EPS0, MU0
from ...logsettings import DEBUG_COLLECTOR



############################################################
#                        EXCEPTIONS                       #
############################################################

class BoundaryConditionException(Exception):
    pass

############################################################
#                     UTILITY FUNCTIONS                    #
############################################################

def _inner_product(function: Callable, x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: Axis) -> float:
            Exyz = function(x,y,z)
            return np.sum(Exyz[0,:]*ax.x + Exyz[1,:]*ax.y + Exyz[2,:]*ax.z)
        
############################################################
#                   MAIN BC MANAGER CLASS                  #
############################################################

class MWBoundaryConditionSet(BoundaryConditionSet):

    def __init__(self, periodic_cell: PeriodicCell | None):
        super().__init__()

        self.PEC: type[PEC] = self._construct_bc(PEC)
        self.PMC: type[PMC] = self._construct_bc(PMC)
        self.AbsorbingBoundary: type[AbsorbingBoundary] = self._construct_bc(AbsorbingBoundary)
        self.ModalPort: type[ModalPort] = self._construct_bc(ModalPort)
        self.LumpedPort: type[LumpedPort] = self._construct_bc(LumpedPort)
        self.LumpedElement: type[LumpedElement] = self._construct_bc(LumpedElement)
        self.SurfaceImpedance: type[SurfaceImpedance] = self._construct_bc(SurfaceImpedance)
        self.RectangularWaveguide: type[RectangularWaveguide] = self._construct_bc(RectangularWaveguide)
        self.Periodic: type[Periodic] = self._construct_bc(Periodic)
        self.FloquetPort: type[FloquetPort] = self._construct_bc(FloquetPort)
        self.UserDefinedPort: type[UserDefinedPort] = self._construct_bc(UserDefinedPort)
        self.CoaxPort: type[CoaxPort] = self._construct_bc(CoaxPort)
        self.ScatteredField: type[ScatteredField] = self._construct_bc(ScatteredField)

        self._cell: PeriodicCell | None = None

    def get_conductors(self) -> list[BoundaryCondition]:
        """Returns a list of all boundary conditions that ought to be considered as a "conductor" 
        for the purpose of modal analyses.

        Returns:
            list[BoundaryCondition]: All conductor like boundary conditions
        """
        bcs = self.oftype(PEC)
        for bc in self.oftype(SurfaceImpedance):
            if bc.sigma > 10.0:
                bcs.append(bc)
        return bcs
        
    def get_type(self, bctype: Literal['PEC','ModalPort','LumpedPort','PMC','LumpedElement','RectangularWaveguide','Periodic','FloquetPort','SurfaceImpedance']) -> FaceSelection:
        tags = []
        for bc in self.boundary_conditions:
            if bctype in str(bc.__class__):
                tags.extend(bc.selection.tags)
        return FaceSelection(tags)

    def floquet_port(self, poly: GeoSurface, port_number: int) -> FloquetPort:
        if self._cell is None:
            raise ValueError('Periodic cel must be defined for this simulation.')
        if isinstance(self._cell, RectCell):
            port = self.FloquetPort(poly, port_number)
            port.width = self._cell.width
            port.height = self._cell.height
            port.area = port.width*port.height
        elif isinstance(self._cell, HexCell):
            port = self.FloquetPort(poly, port_number)
            port.area = self._cell.area
        self._cell._ports.append(port)
        return port

    def iter_port_modes(self) -> Generator[tuple[PortBC, int | float, int], None, None]:
        """Iterates through all port + smatrix + port mode idex

        Yields:
            Generator[tuple[PortBC, int | float, int], None, None]: _description_
        """
        ports = sorted(self.oftype(PortBC), key=lambda x: x.port_number)
        for port in ports:
            for mat_index, mode_nr in port._iter_port_numbers():
                yield port, mat_index, mode_nr
        
    # Checks
    def _is_excited(self) -> bool:
        for bc in self.boundary_conditions:
            if not isinstance(bc, RobinBC):
                continue
            if bc._include_force:
                return True
        raise BoundaryConditionException('The simulation has no boundary conditions that insert energy. Make sure to include at least one Port into your simulation.')
    
    def _check_ports(self) -> None:
        numbers = []
        for bc in self.oftype(PortBC):
            numbers.append(bc.port_number)
        numbers = sorted(numbers)
        N = len(numbers)
        
        # check subsequent numbers
        if not all([pa==pb for pa,pb in zip(range(1,N+1), numbers)]):
            raise BoundaryConditionException(f'Port numbers are not subsequent integers. Values are {numbers} instead of {list(range(1,N+1))}')

############################################################
#                    BOUNDARY CONDITIONS                   #
############################################################


class PEC(BoundaryCondition, Saveable):
    _color: str = "#f70a80"
    _name: str = "PEC"
    _texture: str = "tex1.png"
    dim: int = 2
    def __init__(self,
                 face: FaceSelection | GeoSurface):
        """The general perfect electric conductor boundary condition.

        The physics compiler will by default always turn all exterior faces into a PEC.

        Args:
            face (FaceSelection | GeoSurface): The boundary surface
        """
        super().__init__(face)

class PMC(BoundaryCondition, Saveable):
    _color: str = "#0084ff"
    _name: str = "PMC"
    _texture: str = "tex4.png"
    dim: int = 2
    pass

class RobinBC(BoundaryCondition, Saveable):
    _color: str = "#e7c736"
    _name: str = "RobinBC"
    _texture: str = "tex5.png"
    _include_stiff: bool = False
    _include_mass: bool = False
    _include_force: bool = False
    _isabc: bool = False
    o2coeffs: tuple[float, float] = {'A': (1.0, -0.5),
                                    'B': (1.00023, -0.51555),
                                    'C': (1.03084, -0.73631),
                                    'D': (1.06103, -0.84883),
                                    'E': (1.12500, -1.00000)
                                    }
    dim: int = 2

    def __init__(self, selection: GeoSurface | Selection):
        """A Generalization of any boundary condition of the third kind (Robin).

        This should not be created directly. A robin boundary condition is the generalized type behind
        port boundaries, radiation boundaries etc. Since all boundary conditions of the thrid kind (Robin)
        are assembled the same, this class is used during assembly.

        Args:
            selection (GeoSurface | Selection): The boundary surface.
        """
        super().__init__(selection)
        self.v_integration: bool = False
        self.vintline: Line | None = None
    
    def get_basis(self) -> np.ndarray:
        raise NotImplementedError('This method is not implemented')
    
    def get_inv_basis(self) -> np.ndarray | None :
        raise NotImplementedError('This method is not implemented')
    
    def get_beta(self, k0: float) -> float:
        raise NotImplementedError('get_beta not implemented for Port class')
    
    def get_gamma(self, k0: float) -> complex:
        raise NotImplementedError('get_gamma not implemented for Port class')
    
    def get_Uinc(self, x_local: np.ndarray, y_local: np.ndarray, k0: float, mode_nr: int = 1) -> np.ndarray:
        raise NotImplementedError('get_Uinc not implemented for Port class')

class PortBC(RobinBC, Saveable):
    Zvac: float = Z0
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "PortBC"
    dim: int = 2
    
    def __init__(self, face: FaceSelection | GeoSurface):
        """(DO NOT USE) A generalization of the Port boundary condition.
        
        DO NOT USE THIS TO DEFINE PORTS. This class is only indeded for 
        class inheritance and type checking. 

        Args:
            face (FaceSelection | GeoSurface): The port face
        """
        super().__init__(face)
        self.port_number: int = -1
        self.cs: CoordinateSystem = GCS
        self.selected_mode: int = 0
        self.port_modes: dict[int, tuple[complex,...]] = {
            1: (1.0 + 0.0j,)
        }
        self.Z0: complex | float | None = None
        self.active: bool | None = False
        self.driven: bool = True
        self.power: float = 1.0
        self.vintline: list[Line] = []
        self.iintline: list[Line] = []
        
    @property
    def voltage(self) -> complex | None:
        return None
    
    def _gen_port_number(self, mode_nr: int) -> int | float:
        if len(self.port_modes) == 1:
            return self.port_number
        else:
            return float(f'{self.port_number}.{mode_nr}')
        
    def _iter_port_numbers(self) -> Generator[tuple[int | float, int], None, None]:
        """Iterates over all the mode/port numbers to include.

        Returns:
            _type_: _description_

        Yields:
            Generator[int | float, None, None]: _description_
        """
        if len(self.port_modes)==1:
            yield self.port_number, 1
        else:
            for number in sorted(self.port_modes.keys()):
                yield self._gen_port_number(number), number
                
    def _iter_modes(self, k0: float) -> Generator[tuple[int | float, Callable], None, None]:
        if len(self.port_modes) == 1:
            def Ufunc(x,y,z): 
                return self.get_Uinc(x,y,z,k0)
            yield self.port_number, Ufunc
        else:
            for mode_nr in sorted(self.port_modes.keys()):
                def Ufunc(x,y,z): 
                    return self.get_Uinc(x,y,z,k0, mode_nr=mode_nr)
                yield self._gen_port_number(mode_nr), Ufunc
               
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
    
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def portZ0(self, k0: float) -> complex | float | None:
        """Returns the port characteristic impedance given a phase constant

        Args:
            k0 (float): The phase constant

        Returns:
            complex: The port impedance
        """
        return self.Z0

    def modetype(self, k0: float) -> Literal['TEM','TE','TM']:
        return 'TEM'
    
    def Zmode(self, k0: float) -> float:
        if self.modetype(k0)=='TEM':
            return self.Zvac
        elif self.modetype(k0)=='TE':
            return k0*299792458/self.get_beta(k0) * MU0
        elif self.modetype(k0)=='TM':
            return self.get_beta(k0)/(k0*299792458*EPS0)
        else:
            raise ValueError(f'Port mode type should be TEM, TE or TM but instead is {self.modetype(k0)}')
    
    def _qmode(self, k0: float) -> float:
        """Computes a mode amplitude correction factor.
        The total output power of a port as a function of the field amplitude is not constant.
        For TE and TM modes the output power depends on the mode impedance. This factor corrects
        the mode output power to 1W by scaling the E-field appropriately.

        Args:
            k0 (float): The phase constant of the simulation

        Returns:
            float: The mode amplitude correction factor.
        """
        return np.sqrt(self.Zmode(k0)/Z0)
    
    @property
    def mode_number(self) -> int:
        return self.selected_mode + 1

    def get_beta(self, k0) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0
    
    def get_gamma(self, k0) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def port_mode_3d(self, 
                     xs: np.ndarray,
                     ys: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E',
                     mode_nr: int = 1) -> np.ndarray:
        raise NotImplementedError('port_mode_3d not implemented for Port class')
    
    def port_mode_3d_global(self, 
                                x_global: np.ndarray,
                                y_global: np.ndarray,
                                z_global: np.ndarray,
                                k0: float,
                                which: Literal['E','H'] = 'E',
                                mode_nr: int = 1) -> np.ndarray:
            xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
            Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, mode_nr=mode_nr)
            Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
            return np.array([Exg, Eyg, Ezg])

class AbsorbingBoundary(RobinBC, Saveable):

    _include_stiff: bool = True
    _include_mass: bool = True
    _include_force: bool = False
    _isabc: bool = True
    _color: str = "#1ce13d"
    _name: str = "AbsorbingBC"
    _texture: str = "tex3.png"
    dim: int = 2
    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 order: int = 2,
                 origin: tuple | None = None,
                 abctype: Literal['A','B','C','D','E'] = 'B'):
        """Creates an AbsorbingBoundary condition.

        Currently only a first order boundary condition is possible. Second order will be supported later.
        The absorbing boundary is effectively a port boundary condition (Robin) with an assumption on
        the out-of-plane phase constant. For now it always assumes the free-space propagation (normal).

        Args:
            face (FaceSelection | GeoSurface): The absorbing boundary face(s)
            order (int, optional): The order (only 1 is supported). Defaults to 1.
            origin (tuple, optional): The radiation origin. Defaults to None.
        """
        super().__init__(face)
        if origin is None:
            origin = (0., 0., 0.)
        self.order: int = order
        self.origin: tuple = origin
        self.cs: CoordinateSystem = GCS
        self.material: Material = AIR
        self.abctype: Literal['A','B','C','D','E']  = abctype
        
        
    def get_basis(self) -> np.ndarray:
        return np.eye(3)

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        f = k0*C0/(2*np.pi)
        if self.order==1:
            return 1j*k0*self.material.neff(f)
        
        return 1j*k0*self.o2coeffs[self.abctype][0]*self.material.neff(f)
      
@dataclass
class PortMode(Saveable):
    modefield: np.ndarray
    E_function: Callable
    H_function: Callable
    k0: float
    beta: float
    residual: float
    energy: float = 0
    norm_factor: float = 1
    freq: float = 0
    neff: float = 1
    Z0: float = 50.0
    polarity: float = 1.0
    modetype: Literal['TEM','TE','TM'] = 'TEM'

    def __post_init__(self):
        self.neff = self.beta/self.k0
        self.energy = np.mean(np.abs(self.modefield)**2)

    def __str__(self):
        return f'PortMode(k0={self.k0}, beta={self.beta}({self.neff:.3f}))'
    
    def set_power(self, power: complex) -> None:
        self.norm_factor = np.sqrt(1/np.abs(power))
        logger.info(f'Setting port mode amplitude to: {self.norm_factor:.2f} ')

class FloquetPort(PortBC, Saveable):
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "FloquetPort"
    dim: int = 2
    
    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 port_number: int,
                 cs: CoordinateSystem |  None = None,
                 power: float = 1.0,
                 er: float = 1.0):
        super().__init__(face)
        if cs is None:
            cs = GCS
        self.port_number: int= port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = 'TEM'
        self.cs: CoordinateSystem = cs
        self.scan_theta: float = 0
        self.scan_phi: float = 0
        self.port_modes: dict[int, tuple[complex, complex]] = {
            1: (1.0 + 0.0j, 0.0 + 0.0j),
            2: (0.0 + 0.0j, 1.0 + 0.0j)
        }
        self.area: float = 1
        self.width: float | None = None
        self.height: float | None = None

        if self.cs is None:
            self.cs = GCS

    def portZ0(self, k0: float | None = None) -> complex | float | None:
        return Z0

    def get_amplitude(self, k0: float) -> float:
        amplitude = np.sqrt(2*Z0*self.power/(self.area*np.cos(self.scan_theta)))
        return amplitude
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return k0*np.cos(self.scan_theta)
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float, mode_nr: int = 1) -> np.ndarray:
        return -2 * 1j * self.get_beta(k0) * self.port_mode_3d_global(x_global, y_global, z_global, k0, mode_nr=mode_nr)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E',
                     mode_nr: int = 1
                     ) -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        S, P = self.port_modes[mode_nr]
        kx = k0*np.sin(self.scan_theta)*np.cos(self.scan_phi)
        ky = k0*np.sin(self.scan_theta)*np.sin(self.scan_phi)
        #kz = k0*np.cos(self.scan_theta)
        phi = np.exp(-1j*(x_local*kx + y_local*ky))


        E0 = self.get_amplitude(k0)
        Ex = E0*(-S*np.sin(self.scan_phi) - P*np.cos(self.scan_theta)*np.cos(self.scan_phi))*phi
        Ey = E0*(S*np.cos(self.scan_phi) - P*np.cos(self.scan_theta)*np.sin(self.scan_phi))*phi
        Ez = E0*(-P*E0*np.sin(self.scan_theta))*phi
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E',
                            mode_nr: int = 1
                            ) -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        if self.cs is None:
            raise ValueError('No coordinate system is defined for this FloquetPort')
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0, mode_nr=mode_nr)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])
        
class ModalPort(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _texture: str = "tex5.png"
    _name: str = "ModalPort"
    dim: int = 2
    
    def __init__(self,
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 cs: CoordinateSystem | None = None,
                 power: float = 1,
                 modetype: Literal['TE','TM','TEM'] | None = None,
                 number_of_modes: int = 1,
                 mixed_materials: bool = False,
                 impedance_definition: Literal['PV','PI','VI'] = 'PV'):
        """Generes a ModalPort boundary condition for a port that requires eigenmode solutions for the mode.

        The boundary condition requires a FaceSelection (or GeoSurface related) object for the face and a port
        number. 
        If the face coordinate system is not provided a local coordinate system will be derived automatically
        by finding the plane that spans the face nodes with minimial out-of-plane error. 

        All modal ports require the execution of a .modal_analysis() by the physics class to define
        the port mode. 

        Args:
            face (FaceSelection, GeoSurface): The port mode face
            port_number (int): The port number as an integer
            cs (CoordinateSystem, optional): The local coordinate system of the port face. Defaults to None.
            power (float, optional): The radiated power. Defaults to 1.
            modetype (str[TE, TM, TEM], optional): Wether the mode should be considered as a TEM mode. Defaults to False
            mixed_materials (bool, optional): Wether the port consists of multiple different dielectrics. This requires
                A recalculation of the port mode at every frequency
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = False
        self.power: float = power
        self.alignment_vectors: list[Axis] = []

        self.selected_mode: int = 0
        self.available_modes: dict[float, list[PortMode]] = defaultdict(list)
        self.port_modes: dict[int | float, tuple[complex, ...]] = dict()
        for i in range(1, number_of_modes + 1):
            vec = [0.0 + 0.0j for _ in range(number_of_modes)]
            vec[i-1] = 1.0 + 0.0j
            self.port_modes[i] = vec
        self._desired_number_of_modes: int = number_of_modes
        self.forced_modetype: Literal['TE','TM','TEM'] | None = modetype
        self.mixed_materials: bool = mixed_materials
        self.initialized: bool = False
        self._first_k0: float | None = None
        self._last_k0: float | None = None
        
        self.plus_terminal: list[tuple[int, int]] = []
        self.minus_terminal: list[tuple[int, int]] = []
        self.N_mesh_tris: int = 50
        self.impedance_definition: Literal['PV','PI','VI'] = impedance_definition
        if cs is None:
            logger.info('Constructing coordinate system from port normal')
            self.cs = Axis(self.selection.normal).construct_cs() # type: ignore
        else:
            raise ValueError('No Coordinate System could be derived.')
        self._er: np.ndarray | None = None
        self._ur: np.ndarray | None = None
        
        self.vintline: list[Line] = []
        self.iintline: list[Line] = []
        
    def neff_estimate(self, k0: float) -> float:
        if len(self.available_modes)==0:
            return None
        else:
            
            return self.get_modes(k0)[0].neff
        
    @property
    def _size_constraint(self) -> float:
        area = self.selection.area
        return np.sqrt(area/self.N_mesh_tris*4/np.sqrt(3))
    
    def set_integration_line(self, c1: tuple[float, float, float], c2: tuple[float, float, float], N: int = 21) -> None:
        """Define the integration line start and end point

        Args:
            c1 (tuple[float, float, float]): The start coordinate
            c2 (tuple[float, float, float]): The end coordinate
            N (int, optional): The number of integration points. Defaults to 21.
        """
        self.vintline.append(Line.from_points(c1, c2, N))
    
    def reset(self) -> None:
        self.available_modes: dict[float, list[PortMode]] = defaultdict(list)
        self.initialized: bool = False
        self.plus_terminal: list[tuple[int, int]] = []
        self.minus_terminal: list[tuple[int, int]] = []
        
    def portZ0(self, k0: float) -> complex | float | None:
        return self.get_modes(k0)[0].Z0
    
    def modetype(self, k0: float) -> Literal['TEM','TE','TM']:
        return self.get_modes(k0)[0].modetype
    
    def align_modes(self, *axes: tuple | np.ndarray | Axis) -> None:
        """Set a reriees of Axis objects that define a sequence of mode field
        alignments.

        The modes will be sorted to maximize the inner product: |∬ E(x,y) · ax dS|
        
        Args:
            *axes (tuple, np.ndarray, Axis): The alignment vectors.
        """ 
        self.alignment_vectors = [_parse_axis(ax) for ax in axes]
    
    def _get_alignment_vector(self, index: int) -> np.ndarray | None:
        if len(self.alignment_vectors) > index:
            return self.alignment_vectors[index].np
        return None
    
    def set_terminals(self, positive: Selection | GeoObject | None = None,
                      negative: Selection | GeoObject | None = None,
                      ground: Selection | GeoObject | None = None) -> None:
        """Define which objects/faces/selection should be assigned the positive terminal
        and which one the negative terminal.
        
        The terminal assignment will be used to find an integration line for the impedance calculation.

        Note: Ground is currently unused.
        
        Args:
            positive (Selection | GeoObject | None, optional): The postive terminal. Defaults to None.
            negative (Selection | GeoObject | None, optional): The negative terminal. Defaults to None.
            ground (Selection | GeoObject | None, optional): _description_. Defaults to None.
        """
        if positive is not None:
            self.plus_terminal = positive.dimtags
        if negative is not None:
            self.minus_terminal = negative.dimtags
        
    @property
    def nmodes(self) -> int:
        if self._last_k0 is None:
            DEBUG_COLLECTOR.add_report('The modal analysis turned up with no solutions. This can be because:\n' 
                                       ' - You assigned the wrong materials to geometries.\n' + 
                                       ' - You simulate at a frequency that is too low.\n' + 
                                       ' - Your mode face is not appropriately supporting a modal solution.'
                                       )
            raise ValueError('ModalPort is not properly configured. No modes are defined.')
        return len(self.available_modes[self._last_k0])
    
    @property
    def voltage(self) -> complex:
        mode = self.get_modes(0)[0]
        return np.sqrt(mode.Z0)
    
    def _check_mode_betas(self) -> None:
        """Performs a check if the port mode vectors are properly configured
        """
        if len(self.port_modes) <= 1:
            return
        for k0, modes in self.available_modes.items():
            if len(modes)==1:
                continue
            betas = [np.real(mode.beta) for mode in modes]
            mean_beta = sum(betas)/len(betas)
            std_beta = (sum([(beta-mean_beta)**2 for beta in betas])/len(betas))**0.5
            str_betas = ','.join([f'{beta:.1f}' for beta in betas])
            if std_beta/mean_beta > 1e-2:
                logger.warning(f'A variation in port mode propagation constants (k0={k0:.1f}) of {std_beta/mean_beta*100:.2f}% is detected. Only multi mode ports with similar'+
                                f'propagation constants are suppored. Betas are {str_betas}')
                DEBUG_COLLECTOR.add_report('A variation in port mode propagation constants (k0={k0:.1f}) is detected. Only multi mode ports with similar'+
                                f'propagation constants are suppored. Betas are {str_betas}')
    def sort_modes(self) -> None:
        """Sorts the port modes based on total energy
        """
        
        if len(self.alignment_vectors) > 0:
            logger.trace(f'Sorting modes based on alignment vectors: {self.alignment_vectors}')
            X, Y, Z = self.selection.sample(5)
            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()
            for k0, modes in self.available_modes.items():
                logger.trace(f'Aligning modes for k0={k0:.3f} rad/m')
                new_modes = []
                for ax in self.alignment_vectors:
                    logger.trace(f'.mode vector {ax}')
                    integrals = [_inner_product(m.E_function, X, Y, Z, ax) for m in modes]
                    integral, opt_mode = sorted([pair for pair in zip(integrals, modes)], key=lambda x: abs(x[0]), reverse=True)[0]
                    opt_mode.polarity = np.sign(integral.real)
                    logger.trace(f'Optimal mode = {opt_mode} ({integral}), polarization alignment = {opt_mode.polarity}')
                    new_modes.append(opt_mode)
                    
                self.available_modes[k0] = new_modes
            return
        for k0, modes in self.available_modes.items():
            self.available_modes[k0] = sorted(modes, key=lambda m: m.beta, reverse=True)

    def get_modes(self, k0: float) -> list[PortMode]:
        """Returns a list of mode solution in the form of a PortMode object for a specific k0.

        Args:
            k0 (float): The propagation constant

        Returns:
            PortMode: The requested PortMode object
        """
        options = self.available_modes[min(self.available_modes.keys(), key=lambda k: abs(k - k0))]
        return options
    
    def global_field_function(self, k0: float = 0, which: Literal['E','H'] = 'E', mode_nr: int = 1) -> Callable:
        ''' The field function used to compute the E-field. 
        This field-function is defined in global coordinates (not local coordinates).'''
        modes = self.get_modes(k0)
        
        if which == 'E':
            def modef(x,y,z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp*(mode.norm_factor * self._qmode(k0) * mode.E_function(x,y,z)*mode.polarity)
                return out
            return modef
        else:
            def modef(x,y,z):
                amplitudes = self.port_modes[mode_nr]
                out = np.zeros((3, x.shape[0]), dtype=np.complex128)
                for amp, mode in zip(amplitudes, modes):
                    if amp == 0:
                        continue
                    out += amp*(mode.norm_factor * self._qmode(k0) * mode.H_function(x,y,z)*mode.polarity)
                return out
            return modef
    
    def clear_modes(self) -> None:
        """Clear all port mode data"""
        self.available_modes = defaultdict(list)
        self.initialized = False

    def add_mode(self, 
                 field: np.ndarray,
                 E_function: Callable,
                 H_function: Callable,
                 beta: float,
                 k0: float,
                 residual: float,
                 freq: float) -> PortMode | None:
        """Add a mode function to the ModalPort

        Args:
            field (np.ndarray): The field value array
            E_function (Callable): The E-field callable
            H_function (Callable): The H-field callable
            beta (float): The out-of-plane propagation constant 
            k0 (float): The free space phase constant
            residual (float): The solution residual
            freq (float): The frequency of the port mode

        Returns:
            PortMode: The port mode object.
        """
        mode = PortMode(field, E_function, H_function, k0, beta, residual, freq=freq)
        
        if mode.energy < 1e-4:
            logger.debug(f'Ignoring mode due to a low mode energy: {mode.energy}')
            return None
        
        self.available_modes[k0].append(mode)
        self.initialized = True

        self._last_k0 = k0
        if self._first_k0 is None:
            self._first_k0 = k0
        else:
            ref_field = self.get_modes(self._first_k0)[-1].modefield
            polarity = np.sign(np.sum(field*ref_field).real)
            logger.debug(f'Mode polarity = {polarity}')
            mode.polarity = polarity

        return mode

    def get_basis(self) -> np.ndarray:
        return self.cs._basis

    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def get_beta(self, k0: float) -> float:
        mode = self.get_modes(k0)[0]
        if self.forced_modetype=='TEM':
            beta = mode.beta/mode.k0 * k0
        else:
            freq = k0*299792458/(2*np.pi)
            beta = np.sqrt(mode.beta**2 + k0**2 * (1-((mode.freq/freq)**2)))
        return beta

    def get_gamma(self, k0: float) -> complex:
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0, mode_nr: int = 1) -> np.ndarray:
        
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0, mode_nr=mode_nr)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E',
                     mode_nr: int = 1) -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(x_local, y_local, 0*x_local)

        Egxyz = self.port_mode_3d_global(x_global,y_global,z_global, k0, which=which, mode_nr=mode_nr)
        
        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0,:], Egxyz[1,:], Egxyz[2,:])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self,
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E',
                            mode_nr: int = 1) -> np.ndarray:
        Ex, Ey, Ez = self.global_field_function(k0, which, mode_nr)(x_global,y_global,z_global)
        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

class RectangularWaveguide(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _name: str = "RectWG"
    _texture: str = "tex5.png"
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 mode: tuple[int, int] = (1,0),
                 er: float = 1.0,
                 cs: CoordinateSystem | None = None,
                 dims: tuple[float, float] | None = None,
                 power: float = 1):
        """Creates a rectangular waveguide as a port boundary condition.
        
        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            mode: (tuple[int, int], optional): The TE mode number. Defaults to (1,0).
            er: (float, optional): The Dielectric constant. Defaults to 1.0.
            cs (CoordinateSystem, optional): The local coordinate system. Defaults to None.
            dims (tuple[float, float], optional): The port face. Defaults to None.
            power (float): The port power. Default to 1.
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = 'TE'
        self.mode: tuple[int,int] = mode
        self.mode_axis: Axis | None = None
        self.er: float = er
        self._polarization: float = 1.0
        
        if dims is None:
            logger.debug(f" - Establishing RectangularWaveguide port face based on selection {self.selection}")
            cs, (width, height) = self.selection.rect_basis() # type: ignore
            self.cs = cs # type: ignore
            self.dims = (width, height)
            logger.debug(f' - Port CS: {self.cs}')
            logger.debug(f' - Detected port {self.port_number} size = {width*1000:.1f} mm x {height*1000:.1f} mm')
        else:
            self.dims = dims
            self.cs = cs
            
        if self.cs is None:
            logger.info(' - Constructing coordinate system from port normal')
            self.cs = Axis(self.selection.normal).construct_cs()
            logger.debug(f' - Port CS: {self.cs}')
        
    
    def align_modes(self, axis: Axis) -> None:
        """Defines the positive E-field direction for the fundamental TE mode.

        Args:
            axis (Axis): The alignment vector for the mode
        """

        self.mode_axis = axis
        self._polarization: float = float(np.sign(self.cs.yax.dot(self.mode_axis)))
        if self._polarization == 0.0:
            self._polarization = 1.0
        
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def portZ0(self, k0: float) -> complex:
        return k0*299792458 * MU0/self.get_beta(k0)

    def modetype(self, k0):
        return self.type
    
    def get_amplitude(self, k0: float) -> float:
        Zte = Z0
        amplitude= np.sqrt(self.power*4*Zte/(self.dims[0]*self.dims[1]))
        return amplitude

    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        width=self.dims[0]
        height=self.dims[1]
        beta = np.sqrt(self.er*k0**2 - (np.pi*self.mode[0]/width)**2 - (np.pi*self.mode[1]/height)**2)
        return beta
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float, mode_nr: int = None) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0, mode_nr=mode_nr)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     mode_nr: int = None,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''

        width = self.dims[0]
        height = self.dims[1]
        m, n= self.mode
        Ev = self._polarization*self.get_amplitude(k0)*np.cos(np.pi*m*(x_local)/width)*np.cos(np.pi*n*(y_local)/height)
        Eh = self._polarization*self.get_amplitude(k0)*np.sin(np.pi*m*(x_local)/width)*np.sin(np.pi*n*(y_local)/height)
        Ex = Eh
        Ey = Ev
        Ez = 0*Eh
        Exyz =  self._qmode(k0) * np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            mode_nr: int = None,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])

class CoaxPort(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1bd1c"
    _name: str = "Coax"
    _texture: str = "tex5.png"
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 rad_in_out: tuple[float, float],
                 cs: CoordinateSystem,
                 er: float = 1.0,
                 power: float = 1):
        """Creates a rectangular waveguide as a port boundary condition.
        
        Currently the Rectangular waveguide only supports TE0n modes. The mode field
        is derived analytically. The local face coordinate system and dimensions can be provided
        manually. If not provided the class will attempt to derive the local coordinate system and
        face dimensions itself. It always orients the longest edge along the local X-direction.
        The information on the derived coordiante system will be shown in the DEBUG level logs.

        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            mode: (tuple[int, int], optional): The TE mode number. Defaults to (1,0).
            er: (float, optional): The Dielectric constant. Defaults to 1.0.
            cs (CoordinateSystem, optional): The local coordinate system. Defaults to None.
            dims (tuple[float, float], optional): The port face. Defaults to None.
            power (float): The port power. Default to 1.
        """
        super().__init__(face)

        self.port_number: int= port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = 'TEM'
        self.er: float = er
        self._polarization: float = 1.0
        self.rad_in_out: tuple[float, float] = rad_in_out
        self.cs: CoordinateSystem = cs
        self.center: np.ndarray = self.cs.origin
        self.normal: Axis = self.cs.zax
        self.N_mesh_tris: int = 50
        # points = self.selection.points
    
    @property
    def _size_constraint(self) -> float:
        area = self.selection.area
        return np.sqrt(area/self.N_mesh_tris*4/np.sqrt(3))
    
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def portZ0(self, k0: float) -> complex:
        return k0*299792458 * MU0/self.get_beta(k0)

    def modetype(self, k0):
        return self.type
    
    def get_amplitude(self, k0: float) -> float:
        Zte = Z0
        amplitude= np.sqrt(self.power*4*Zte/(self.dims[0]*self.dims[1]))
        return amplitude

    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return np.sqrt(self.er)*k0
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float, mode_nr: int = None) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0, mode_nr=mode_nr)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     mode_nr: int = None,
                     which: Literal['E','H'] = 'E') -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''
        from ...const import Z0
        # Constants
        eta = Z0 / np.sqrt(self.er)
        
        Ri, Ro = self.rad_in_out
        
        pZ0 = (eta / (2 * np.pi)) * np.log(Ro / Ri)
        
        # 2. Relate Power to Voltage Amplitude: Pout = |V0|^2 / (2 * Z0)
        V0 = np.sqrt(2 * pZ0 * self.power)
        
        # 3. Geometric calculations
        rho = np.sqrt(x_local**2 + y_local**2)
        phi = np.arctan2(y_local, x_local)
        
        # 4. Magnitude of Eradial
        E_rho_mag = V0 / (rho * np.log(Ro / Ri))
        
        # 5. Account for Phase (Propagation along z)
        E_rho = E_rho_mag 
        
        # 6. Convert Cylindrical (rho) to Cartesian (x, y)
        Ex = (E_rho * np.cos(phi)).astype(np.complex128)
        Ey = (E_rho * np.sin(phi)).astype(np.complex128)
        Ez = 0.0*Ex # TEM mode has no longitudinal component
        
        Exyz =  self._qmode(k0) * np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            mode_nr: int = None,
                            which: Literal['E','H'] = 'E') -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex, Ey, Ez = self.port_mode_3d(xl, yl, k0)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])
    
def _f_zero(k0,x,y,z):
    "Zero field function"
    return np.zeros_like(x, dtype=np.complex128)

class UserDefinedPort(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#be9f11"
    _name: str = "UserDefined"
    _texture: str = "tex5.png"
    skip_fields = ('_fex','_fey','_fez','_fkz')
    dim: int = 2
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 Ex: Callable | None = None,
                 Ey: Callable | None = None,
                 Ez: Callable | None = None,
                 kz: Callable | None = None,
                 power: float = 1.0,
                 modetype: Literal['TEM','TE','TM'] = 'TEM',
                 cs: CoordinateSystem | None = None):
        """Creates a user defined port field
        
        The UserDefinedPort is defined based on user defined field callables. All undefined callables will default to 0 field or k0.
        
        All spatial field functions should be defined using the template:
        >>> def Ec(k0: float, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray
        >>>     return #shape like x
        
        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            Ex (Callable): The Ex(k0,x,y,z) field
            Ey (Callable): The Ey(k0,x,y,z) field
            Ez (Callable): The Ez(k0,x,y,z) field
            kz (Callable): The out of plane propagation constant kz(k0)
            power (float): The port output power
        """
        super().__init__(face)
        if cs is None:
            cs = GCS

        self.cs = cs
        self.port_number: int = port_number
        self.active: bool = False
        self.power: float = power
        self.type: str = 'TE'
        if Ex is None:
            Ex = _f_zero
        if Ey is None:
            Ey = _f_zero
        if Ez is None:
            Ez = _f_zero
        if kz is None:
            kz = lambda k0: k0

        self._fex: Callable = Ex
        self._fey: Callable = Ey
        self._fez: Callable = Ez
        self._fkz: Callable = kz
        self.type = modetype

    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def modetype(self, k0):
        return self.type
    
    def get_amplitude(self, k0: float) -> float:
        return np.sqrt(self.power)

    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return self._fkz(k0)
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*self.get_beta(k0)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float, mode_nr: int = 1) -> np.ndarray:
        return -2*1j*self.get_beta(k0)*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E',
                     mode_nr: int = 1) -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(x_local, y_local, 0*x_local)

        Egxyz = self.port_mode_3d_global(x_global,y_global,z_global,k0,which=which)
        
        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0,:], Egxyz[1,:], Egxyz[2,:])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E',
                            mode_nr: int = 1) -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
        Ex = self._fex(k0, x_global, y_global, z_global)
        Ey = self._fey(k0, x_global, y_global, z_global)
        Ez = self._fez(k0, x_global, y_global, z_global)
        Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])



class ScatteredField(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _isabc: bool = True
    _color: str = "#be9f11"
    _name: str = "UserDefined"
    _texture: str = "tex5.png"
    skip_fields = ('_fex','_fey','_fez','_fkz')
    dim: int = 2
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 power: float = 1.0,
                 cs: CoordinateSystem | None = None):
        """Creates a user defined port field
        
        The UserDefinedPort is defined based on user defined field callables. All undefined callables will default to 0 field or k0.
        
        All spatial field functions should be defined using the template:
        >>> def Ec(k0: float, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray
        >>>     return #shape like x
        
        Args:
            face (FaceSelection, GeoSurface): The port boundary face selection
            port_number (int): The port number
            Ex (Callable): The Ex(k0,x,y,z) field
            Ey (Callable): The Ey(k0,x,y,z) field
            Ez (Callable): The Ez(k0,x,y,z) field
            kz (Callable): The out of plane propagation constant kz(k0)
            power (float): The port output power
        """
        super().__init__(face)
        if cs is None:
            cs = GCS
        self.port_number: int = 1 
        self.cs = cs
        self.active: bool = False
        self.power: float = power
        self.type: str = 'TEM'
        self.order = 1
        self.abctype = 'A'
        self.radius: float = None
        self.driven: bool = True
        E0 = (Z0*2)**0.5
        self._fex: Callable = lambda k0,x,y,z: 0*x
        self._fey: Callable = lambda k0,x,y,z: 0*x 
        self._fez: Callable = lambda k0,x,y,z: 1j*2*k0*E0*np.exp(-1j*k0*x)
        
        self._fex_curl: Callable = lambda k0,x,y,z: 0*x
        self._fey_curl: Callable = lambda k0,x,y,z: 1j*2*k0*E0*np.exp(-1j*k0*x)
        self._fez_curl: Callable = lambda k0,x,y,z: 0*x
        
        def fnorm(x,y,z):
            R = ((x**2+y**2+z**2)**0.5)
            rx = x/R
            ry = y/R
            rz = z/R
            rhat =  np.array([rx, ry, rz])
            return rhat
        
        self._fnormal: Callable = fnorm
        self._fkz: Callable = lambda k0: k0
        self.type = 'TEM'

    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def modetype(self, k0):
        return self.type
    
    def get_amplitude(self, k0: float) -> float:
        return np.sqrt(self.power)

    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''
        return self._fkz(k0)
    
    def gen_Uinc(self, k0: float) -> tuple[Callable, Callable]:
        
        def Ufunc(x,y,z): 
            return self.get_Uinc(x,y,z,k0)
        def Ufunc_curl(x,y,z):
            return self.get_Uinc(x,y,z,k0, mode_nr=1, which='curl')
        
        return Ufunc, Ufunc_curl

    @property
    def curvature(self) -> float:
        if self.radius is None:
            return 0
        return 1/self.radius
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return (1j*self.get_beta(k0) + self.curvature)
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0: float, mode_nr: int = 1, which='E') -> np.ndarray:
        return self.port_mode_3d_global(x_global, y_global, z_global, k0, which=which)
        
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H','curl'] = 'E',
                     mode_nr: int = 1) -> np.ndarray:
        x_global, y_global, z_global = self.cs.in_global_cs(x_local, y_local, 0*x_local)

        Egxyz = self.port_mode_3d_global(x_global,y_global,z_global,k0,which=which)
        
        Ex, Ey, Ez = self.cs.in_local_basis(Egxyz[0,:], Egxyz[1,:], Egxyz[2,:])

        Exyz = np.array([Ex, Ey, Ez])
        return Exyz

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E',
                            mode_nr: int = 1) -> np.ndarray:
        '''Compute the port mode field for global xyz coordinates.'''
        if which == 'E':
            xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
            Ex = self._fex(k0, x_global, y_global, z_global)
            Ey = self._fey(k0, x_global, y_global, z_global)
            Ez = self._fez(k0, x_global, y_global, z_global)
            Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        elif which == 'curl':
            xl, yl, _ = self.cs.in_local_cs(x_global, y_global, z_global)
            Ex = self._fex_curl(k0, x_global, y_global, z_global)
            Ey = self._fey_curl(k0, x_global, y_global, z_global)
            Ez = self._fez_curl(k0, x_global, y_global, z_global)
            Exg, Eyg, Ezg = self.cs.in_global_basis(Ex, Ey, Ez)
        return np.array([Exg, Eyg, Ezg])
    

class LumpedPort(PortBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = True
    _color: str = "#e1851c"
    _name: str = "LumpedPort"
    _texture: str = "tex5.png"
    dim: int = 2
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 port_number: int, 
                 width: float | None = None,
                 height: float | None = None,
                 direction: Axis | tuple[float, float, float] | None = None,
                 power: float = 1,
                 Z0: float = 50):
        """Generates a lumped power boundary condition.
        
        The lumped port boundary condition assumes a uniform E-field along the "direction" axis.
        The port with and height must be provided manually in meters. The height is the size
        in the "direction" axis along which the potential is imposed. The width dimension
        is orthogonal to that. For a rectangular face its the width and for a cyllindrical face
        its the circumpherance.

        Args:
            face (FaceSelection, GeoSurface): The port surface
            port_number (int): The port number
            width (float): The port width (meters).
            height (float): The port height (meters).
            direction (Axis): The port direction as an Axis object (em.Axis(..) or em.ZAX)
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)

        if width is None:
            if not isinstance(face, GeoObject):
                raise ValueError(f'The width, height and direction must be defined. Information cannot be extracted from {face}')
            width, height, direction = face._data('width','height','vdir')
            if width is None or height is None or direction is None:
                raise ValueError(f'The width, height and direction could not be extracted from {face}')
        
        logger.debug(f'Lumped port: width={1000*width:.1f}mm, height={1000*height:.1f}mm, direction={direction}') # type: ignore
        self.port_number: int= port_number
        self.active: bool = False

        self.power: float = power
        self.Z0: float = Z0
        
        self.width: float = abs(width)
        self.height: float = abs(height) # type: ignore
        self.Vdirection: Axis = _parse_axis(direction) # type: ignore
        self.type = 'TEM'
        
        # self.cs = Axis(self.selection.normal).construct_cs()  # type: ignore
        self.cs = GCS
        self.vintline: list[Line] = []
        self.v_integration = True

        # Sanity checks
        if self.width > 0.5 or self.height > 0.5:
            DEBUG_COLLECTOR.add_report(f'{self}: A lumped port width/height larger than 0.5m has been detected: width={self.width:.3f}m. Height={self.height:.3f}.m. Perhaps you forgot a unit like mm, um, or mil')

    
    @property
    def _size_constraint(self) -> float:
        return min(self.width, self.height) / 4
    
    @property
    def surfZ(self) -> float:
        """The surface sheet impedance for the lumped port

        Returns:
            float: The surface sheet impedance
        """
        return self.Z0*self.width/self.height
    
    @property
    def voltage(self) -> float:
        """The Port voltage required for the provided output power (time average)

        Returns:
            float: The port voltage
        """
        return np.sqrt(2*self.power*self.Z0)
    
    def get_basis(self) -> np.ndarray:
        return self.cs._basis
        
    def get_inv_basis(self) -> np.ndarray:
        return self.cs._basis_inv
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0
    
    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*k0*Z0/self.surfZ
    
    def get_Uinc(self, x_global: np.ndarray, y_global: np.ndarray, z_global: np.ndarray, k0, mode_nr: int = 1) -> np.ndarray:
        Emag = -1j*2*k0 * self.voltage/self.height * (Z0/self.surfZ)
        return Emag*self.port_mode_3d_global(x_global, y_global, z_global, k0)
    
    def port_mode_3d(self, 
                     x_local: np.ndarray,
                     y_local: np.ndarray,
                     k0: float,
                     which: Literal['E','H'] = 'E',
                     mode_nr: int = 1) -> np.ndarray:
        ''' Compute the port mode E-field in local coordinates (XY) + Z out of plane.'''
        raise RuntimeError('This function should never be called in this context.')

    def port_mode_3d_global(self, 
                            x_global: np.ndarray,
                            y_global: np.ndarray,
                            z_global: np.ndarray,
                            k0: float,
                            which: Literal['E','H'] = 'E',
                            mode_nr: int = 1) -> np.ndarray:
        """Computes the port-mode field in global coordinates.

        The mode field will be evaluated at x,y,z coordinates but projected onto the local 2D coordinate system.
        Additionally, the "which" parameter may be used to request the H-field. This parameter is not always supported.

        Args:
            x_global (np.ndarray): The X-coordinate
            y_global (np.ndarray): The Y-coordinate
            z_global (np.ndarray): The Z-coordinate
            k0 (float): The free space propagation constant
            which (Literal["E","H"], optional): Which field to return. Defaults to 'E'.

        Returns:
            np.ndarray: The E-field in (3,N) indexing.
        """
        ON = np.ones_like(x_global)
        Ex, Ey, Ez = self.Vdirection.np
        return np.array([Ex*ON, Ey*ON, Ez*ON])

class LumpedElement(RobinBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False
    _color: str = "#e11c1c"
    _name: str = "LumpedElement"
    dim: int = 2
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 impedance_function: Callable | None = None,
                 width: float | None = None,
                 height: float | None = None,
                 ):
        """Generates a lumped power boundary condition.
        
        The lumped port boundary condition assumes a uniform E-field along the "direction" axis.
        The port with and height must be provided manually in meters. The height is the size
        in the "direction" axis along which the potential is imposed. The width dimension
        is orthogonal to that. For a rectangular face its the width and for a cyllindrical face
        its the circumpherance.

        Args:
            face (FaceSelection, GeoSurface): The port surface
            port_number (int): The port number
            width (float): The port width (meters).
            height (float): The port height (meters).
            direction (Axis): The port direction as an Axis object (em.Axis(..) or em.ZAX)
            active (bool, optional): Whether the port is active. Defaults to False.
            power (float, optional): The port output power. Defaults to 1.
            Z0 (float, optional): The port impedance. Defaults to 50.
        """
        super().__init__(face)

        if width is None:
            if not isinstance(face, GeoObject):
                raise ValueError(f'The width, height and direction must be defined. Information cannot be extracted from {face}')
            width, height, impedance_function = face._data('width','height','func')
            if width is None or height is None or impedance_function is None:
                raise ValueError(f'The width, height and impedance function could not be extracted from {face}')
        
        logger.debug(f'Lumped port: width={1000*width:.1f}mm, height={1000*height:.1f}mm') # type: ignore

        self.Z0: Callable = impedance_function # type: ignore
        self.width: float = width # type: ignore
        self.height: float = height # type: ignore

    def surfZ(self, k0: float) -> float:
        """The surface sheet impedance for the lumped Element

        Returns:
            float: The surface sheet impedance
        """
        Z0 = self.Z0(k0*299792458/(2*np.pi))*self.width/self.height
        return Z0
    
    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        return 1j*k0*Z0/self.surfZ(k0)
    
class SurfaceImpedance(RobinBC, Saveable):
    
    _include_stiff: bool = True
    _include_mass: bool = False
    _include_force: bool = False
    _color: str = "#49e8ed"
    _name: str = "SurfaceImpedance"
    skip_fields = ('_Zf',)
    dim: int = 2
    
    def __init__(self, 
                 face: FaceSelection | GeoSurface,
                 material: Material | None = None,
                 surface_conductance: float | None = None,
                 surface_roughness: float = 0,
                 thickness: float | None = None,
                 sr_model: Literal['Hammerstad-Jensen'] = 'Hammerstad-Jensen',
                 impedance_function: Callable | None = None,
                 ):
        """Generates a SurfaceImpedance bounary condition.

        The surface impedance model treats a 2D surface selection as a finite conductor. It is not
        intended to be used for dielectric materials.

        The surface resistivity is computed based on the material properties: σ, ε and μ. 

        The user may also supply the surface condutivity directly. 

        Optionally, a surface roughness in meters RMS may be supplied. In the current implementation
        The Hammersstad-Jensen model is used increasing the resistivity by a factor (1 + 2/π tan⁻¹(1.4(Δ/δ)²).

        Args:
            face (FaceSelection | GeoSurface): The face to apply this condition to.
            material (Material | None, optional): The matrial to assign. Defaults to None.
            surface_conductance (float | None, optional): The specific bulk conductivity to use. Defaults to None.
            surface_roughness (float, optional): The surface roughness. Defaults to 0.
            thickness (float | None, optional): The layer thickness. Defaults to None
            sr_model (Literal["Hammerstad-Jensen", optional): The surface roughness model. Defaults to 'Hammerstad-Jensen'.
            impedance_function (Callable, optional): A user defined surface impedance function as function of frequency.
        """
        super().__init__(face)

        self._material: Material | None = material
        self._mur: float | complex = 1.0
        self._epsr: float | complex = 1.0
        self.sigma: float = 0.0
        self.thickness: float | None = thickness
        
        if isinstance(face, GeoObject) and thickness is None:
            self.thickness = face._load('thickness')
            
        if material is not None:
            self.sigma = material.cond.scalar(1e9)
            self._mur = material.ur
            self._epsr = material.er
        
        if surface_conductance is not None:
            self.sigma = surface_conductance
        
        self._sr: float = surface_roughness
        self._sr_model: str = sr_model
        self._Zf: Callable | None = None
    
    def get_basis(self) -> np.ndarray | None:
        return None

    def get_inv_basis(self) -> np.ndarray | None:
        return None
    
    def get_beta(self, k0: float) -> float:
        ''' Return the out of plane propagation constant. βz.'''

        return k0

    def get_gamma(self, k0: float) -> complex:
        """Computes the γ-constant for matrix assembly. This constant is required for the Robin boundary condition.

        Args:
            k0 (float): The free space propagation constant.

        Returns:
            complex: The γ-constant
        """
        w0 = k0*C0
        f0 = w0/(2*np.pi)
        if self._Zf is not None:
            return 1j*k0*Z0/self._Zf(f0)
        
        sigma = self.sigma
        mur = self._material.ur.scalar(f0)
        er = self._material.er.scalar(f0)
        eps = EPS0*er
        mu = MU0*mur
        rho = 1/sigma
        d_skin = (2*rho/(w0*mu) * ((1+(w0*eps*rho)**2)**0.5 + rho*w0*eps))**0.5
        logger.debug(f'Computed skin depth δ={d_skin*1e6:.2}μm')
        R = (1+1j)*rho/d_skin
        #R = (w0*mu/(2*sigma))**0.5
        if self.thickness is not None:
            eps_c = eps - 1j * sigma / w0
            gamma_m = 1j * w0 * np.sqrt(mu*eps_c)
            R = R / np.tanh(gamma_m * self.thickness)
            logger.debug(f'Impedance scaler due to thickness: {1/ np.tanh(gamma_m * self.thickness) :.4f}')
        if self._sr_model=='Hammerstad-Jensen' and self._sr > 0.0:
            R = R * (1 + 2/np.pi * np.arctan(1.4*(self._sr/d_skin)**2))
        return 1j*k0*Z0/R