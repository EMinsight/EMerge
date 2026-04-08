from typing import Literal, get_args, Callable
from dataclasses import dataclass
from emsutil import Saveable
from ...const import MU0
import numpy as np


############################################################
#               ELEVATION AZIMUTH DEFINITION              #
############################################################

def fE_k(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float, phi: float, psi: float, k0: float, origin: tuple[float,float,float]) -> np.ndarray:
    kx = k0*np.cos(theta)*np.cos(phi)
    ky = k0*np.cos(theta)*np.sin(phi)
    kz = -k0*np.sin(theta)
    return np.array([kx, ky, kz])


def fE_EA(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float, phi: float, psi: float, k0: float, origin: tuple[float,float,float]) -> np.ndarray:
    kx = k0*np.cos(theta)*np.cos(phi)
    ky = k0*np.cos(theta)*np.sin(phi)
    kz = -k0*np.sin(theta)
    xp = x-origin[0]
    yp = y-origin[1]
    zp = z-origin[2]
    Phi = np.exp(-1j*(kx*xp + ky*yp + kz*zp))
    Ex = (np.sin(theta)*np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(psi))*Phi
    Ey = (np.sin(theta)*np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(psi))*Phi
    Ez = (np.cos(theta)*np.cos(psi))*Phi
    return np.array([Ex, Ey, Ez])

def fH_EA(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float, phi: float, psi: float, k0: float, origin: tuple[float,float,float]) -> np.ndarray:
    w0 = k0*299792458
    kx = k0*np.cos(theta)*np.cos(phi)
    ky = k0*np.cos(theta)*np.sin(phi)
    kz = -k0*np.sin(theta)
    xp = x-origin[0]
    yp = y-origin[1]
    zp = z-origin[2]
    Phi = np.exp(-1j*(kx*xp + ky*yp + kz*zp))
    Ex = (np.sin(theta)*np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(psi))*Phi
    Ey = (np.sin(theta)*np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(psi))*Phi
    Ez = (np.cos(theta)*np.cos(psi))*Phi

    CEx = -1j*(ky*Ez - kz*Ey) * 1j/(w0*MU0)
    CEy = -1j*(kz*Ex - kx*Ez) * 1j/(w0*MU0)
    CEz = -1j*(kx*Ey - ky*Ex) * 1j/(w0*MU0)
    return np.array([CEx, CEy, CEz])

def fEcurl_EA(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float, phi: float, psi: float, k0: float, origin: tuple[float,float,float]) -> np.ndarray:
    kx = k0*np.cos(theta)*np.cos(phi)
    ky = k0*np.cos(theta)*np.sin(phi)
    kz = -k0*np.sin(theta)
    xp = x-origin[0]
    yp = y-origin[1]
    zp = z-origin[2]
    Phi = np.exp(-1j*(kx*xp + ky*yp + kz*zp))
    Ex = (np.sin(theta)*np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(psi))*Phi
    Ey = (np.sin(theta)*np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(psi))*Phi
    Ez = (np.cos(theta)*np.cos(psi))*Phi

    CEx = 1j*(ky*Ez - kz*Ey)
    CEy = 1j*(kz*Ex - kx*Ez)
    CEz = 1j*(kx*Ey - ky*Ex)
    return np.array([CEx, CEy, CEz])


r2d = 180/np.pi
DEFINITIONS = Literal['EA']

@dataclass
class BackgroundField(Saveable):
    k0: float
    theta: float
    phi: float
    psi: float
    origin: tuple[float, float, float]
    E0: complex = 1.0 + 0.0j
    definition: DEFINITIONS = 'EA'

    def __post_init__(self):
        allowed = get_args(DEFINITIONS)
        if self.definition not in allowed:
            return ValueError(f'Cannot define a background field of definition {self.definition}. Please choose from: {allowed}')
        self.origin = tuple(self.origin)
    
    @property
    def angleset(self) -> tuple[float, float, float]:
        return (self.theta, self.phi, self.psi)
    
    def __str__(self) -> str:
        r2d = 180/np.pi
        return f'BackgroundField[amp={self.E0:.3f}V/m, k0={self.k0:.3f}, θ={self.theta*r2d:.1f}°, φ={self.phi*r2d:.1f}°, Ψ={self.psi*r2d:.1f}°, {self.definition}]'
    
    def Uinc(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return -1j*self.k0*self.E(x,y,z)
    
    def Uinc_curl(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return self.curlE(x,y,z)
    
    def k(self, x: np.ndarray, y: np.ndarray, z:np.ndarray) -> np.ndarray:
        if self.definition == 'EA':
            return fE_k(x,y,z,self.theta, self.phi, self.psi, self.k0, self.origin)
        
    def E(self, x: np.ndarray, y: np.ndarray, z:np.ndarray) -> np.ndarray:
        if self.definition == 'EA':
            return self.E0*fE_EA(x,y,z,self.theta, self.phi, self.psi, self.k0, self.origin)
        
    def H(self, x: np.ndarray, y: np.ndarray, z:np.ndarray) -> np.ndarray:
        if self.definition == 'EA':
            return self.E0*fH_EA(x,y,z,self.theta, self.phi, self.psi, self.k0, self.origin)
        
    def curlE(self, x: np.ndarray, y: np.ndarray, z:np.ndarray) -> np.ndarray:
        if self.definition == 'EA':
            return self.E0*fEcurl_EA(x,y,z,self.theta, self.phi, self.psi, self.k0, self.origin)
        
    def __hash__(self) -> int:
        return hash((self.theta, self.phi, self.psi, self.k0, self.origin, self.E0, self.definition))