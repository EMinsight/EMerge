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
from pathlib import Path
from typing import Callable
import re
from loguru import logger
from scipy.interpolate import interp1d

_FUNIT = {
    'hz': 1,
    'khz': 1000,
    'mhz': 1e6,
    'ghz': 1e9,
}
_NPORTS = {
    '.s1p': 1,
    '.s2p': 2,
    '.s3p': 3,
    '.s4p': 4,
    '.snp': None
}


############################################################
#                         FUNCTIONS                        #
############################################################


def _ma_ri(mag: float, angle: float) -> complex:
    """Converts magnitude and angle to complex number.

    Args:
        mag (float): The magnitude of the complex number.
        angle (float): The angle of the complex number.

    Returns:
        complex: The complex number.
    """    
    return mag * np.exp(1j * np.radians(angle))

def _db_ri(db: float, angle: float) -> complex:
    """Converts dB and angle to complex number.

    Args:
        db (float): The magnitude in dB.
        angle (float): The angle in degrees.

    Returns:
        complex: The complex number.
    """    
    return 10**(db/20) * np.exp(1j * np.radians(angle))

def _ri_ri(real: float, imag: float) -> complex:
    """Converts real and imaginary parts to complex number.

    Args:
        real (float): The real part of the complex number.
        imag (float): The imaginary part of the complex number.

    Returns:
        complex: The complex number.
    """    
    return real + 1j * imag

_DATAMAP = {
    'ma': _ma_ri,
    'db': _db_ri,
    'ri': _ri_ri
}


############################################################
#                        MAIN CLASS                       #
############################################################


class TouchstoneData:
    """A touchstone file importer for S-parameter data."

    The TouchstoneData class can be used to quickly import Touchstone files into your model to plot the data in your graphs.
    
    Args:
        filename (str): The filename of the touchstone file.
        n_ports (int, optional): The number of ports. Defaults to None.
        ignore_extension (bool, optional): Ignore the file extension. Defaults to False.
        interp_type (str, optional): The interpolation type. Defaults to 'cubic'.

    Example:
        ```python
        touchstone = ToustoneData('touchstone.s2p')
        touchstone.renormalize(75)
        ```

    """
    def __init__(self, filename: str,
                 n_ports: int = None,
                 ignore_extension: bool = False,
                 interp_type: str = 'cubic'):
        super().__init__()
        path = Path(filename)
        if not path.exists():
            raise ValueError(f'The provided filename {path} does not exist.')

        if path.suffix not in ('.s1p','.s2p','.s3p','.s4p','.snp','.ts') and not ignore_extension:
            raise ValueError(f'Can only open touchstone files, files with {path.suffix} are not a valid touchstone extension.')
        
        with open(filename,'r') as f:
            self._lines = f.read().split('\n')
        
        self.Sdata = None

        self.freq_unit: float = 1e9
        self.param_type: str = 's'
        self.data_type: str = 'ma'
        self._converter: Callable = _DATAMAP['ma']
        self.refimp: float = 50
        self.n_ports: int = None
        self.interp_type: str = interp_type
        self.Sdata: np.ndarray = None
        self.Fdata: np.ndarray = None

        if self.param_type != 's':
            raise ValueError('Only S-parameter touchstone files are supported as of this moment.')

        self._parse_touchstone()
    
    @property 
    def f(self) -> np.ndarray:
        return self.Fdata
    
    def summarize_data(self):
        # ALLOWED PRINT
        print(f'Frequency unit: {self.freq_unit}')
        # ALLOWED PRINT
        print(f'Parameter type: {self.param_type}')
        # ALLOWED PRINT
        print(f'Data type: {self.data_type}')
        # ALLOWED PRINT
        print(f'Converter: {self._converter}')
        # ALLOWED PRINT
        print(f'Reference impedance: {self.refimp}')
        # ALLOWED PRINT
        print(f'Number of ports: {self.n_ports}')

    def _parse_touchstone(self) -> None:
        s_data = []
        f_data = []
        s_collector = []
        freq = None
        for line in self._lines:
            if not line:
                continue
            
            stripped = line.strip().lower()
            # Comment line
            if stripped[0]=='!':
                continue
            
            # Options line
            if stripped[0]=='#':
                funit = re.findall(r'(hz|khz|mhz|ghz)', stripped)
                param_type = re.findall(r'([syzgh])', stripped)
                data_type = re.findall(r'(ma|db|ri)', stripped)
                refimp = re.findall(r'r\s+(\d+)', stripped)

                if funit:
                    self.freq_unit = _FUNIT[funit[0]]
                if param_type:
                    self.param_type = param_type[0]
                if data_type:
                    self.data_type = data_type[0]
                    self._converter = _DATAMAP[data_type[0]]
                if refimp:
                    self.refimp = float(refimp[0])
            
                continue
            
            # Version 2.0 Keyword line:
            if stripped[0]=='[':
                logger.warning('Version 2.0 touchstone files are not supported yet and keywords will be ignored')
                continue
            # Data line
            # remove the exclamation mark plus all symbols behind it if it occurs
            line = line.split('!')[0]

            nums = [float(x) for x in line.split(' ') if x]

            if len(nums)%2 == 1:
                if len(s_collector) > 0:
                    s_data.append(s_collector)
                    f_data.append(freq)
                s_collector = nums[1:]
                freq = nums[0]
            else:
                s_collector.extend(nums)
            
        s_data.append(s_collector)
        f_data.append(freq)

        ssample = s_data[0]
        self.n_ports = int(np.sqrt(len(ssample) // 2))
        self.n_nodes = self.n_ports

        self.n_frequencies = len(f_data)
        self.Fdata = np.array(f_data) * self.freq_unit
        self.Sdata = np.zeros((self.n_frequencies, self.n_ports, self.n_ports), dtype=complex)

        for i, s in enumerate(s_data):
            sdata = [self._converter(s1,s2) for s1,s2 in zip(s[::2], s[1::2])]
            Smat = np.array(sdata).reshape(self.n_ports, self.n_ports)
            if self.n_ports == 2:
                Smat = Smat.T
            self.Sdata[i,:,:] = Smat

    def renormalize(self, new_impedance: float) -> TouchstoneData:
        """ Renormalize the touchstone file to a new impedance.

        Args:
            new_impedance (float): The new impedance to normalize the touchstone file to.

        Returns:
            FileBasedNPort: The renormalized touchstone file subcircuit.
        """        
        Zold = self.refimp
        Znew = new_impedance
        A = np.eye(self.n_ports, self.n_ports) * np.sqrt(Znew/Zold)*(1/(Znew+Zold))
        R = np.eye(self.n_ports, self.n_ports) * ((Znew-Zold)/(Znew+Zold))
        iA = np.linalg.pinv(A)
        E = np.eye(self.n_ports, self.n_ports)
        for iif in range(self.n_frequencies):
            S = self.Sdata[iif,:,:]
            self.Sdata[iif,:,:] = iA @ (S - R) @ np.linalg.pinv(E - R @ S) @ A
        self.refimp = new_impedance
        return self
        
    def S(self, i: int, j: int) -> np.ndarray:
        """Return the S-parameter array corresponding to the given port number

        Args:
            i (int): The output port
            j (int): The input Port

        Returns:
            np.ndarray: The S-parameter array
        """
        return self.Sdata[:, i-1, j-1].squeeze()
    
    def RemoveSP(self, touchstoneClass: 'TouchstoneData', port: int) -> 'TouchstoneData':
        """De-embed a 2-port fixture network from this touchstone measurement.

        Supports both 1-port and 2-port DUTs:
        - 1-port: solves Γ_dut from Γ_meas = S11 + S12*S21*Γ_dut / (1 - S22*Γ_dut)
        - 2-port: uses T-parameter de-embedding T_dut = inv(T_fix) @ T_meas

        Args:
            touchstoneClass (TouchstoneData): A 2-port touchstone object representing
                the fixture to de-embed (e.g. a female-female adapter).
            port (int): The port number (1-based) from which to de-embed.

        Returns:
            TouchstoneData: self, with S-parameters updated after de-embedding.
        """
        if touchstoneClass.n_ports != 2:
            raise ValueError('The fixture touchstone data must be a 2-port (s2p) file.')

        if port < 1 or port > self.n_ports:
            raise ValueError(f'Port {port} is out of range for a {self.n_ports}-port network.')

        # Interpolate fixture data onto our frequency axis
        fixture_S = np.zeros((self.n_frequencies, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                s_param = touchstoneClass.S(i + 1, j + 1)
                f_fixture = touchstoneClass.f
                re_interp = interp1d(f_fixture, s_param.real,
                                    kind=self.interp_type, fill_value='extrapolate')
                im_interp = interp1d(f_fixture, s_param.imag,
                                    kind=self.interp_type, fill_value='extrapolate')
                fixture_S[:, i, j] = re_interp(self.Fdata) + 1j * im_interp(self.Fdata)

        if self.n_ports == 1:
            # 1-port de-embedding using signal flow graph solution
            for iif in range(self.n_frequencies):
                S11f = fixture_S[iif, 0, 0]
                S12f = fixture_S[iif, 0, 1]
                S21f = fixture_S[iif, 1, 0]
                S22f = fixture_S[iif, 1, 1]

                gamma_meas = self.Sdata[iif, 0, 0]

                denom = S22f * (gamma_meas - S11f) + S12f * S21f
                if abs(denom) < 1e-30:
                    raise ValueError(f'De-embedding denominator near zero at freq index {iif}.')

                gamma_dut = (gamma_meas - S11f) / denom
                self.Sdata[iif, 0, 0] = gamma_dut

        elif self.n_ports == 2:
            for iif in range(self.n_frequencies):
                # Convert fixture S to T-parameters
                S = fixture_S[iif, :, :]
                if abs(S[1, 0]) < 1e-30:
                    raise ValueError(f'Fixture S21 near zero at freq index {iif}.')
                T_fix = np.array([
                    [-(S[0,0]*S[1,1] - S[0,1]*S[1,0]) / S[1,0], S[0,0] / S[1,0]],
                    [-S[1,1] / S[1,0], 1.0 / S[1,0]]
                ])

                # Convert measured S to T-parameters
                Sm = self.Sdata[iif, :, :]
                if abs(Sm[1, 0]) < 1e-30:
                    raise ValueError(f'Measured S21 near zero at freq index {iif}.')
                T_meas = np.array([
                    [-(Sm[0,0]*Sm[1,1] - Sm[0,1]*Sm[1,0]) / Sm[1,0], Sm[0,0] / Sm[1,0]],
                    [-Sm[1,1] / Sm[1,0], 1.0 / Sm[1,0]]
                ])

                # De-embed from the appropriate port
                if port == 1:
                    T_result = np.linalg.inv(T_fix) @ T_meas
                else:
                    T_result = T_meas @ np.linalg.inv(T_fix)

                # Convert T back to S
                if abs(T_result[0, 0]) < 1e-30:
                    raise ValueError(f'T11 near zero at freq index {iif}.')
                self.Sdata[iif, :, :] = np.array([
                    [T_result[0,1] / T_result[0,0],
                    (T_result[0,0]*T_result[1,1] - T_result[0,1]*T_result[1,0]) / T_result[0,0]],
                    [1.0 / T_result[0,0],
                    -T_result[1,0] / T_result[0,0]]
                ])
        else:
            raise ValueError(f'De-embedding is only supported for 1-port and 2-port DUTs, '
                            f'not {self.n_ports}-port.')

        return self