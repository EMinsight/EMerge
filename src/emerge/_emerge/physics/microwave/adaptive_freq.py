#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-2.0-or-later
# Last Cleanup: 2025-01-01
"""
adaptive_frequency.py
---------------------
Copyright (c) 2014  Phil Reinhold
Copyright (c) 2025  Robert Fennis

This file is part of EMerge.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License **version 2**,
or (at your option) **any later version**, as published by the Free
Software Foundation.

This program is distributed in the hope that it will be useful,
but **WITHOUT ANY WARRANTY**; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program (see the file “COPYING” in the project root).
If not, see <https://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------
Origins
-------------------------------------------------------------------------------
This module is based on “Duplication of the vector fitting algorithm in
Python” (GPL-2.0) by Phil Reinhold, which re-implements the original MATLAB
code by **Bjørn Gustavsen**.  Key academic references:

  [1] B. Gustavsen & A. Semlyen, “Rational approximation of frequency
      domain responses by Vector Fitting”, *IEEE Trans. Power Delivery*,
      14 (3): 1052-1061, Jul 1999.

  [2] B. Gustavsen, “Improving the pole relocating properties of vector
      fitting”, *IEEE Trans. Power Delivery*, 21 (3): 1587-1592, Jul 2006.

  [3] D. Deschrijver *et al.*, “Macromodeling of Multiport Systems Using a
      Fast Implementation of the Vector Fitting Method”, *IEEE MWC Lett.*,
      18 (6): 383-385, Jun 2008.

-------------------------------------------------------------------------------
Modification history
-------------------------------------------------------------------------------
* 2025-06-23  Robert Fennis  fennisrobert@gmail.com
      - Refactored API with logic realized in the SparamModel class.

"""

# ty: ignore

import numpy as np
from typing import Literal
from loguru import logger

def cc(z):
    return z.conjugate()

def model(s, poles, residues, d, h):
    return sum([r/(s-p) for p, r in zip(poles, residues)]) + d + s*h

def vectfit_step(f: np.ndarray, s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """
    f = complex data to fit
    s = j*frequency
    poles = initial poles guess
        note: All complex poles must come in sequential complex conjugate pairs
    returns adjusted poles
    """
    N = len(poles)
    Ns = len(s)

    cindex = np.zeros(N)
    # cindex is:
    #   - 0 for real poles
    #   - 1 for the first of a complex-conjugate pair
    #   - 2 for the second of a cc pair
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    # First linear equation to solve. See Appendix A
    A = np.zeros((Ns, 2*N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise ValueError("cindex[%s] = %s" % (i, cindex[i]))

        A [:, N+2+i] = -A[:, i] * f

    A[:, N] = 1
    A[:, N+1] = s

    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((A.real, A.imag))
    b = np.concatenate((b.real, b.imag))
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=-1)

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = p.real, p.imag
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0

    H = A - np.outer(b, c)
    H = H.real
    new_poles = np.sort(np.linalg.eigvals(H))
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    return new_poles

# Dear gods of coding style, I sincerely apologize for the following copy/paste
def calculate_residues(f: np.ndarray, s: np.ndarray, poles: np.ndarray, rcond=-1) -> tuple[np.ndarray, float, float]:
    Ns = len(s)
    N = len(poles)

    cindex = np.zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert cc(poles[i]) == poles[i+1], ("Complex poles must come in conjugate pairs: %s, %s" % poles[i:i+1])
                cindex[i] = 1
            else:
                cindex[i] = 2

    # use the new poles to extract the residues
    A = np.zeros((Ns, N+2), dtype=np.complex128)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - cc(p))
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - cc(p))
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

    A[:, N] = 1
    A[:, N+1] = s
    # Solve Ax == b using pseudo-inverse
    b = f
    A = np.vstack((A.real, A.imag))
    b = np.concatenate((b.real, b.imag))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        logger.warning('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
        logger.warning('Cond(A)', cA)
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=rcond)

    # Recover complex values
    x = x.astype(np.complex128)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def vectfit_auto(f: np.ndarray, 
                 s: np.ndarray, 
                 n_poles: int = 10, 
                 n_iter: int = 10,
                 inc_real: bool = False, 
                 loss_ratio: float = 1e-2, 
                 rcond: int = -1, 
                 track_poles: bool = False) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    w = s.imag
    pole_locs = np.linspace(w[0], w[-1], n_poles+2)[1:-1]
    lr = loss_ratio
    poles = np.concatenate([[p*(-lr + 1j), p*(-lr - 1j)] for p in pole_locs])

    if inc_real:
        poles = np.concatenate((poles, [1]))

    poles_list = []
    for _ in range(n_iter):
        poles = vectfit_step(f, s, poles)
        poles_list.append(poles)

    residues, d, h = calculate_residues(f, s, poles, rcond=rcond)

    if track_poles:
        return poles, residues, d, h, np.array(poles_list)
    return poles, residues, d, h, np.array([])

def vectfit_auto_rescale(f: np.ndarray, s: np.ndarray, 
                         n_poles: int = 10, 
                         n_iter: int = 10,
                         inc_real: bool = False, 
                         loss_ratio: float = 1e-2, 
                         rcond: int = -1, 
                         track_poles: bool = False) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])
    poles_s, residues_s, d_s, h_s, tracked_poles = vectfit_auto(f / f_scale, 
                                                 s / s_scale,
                                                  n_poles=n_poles,
                                                  n_iter = n_iter,
                                                  inc_real=inc_real,
                                                  loss_ratio=loss_ratio,
                                                  rcond=rcond,
                                                  track_poles=track_poles)
    poles = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale
    return poles, residues, d, h, tracked_poles


class SparamModel:

    def __init__(self, 
                 frequencies: np.ndarray,
                 Sparam: np.ndarray,
                 n_poles: int | Literal['auto'] = 10,
                 inc_real: bool = False,
                 maxpoles: int = 40,
                 _warn: bool = True):
        self.f: np.ndarray = frequencies
        self.S: np.ndarray = Sparam

        s = 1j*frequencies

        if n_poles == 'auto':
            fdense = np.linspace(min(self.f), max(self.f), max(201, 10*self.f.shape[0]))
            success = False
            for nps in range(1,maxpoles):
                poles, residues, d, h, _ = vectfit_auto_rescale(Sparam, s, n_poles=nps, inc_real=inc_real)
                self.poles: np.ndarray = poles
                self.residues: np.ndarray = residues
                self.d = d
                self.h = h

                S = self(fdense)

                self.error = np.mean(np.abs(Sparam-self(self.f)))
                if all(np.abs(S) <= 1.0) and self.error < 1e-2:
                    logger.debug(f'Using {nps} poles.')
                    success = True
                    break
            if not success and _warn:
                logger.warning('Could not model S-parameters. Try simulating at a denser grid of poitns.')

        else:
            poles, residues, d, h, _ = vectfit_auto_rescale(Sparam, s, n_poles=n_poles, inc_real=inc_real)
            self.poles: np.ndarray = poles
            self.residues: np.ndarray = residues
            self.d = d
            self.h = h
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return model(1j*f, self.poles, self.residues, self.d, self.h)
