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

# Last Cleanup: 2026-03-04
from ...mesher import Mesher
from emsutil import Material
from ...mesh3d import Mesh3D
from ...coord import Line
from ...geometry import GeoSurface, GeoVolume
from ...elements.leg2 import Legrange2
from ...solver import DEFAULT_ROUTINE, SolveRoutine
from ...system import called_from_main_function
from ...selection import FaceSelection
from ...settings import Settings
from ...simstate import SimState
from ...logsettings import DEBUG_COLLECTOR

from .heatconduction_bc import HCBoundaryConditionSet, BlackBodyRadiation
from .heatconduction_data import HCData
from .assembly.assembler import Assembler
from .simjob import SimJob

from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Callable, Literal, Any
import multiprocessing as mp
from cmath import sqrt as csqrt
from itertools import product
import numpy as np
import threading
import time
from collections import defaultdict
import psutil

from ..physics_generic import SimulationError, GenericPhysics3D

############################################################
#                 MULTI PROCESSING FUNCTION                #
############################################################


def run_job_multi(job: SimJob) -> SimJob:
    """The job launcher for Multi-Processing environements

    Args:
        job (SimJob): The Simulation Job

    Returns:
        SimJob: The solved SimJob
    """
    nr = int(mp.current_process().name.split("-")[1])
    routine = DEFAULT_ROUTINE._configure_routine("MP", proc_nr=nr)
    A, bmat, ids, aux = job.get_Ab()
    solution, report = routine.solve(A, bmat, ids, id=job.id)
    report.add(**aux)
    job.submit_solution(solution, report)
    return job


def _dimstring(data: list[float] | np.ndarray) -> str:
    """A String formatter for dimensions in millimeters

    Args:
        data (list[float]): The list of floating point dimensions

    Returns:
        str: The formatted string
    """
    return "(" + ", ".join([f"{x * 1000:.1f}mm" for x in data]) + ")"


class HeatConduction3D(GenericPhysics3D):
    def __init__(
        self, state: SimState, mesher: Mesher, settings: Settings, order: int = 2
    ):

        self._settings: Settings = settings

        self.T_initial_K: float = 293.15
        self.order: int = order  # Discretization order. Is always 2 (legacy)

        self.mesher: Mesher = mesher  # A reference to the Mesher object
        self._state: SimState = state  # A reference to the Simulation stat object

        self.assembler: Assembler = Assembler(self._settings)  # The assembler class
        self.bc: HCBoundaryConditionSet = HCBoundaryConditionSet(
            None
        )  # The boundary condition set class.
        self.basis: Legrange2 | None = None
        self.solveroutine: SolveRoutine = DEFAULT_ROUTINE
        self.cache_matrices: bool = True

        ## States
        self._bc_initialized: bool = False
        self._simstart: float = 0.0
        self._simend: float = 0.0
        self._container: dict[str, Any] = dict()
        self._completed: bool = False

    @property
    def _params(self) -> dict[str, float]:
        return self._state.params

    @property
    def mesh(self) -> Mesh3D:
        return self._state.mesh

    @property
    def data(self) -> HCData:
        return self._state.data.hc

    def _check_meshed(self) -> None:
        """Checks if a mesh is generated"""
        if not self.mesh.defined:
            raise SimulationError("Mesh is not defined. Call generate_mesh() first!")

    def _initialize_field(self):
        """Initializes the physics basis to the correct FEMBasis object.

        Currently it defaults to Nedelec2. Mixed basis are used for modal analysis.
        This function does not have to be called by the user. Its automatically invoked.
        """
        if self.basis is not None:
            return
        if self.order == 1:
            raise NotImplementedError("Legrange order 1 is currently not supported")
        elif self.order == 2:
            self.basis = Legrange2(self.mesh)

    def _initialize_bc_data(self):
        """Initializes auxilliary required boundary condition information before running simulations."""
        logger.debug("Initializing boundary conditions")
        # Removes non-assigned boundary conditions.
        # This happens for example if the initial boundary PEC gets overwritten.
        self.bc.cleanup()
        self.bc._selections_post_boolean_fragment()

    def _check_physics(self) -> None:
        """Executes a physics check before a simulation can be run.

        Raises:
           BoundaryConditionError: If any boundary condition is not setup correctly

        """

        pass

    def set_initial_temperature(self, temp_K: float) -> None:
        """Define the initial temperature of the simulation.

        This is used by the black body radiation boundary condition

        Args:
            temp_K (float): The temperature in Kelvin
        """
        self.T_initial_K = temp_K

    def run_steady_state(
        self, direct_solver: bool = True, preconditioner: bool = False
    ):

        self._completed = False
        self._simstart = time.time()

        # --------------------------------------------------------------------
        # Local Variables
        # --------------------------------------------------------------------

        material_set: tuple[np.ndarray,] = []

        # --------------------------------------------------------------------
        # Checks
        # --------------------------------------------------------------------

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        # --------------------------------------------------------------------
        # Material Assignments
        # --------------------------------------------------------------------

        logger.debug("Resolving material assingments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        # --------------------------------------------------------------------
        # Initializing solve functions
        # --------------------------------------------------------------------

        thread_local = None

        def get_routine() -> SolveRoutine:
            if not hasattr(thread_local, "routine"):
                worker_nr = int(threading.current_thread().name.split("_")[1]) + 1
                thread_local.routine = self.solveroutine.duplicate()._configure_routine(
                    "MT", thread_nr=worker_nr
                )
            return thread_local.routine

        # Single threaded does not need this routine as the class's own routine can be used.
        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, id=job.id, direct=direct_solver
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        # Assemble the FEM problem
        job, mats = self.assembler.assemble_stationary_matrix(
            self.basis, materials, self.bc.boundary_conditions, self.T_initial_K
        )
        material_set = mats

        # Solver configuration
        if preconditioner:
            self.solveroutine.use_preconditioner = True
        # Finally solve the problems for each frequency individually.
        logger.info("Starting single threaded solve.")
        job = run_job_single(job)

        self.solveroutine.reset()

        logger.info("Solving complete")

        # --------------------------------------------------------------------
        # Writing solve reports
        # --------------------------------------------------------------------

        self.data.setreport(job.reports, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        # --------------------------------------------------------------------
        # Post Processing
        # --------------------------------------------------------------------

        self._post_process(job, material_set)
        self._completed = True
        return self.data

    def run_steady_state_nl(
        self,
        direct_solver: bool = True,
        preconditioner: bool = False,
        max_nonlinear_iter: int = 50,
        nonlinear_tol: float = 1e-6,
        anderson_m: int = 5,
    ):

        self._completed = False
        self._simstart = time.time()

        material_set: tuple[np.ndarray,] = []

        if self.bc._initialized_with_defaults is False:
            raise SimulationError(
                "Cannot run a modal analysis because no default boundary conditions have been assigned."
            )

        self._check_meshed()
        self._initialize_field()
        self._initialize_bc_data()
        self._check_physics()

        if self.basis is None:
            raise SimulationError(
                "Cannot proceed, the simulation basis class is undefined."
            )

        logger.debug("Resolving material assignments.")
        materials = self._get_material_assignment(self.mesher.volumes)

        def run_job_single(job: SimJob) -> SimJob:
            A, bmat, ids, aux = job.get_Ab()
            solution, report = self.solveroutine.solve(
                A, bmat, ids, id=job.id, direct=direct_solver
            )
            report.add(**aux)
            job.submit_solution(solution, report)
            return job

        if preconditioner:
            self.solveroutine.use_preconditioner = True

        has_radiation = len(self.bc.oftype(BlackBodyRadiation)) > 0

        if not has_radiation:
            # ----------------------------------------------------------------
            # Linear solve (existing path)
            # ----------------------------------------------------------------
            job, mats = self.assembler.assemble_stationary_matrix(
                self.basis, materials, self.bc.boundary_conditions, self.T_initial_K
            )
            material_set = mats

            logger.info("Starting single threaded solve.")
            job = run_job_single(job)

        else:
            # ----------------------------------------------------------------
            # Nonlinear Picard iteration with Anderson acceleration
            # ----------------------------------------------------------------
            logger.info("Nonlinear radiation BC detected. Starting Picard iteration.")

            # Initial solve without radiation
            T_current = np.full(self.basis.n_field, self.T_initial_K, dtype=np.float64)

            T_history = []
            R_history = []

            for nl_iter in range(max_nonlinear_iter):
                logger.info(f"Nonlinear iteration {nl_iter + 1}/{max_nonlinear_iter}")

                # Assemble with current temperature for radiation linearization
                job, mats = self.assembler.assemble_stationary_matrix(
                    self.basis, materials, self.bc.boundary_conditions, T_current
                )
                material_set = mats

                job = run_job_single(job)

                T_new = job.solution.copy()

                # Fixed-point residual
                R = T_new - T_current
                res_norm = np.linalg.norm(R)
                sol_norm = np.linalg.norm(T_new)
                rel_res = res_norm / sol_norm if sol_norm > 0 else res_norm

                logger.info(
                    f"  |dT| = {res_norm:.4e}, |T| = {sol_norm:.4e}, "
                    f"rel = {rel_res:.4e}"
                )

                if rel_res < nonlinear_tol:
                    logger.info(
                        f"Nonlinear solver converged in {nl_iter + 1} iterations "
                        f"(rel residual = {rel_res:.2e})"
                    )
                    break

                # Store history for Anderson mixing
                T_history.append(T_current.copy())
                R_history.append(R.copy())

                if len(T_history) > anderson_m + 1:
                    T_history.pop(0)
                    R_history.pop(0)

                # Anderson acceleration or simple update
                if len(T_history) >= 2:
                    T_current = self._anderson_update(T_history, R_history, anderson_m)
                else:
                    T_current = T_new

            else:
                logger.warning(
                    f"Nonlinear solver did NOT converge in {max_nonlinear_iter} "
                    f"iterations (rel residual = {rel_res:.2e})"
                )

        self.solveroutine.reset()
        logger.info("Solving complete")

        self.data.setreport(job.reports, **self._params)

        for variables, data in self.data.sim.iterate():
            logger.trace(f"Sim variable: {variables}")
            for item in data["report"]:
                item.logprint(logger.trace)

        self._post_process(job, material_set)
        self._completed = True
        return self.data

    @staticmethod
    def _anderson_update(T_history, R_history, m):
        """Anderson mixing acceleration for fixed-point iteration."""
        k = len(T_history)
        m_use = min(m, k - 1)

        if m_use == 0:
            return T_history[-1] + R_history[-1]

        dR = np.column_stack(
            [R_history[-m_use + i + 1] - R_history[-m_use + i] for i in range(m_use)]
        )

        alpha, _, _, _ = np.linalg.lstsq(dR, R_history[-1], rcond=None)

        T_new = T_history[-1] + R_history[-1]
        for i in range(m_use):
            idx = -m_use + i
            dT = T_history[idx + 1] - T_history[idx]
            dRi = R_history[idx + 1] - R_history[idx]
            T_new -= alpha[i] * (dT + dRi)

        return T_new

    def _post_process(self, job: SimJob, materialset: list[tuple[np.ndarray,]]):

        scalardata = self.data.scalar.new(**self._params)
        fielddata = self.data.field.new(**self._params)

        fielddata.T = job.solution
        fielddata._dcond_thermal = materialset[0]
        fielddata.basis = self.basis
