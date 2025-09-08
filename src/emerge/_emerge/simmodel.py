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
from .mesher import Mesher, unpack_lists
from .geometry import GeoObject, _GEOMANAGER
from .geo.modeler import Modeler
from .physics.microwave.microwave_3d import Microwave3D
from .mesh3d import Mesh3D
from .selection import Selector, FaceSelection, Selection
from .logsettings import LOG_CONTROLLER
from .plot.pyvista import PVDisplay
from .dataset import SimulationDataset
from .periodic import PeriodicCell
from .cacherun import get_build_section, get_run_section
from .settings import DEFAULT_SETTINGS, Settings
from .solver import EMSolver, Solver
from typing import Literal, Generator, Any
from loguru import logger
import numpy as np
import gmsh # type: ignore
import cloudpickle
import os
import inspect
from pathlib import Path
from atexit import register
import signal
from .. import __version__

############################################################
#                   EXCEPTION DEFINITIONS                  #
############################################################

_GMSH_ERROR_TEXT = """
--------------------------
Known problems/solutions:
(1) - PLC Error:  A segment and a facet intersect at point
    This can be caused when approximating thin curved volumes. Try to decrease the mesh size for that region.
--------------------------
"""

class SimulationError(Exception):
    pass

class VersionError(Exception):
    pass

############################################################
#                 BASE 3D SIMULATION MODEL                 #
############################################################

class Simulation:

    def __init__(self, 
                 modelname: str, 
                 loglevel: Literal['TRACE','DEBUG','INFO','WARNING','ERROR'] = 'INFO',
                 load_file: bool = False,
                 save_file: bool = False,
                 write_log: bool = False,
                 path_suffix: str = ".EMResults"):
        """Generate a Simulation class object.

        As a minimum a file name should be provided. Additionally you may provide it with any
        class that inherits from BaseDisplay. This will then be used for geometry displaying.

        Args:
            modelname (str): The name of the simulation model. This will be used for filenames and path names when saving data.
            loglevel ("DEBUG","INFO","WARNING","ERROR", optional): The loglevel to use for loguru. Defaults to 'INFO'.
            load_file (bool, optional): If the simulatio model should be loaded from a file. Defaults to False.
            save_file (bool, optional): if the simulation file should be stored to a file. Defaults to False.
            write_log (bool, optional): If a file should be created that contains the entire log of the simulation. Defaults to False.
            path_suffix (str, optional): The suffix that will be added to the results directory. Defaults to ".EMResults".
        """

        caller_file = Path(inspect.stack()[1].filename).resolve()
        base_path = caller_file.parent

        self.modelname = modelname
        self.modelpath = base_path / (modelname.lower()+path_suffix)
        self.mesher: Mesher = Mesher()
        self.modeler: Modeler = Modeler()
        
        self.mesh: Mesh3D = Mesh3D(self.mesher)
        self.select: Selector = Selector()
        
        self.settings: Settings = DEFAULT_SETTINGS

        ## Display
        self.display: PVDisplay = PVDisplay(self.mesh)
        
        ## Dataset
        self.data: SimulationDataset = SimulationDataset()
        
        ## STATES
        self.__active: bool = False
        self._defined_geometries: bool = False
        self._cell: PeriodicCell | None = None
        self.save_file: bool = save_file
        self.load_file: bool = load_file
        self._cache_run: bool = False
        self._file_lines: str = ''
        
        ## Physics
        self.mw: Microwave3D = Microwave3D(self.mesher, self.settings, self.data.mw)

        self._initialize_simulation()

        self.set_loglevel(loglevel)
        if write_log:
            self.set_write_log()

        LOG_CONTROLLER._flush_log_buffer()
        self._update_data()
    

    ############################################################
    #                       PRIVATE FUNCTIONS                  #
    ############################################################

    def __setitem__(self, name: str, value: Any) -> None:
        """Store data in the current data container"""
        self.data.sim[name] = value

    def __getitem__(self, name: str) -> Any:
        """Get the data from the current data container"""
        return self.data.sim[name]
    
    def __enter__(self) -> Simulation:
        """This method is depricated with the new atexit system. It still exists for backwards compatibility.

        Returns:
            Simulation: the Simulation object
        """
        return self

    def __exit__(self, type, value, tb):
        """This method no longer does something. It only serves as backwards compatibility."""
        self._exit_gmsh()
        return False
    
    def _install_signal_handlers(self):
        # on SIGINT (Ctrl-C) or SIGTERM, call our exit routine
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """
        Signal handler: do our cleanup, then re-raise
        the default handler so that exit code / traceback
        is as expected.
        """
        try:
            # run your atexit-style cleanup
            self._exit_gmsh()
        except Exception:
            # log but don’t block shutdown
            logger.exception("Error during signal cleanup")
        finally:
            # restore default handler and re‐send the signal
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _initialize_simulation(self):
        """Initializes the Simulation data and GMSH API with proper shutdown routines.
        """
        _GEOMANAGER.sign_in(self.modelname)
        
        # If GMSH is not yet initialized (Two simulation in a file)
        if gmsh.isInitialized() == 0:
            logger.debug('Initializing GMSH')
            gmsh.initialize()
            
            # Set an exit handler for Ctrl+C cases
            self._install_signal_handlers()

            # Restier the Exit GMSH function on proper program abortion
            register(self._exit_gmsh)
        else:
            gmsh.finalize()
            gmsh.initialize()
            
        # Create a new GMSH model or load it
        if not self.load_file:
            gmsh.model.add(self.modelname)
            self.data: SimulationDataset = SimulationDataset()
        else:
            self.load()

        # Set the Simulation state to active
        self.__active = True
        return self

    def _exit_gmsh(self):
        # If the simulation object state is still active (GMSH is running)
        if not self.__active:
            return
        logger.debug('Exiting program')
        # Save the file first
        if self.save_file:
            self.save()
        # Finalize GMSH
        if gmsh.isInitialized():
            gmsh.finalize()

        logger.debug('GMSH Shut down successful')
        # set the state to active
        self.__active = False

    def _update_data(self) -> None:
        """Writes the stored physics data to each phyics class insatnce"""
        self.mw.data = self.data.mw
    
    def _set_mesh(self, mesh: Mesh3D) -> None:
        """Set the current model mesh to a given mesh."""
        self.mesh = mesh
        self.mw.mesh = mesh
        self.display._mesh = mesh
    
    ############################################################
    #                       PUBLIC FUNCTIONS                  #
    ############################################################

    def cache_build(self) -> bool:
        """Checks if all the lines inside this if statement block are the same as those
        stored from a previous run. If so, then it returns false. Else it returns True.
        
        Can be used to capture an entire model simulation.
        
        Example:
        
        >>> if model.cache_build():
        >>>     box = em.geo.Box(...)
        >>>     # Other lines
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """
        
        self.save_file = True
        self._cache_run = True
        filestr = get_build_section()
        self._file_lines = filestr
        cachepath = self.modelpath / 'pylines.txt'
        
        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            logger.info('No cached data detected, running file')
            return True
        
        with open(cachepath, 'r') as file:
            lines = file.read()
        
        if lines==filestr:
            logger.info('Cached data detected! Loading data!')
            self.load()
            return False
        logger.info('Different cached data detected, rebuilding file.')
        return True
    
    def cache_run(self) -> bool:
        """Checks if all the lines before this call are the same as the lines
        stored from a previous run. If so, then it returns false. Else it returns True.
        
        Can be used to capture a run_sweep() call.
        
        Example:
        
        >>> if model.cache_run():
        >>>     model.mw.run_sweep()
        >>> data = model.data.mw

        Returns:
            bool: If the code is not the same
        """
        self.save_file = True
        self._cache_run = True
        filestr = get_run_section()
        self._file_lines = filestr
        cachepath = self.modelpath / 'pylines.txt'
        
        # If there is not pylines file, simulate (can't have been run).
        if not cachepath.exists():
            logger.info('No cached data detected, running simulation!')
            return True
        
        with open(cachepath, 'r') as file:
            lines = file.read()
        
        if lines==filestr:
            logger.info('Cached data detected! Loading data!')
            self.load()
            return False
        logger.info('Different cached data detected, rerunning simulation.')
        return True
    
    def check_version(self, target_version: str, *, log: bool = False) -> None:
        """
        Ensure the script targets an EMerge version compatible with the current runtime.

        Parameters
        ----------
        target_version : str
            The EMerge version this script was written for (e.g. "1.4.0").
        log : bool, optional
            If True and a `logger` is available, emit a single WARNING with the same
            message as the exception. Defaults to False.

        Raises
        ------
        VersionError
            If the script's target version differs from the running EMerge version.
        """
        try:
            from packaging.version import Version as _V
            v_script = _V(target_version)
            v_runtime = _V(__version__)
            newer = v_script > v_runtime
            older = v_script < v_runtime
        except Exception:
            def _parse(v: str):
                try:
                    return tuple(int(p) for p in v.split("."))
                except Exception:
                    # Last-resort: compare as strings to avoid crashing the check itself
                    return tuple(v.split("."))
            v_script = _parse(target_version)
            v_runtime = _parse(__version__)
            newer = v_script > v_runtime
            older = v_script < v_runtime

        if not newer and not older:
            return  # exact match

        if newer:
            msg = (
                f"Script targets EMerge {target_version}, but runtime is {__version__}. "
                "The script may rely on features added after your installed version. "
                "Recommended: upgrade EMerge (`pip install --upgrade emerge`). "
                "If you know the script is compatible, you may remove this check."
            )
        else:  # older
            msg = (
                f"Script targets EMerge {target_version}, but runtime is {__version__}. "
                "APIs may have changed since the targeted version. "
                "Recommended: update the script for the current EMerge, or run a matching older release. "
                "If you know the script is compatible, you may remove this check."
            )

        if log:
            try:
                logger.warning(msg)
            except Exception:
                pass

        raise VersionError(msg)

    def save(self) -> None:
        """Saves the current model in the provided project directory."""
        # Ensure directory exists
        if not self.modelpath.exists():
            self.modelpath.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.modelpath}")

        # Save mesh
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'

        gmsh.option.setNumber('Mesh.SaveParametric', 1)
        gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()

        gmsh.write(str(mesh_path))
        gmsh.write(str(brep_path))
        logger.info(f"Saved mesh to: {mesh_path}")

        # Pack and save data
        dataset = dict(simdata=self.data, mesh=self.mesh)
        data_path = self.modelpath / 'simdata.emerge'
        with open(str(data_path), "wb") as f_out:
            cloudpickle.dump(dataset, f_out)
            
        if self._cache_run:
            cachepath = self.modelpath / 'pylines.txt'
            with open(str(cachepath), 'w') as f_out:
                f_out.write(self._file_lines)
            
        logger.info(f"Saved simulation data to: {data_path}")

    def load(self) -> None:
        """Loads the model from the project directory."""
        mesh_path = self.modelpath / 'mesh.msh'
        brep_path = self.modelpath / 'model.brep'
        data_path = self.modelpath / 'simdata.emerge'

        if not mesh_path.exists() or not data_path.exists():
            raise FileNotFoundError("Missing required mesh or data file.")

        # Load mesh
        gmsh.open(str(brep_path))
        gmsh.merge(str(mesh_path))
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        logger.info(f"Loaded mesh from: {mesh_path}")
        #self.mesh.update([])

        # Load data
        with open(str(data_path), "rb") as f_in:
            datapack= cloudpickle.load(f_in)
        self.data = datapack['simdata']
        self._set_mesh(datapack['mesh'])
        logger.info(f"Loaded simulation data from: {data_path}")
    
    def set_loglevel(self, loglevel: Literal['DEBUG','INFO','WARNING','ERROR']) -> None:
        """Set the loglevel for loguru.

        Args:
            loglevel ('DEBUG','INFO','WARNING','ERROR'): The loglevel
        """
        LOG_CONTROLLER.set_std_loglevel(loglevel)
        if loglevel not in ('TRACE','DEBUG'):
            gmsh.option.setNumber("General.Terminal", 0)

    def set_write_log(self) -> None:
        """Adds a file output for the logger."""
        LOG_CONTROLLER.set_write_file(self.modelpath)
        
    def view(self, 
             selections: list[Selection] | None = None, 
             use_gmsh: bool = False,
             plot_mesh: bool = False,
             volume_mesh: bool = True,
             opacity: float | None = None,
             labels: bool = False) -> None:
        """View the current geometry in either the BaseDisplay object (PVDisplay only) or
        the GMSH viewer.

        Args:
            selections (list[Selection] | None, optional): Optional selections to highlight. Defaults to None.
            use_gmsh (bool, optional): If GMSH's GUI should be used. Defaults to False.
            plot_mesh (bool, optional): If the mesh should be plot instead of the object. Defaults to False.
            volume_mesh (bool, optional): If the internal mesh should be plot instead of only the surface boundary mesh. Defaults to True
            opacity (float | None, optional): The object/mesh opacity. Defaults to None.
            labels: (bool, optional): If geometry name labels should be shown. Defaults to False.
        """
        if not (self.display is not None and self.mesh.defined) or use_gmsh:
            gmsh.model.occ.synchronize()
            gmsh.fltk.run()
            return
        for geo in _GEOMANAGER.all_geometries():
            self.display.add_object(geo, mesh=plot_mesh, opacity=opacity, volume_mesh=volume_mesh, label=labels)
        if selections:
            [self.display.add_object(sel, color='red', opacity=0.3, label=labels) for sel in selections]
        self.display.show()

        return None
        
    def set_periodic_cell(self, cell: PeriodicCell, included_faces: FaceSelection | None = None):
        """Set the given periodic cell object as the simulations peridicity.

        Args:
            cell (PeriodicCell): The PeriodicCell class
            excluded_faces (list[FaceSelection], optional): Faces to exclude from the periodic boundary condition. Defaults to None.
        """
        self.mw.bc._cell = cell
        self._cell = cell
        self._cell.included_faces = included_faces

    def commit_geometry(self, *geometries: GeoObject | list[GeoObject]) -> None:
        """Finalizes and locks the current geometry state of the simulation.

        The geometries may be provided (legacy behavior) but are automatically managed in the background.
        
        """
        geometries_parsed: Any = None
        if not geometries:
            geometries_parsed = _GEOMANAGER.all_geometries()
        else:
            geometries_parsed = unpack_lists(geometries + tuple([item for item in self.data.sim.default.values() if isinstance(item, GeoObject)]))
        
        self.data.sim['geos'] = {geo.name: geo for geo in geometries_parsed}
        self.mesher.submit_objects(geometries_parsed)
        self._defined_geometries = True
        self.display._facetags = [dt[1] for dt in gmsh.model.get_entities(2)]
    
    def all_geos(self) -> list[GeoObject]:
        """Returns all geometries in a list

        Returns:
            list[GeoObject]: A list of all GeoObjects
        """
        return _GEOMANAGER.all_geometries()  
    
    def generate_mesh(self) -> None:
        """Generate the mesh. 
        This can only be done after commit_geometry(...) is called and if frequencies are defined.

        Args:
            name (str, optional): The mesh file name. Defaults to "meshname.msh".

        Raises:
            ValueError: ValueError if no frequencies are defined.
        """
        if not self._defined_geometries:
            self.commit_geometry()
        
        # Set the cell periodicity in GMSH
        if self._cell is not None:
            self.mesher.set_periodic_cell(self._cell)
            
        self.mw._initialize_bcs(_GEOMANAGER.get_surfaces())

        # Check if frequencies are defined: TODO: Replace with a more generic check
        if self.mw.frequencies is None:
            raise ValueError('No frequencies defined for the simulation. Please set frequencies before generating the mesh.')

        gmsh.model.occ.synchronize()

        # Set the mesh size
        self.mesher.set_mesh_size(self.mw.get_discretizer(), self.mw.resolution)
        
        try:
            gmsh.logger.start()
            gmsh.model.mesh.generate(3)
            logs = gmsh.logger.get()
            gmsh.logger.stop()
            for log in logs:
                logger.trace('[GMSH] '+log)
        except Exception:
            logger.error('GMSH Mesh error detected.')
            print(_GMSH_ERROR_TEXT)
            raise
        
        self.mesh._pre_update(self.mesher._get_periodic_bcs())
        self.mesh.exterior_face_tags = self.mesher.domain_boundary_face_tags
        gmsh.model.occ.synchronize()
        
        self._set_mesh(self.mesh)

    def parameter_sweep(self, clear_mesh: bool = True, **parameters: np.ndarray) -> Generator[tuple[float,...], None, None]:
        """Executes a parameteric sweep iteration.

        The first argument clear_mesh determines if the mesh should be cleared and rebuild in between sweeps. This is usually needed
        except when you change only port-properties or material properties. The parameters of the sweep can be provided as a set of 
        keyword arguments. As an example if you defined the axes: width=np.linspace(...) and height=np.linspace(...). You can
        add them as arguments using .parameter_sweep(True, width=width, height=height).

        The rest of the simulation commands should be inside the iteration scope

        Args:
            clear_mesh (bool, optional): Wether to clear the mesh in between sweeps. Defaults to True.

        Yields:
            Generator[tuple[float,...], None, None]: The parameters provided

        Example:
         >>> for W, H in model.parameter_sweep(True, width=widths, height=heights):
         >>>    // build simulation
         >>>    data = model.run_sweep()
         >>> // Extract the data
         >>> widths, heights, frequencies, S21 = data.ax('width','height','freq').S(2,1)
        """
        paramlist = sorted(list(parameters.keys()))
        dims = np.meshgrid(*[parameters[key] for key in paramlist], indexing='ij')
        dims_flat = [dim.flatten() for dim in dims]
        self.mw.cache_matrices = False
        for i_iter in range(dims_flat[0].shape[0]):
            if clear_mesh:
                logger.info('Cleaning up mesh.')
                gmsh.clear()
                mesh = Mesh3D(self.mesher)
                _GEOMANAGER.reset(self.modelname)
                self._set_mesh(mesh)
                self.mw.reset()
            
            params = {key: dim[i_iter] for key,dim in zip(paramlist, dims_flat)}
            self.mw._params = params
            self.data.sim.new(**params)

            logger.info(f'Iterating: {params}')
            if len(dims_flat)==1:
                yield dims_flat[0][i_iter]
            else:
                yield (dim[i_iter] for dim in dims_flat) # type: ignore
        self.mw.cache_matrices = True
    
    def export(self, filename: str):
        """Exports the model or mesh depending on the extension. 
        
        Exporting is realized by GMSH.
        Supported file formats are:
        
        3D Model: .opt, .geo_unrolled, .brep, .xao ,.step and .iges
        Mesh: .msh, .inp, .key, ._0000.rad, .celum, .cgns, .diff, .unv, .ir3, .mes, .mesh 
              .mail, .m, .bdf, .off, .p3d, .stl, .wrl, .vtk, .dat, .ply2, .su2, .neu, .x3d

        Args:
            filename (str): The filename
        """
        gmsh.write(filename)
        
    def set_solver(self, solver: EMSolver | Solver):
        """Set a given Solver class instance as the main solver. 
        Solvers will be checked on validity for the given problem.

        Args:
            solver (EMSolver | Solver): The solver objects
        """
        self.mw.solveroutine.set_solver(solver)
        
    ############################################################
    #                     DEPRICATED FUNCTIONS                #
    ############################################################

    def define_geometry(self, *args):
        """DEPRICATED VERSION: Use .commit_geometry()
        """
        logger.warning('define_geometry() will be derpicated. Use commit_geometry() instead.')
        self.commit_geometry(*args)
        