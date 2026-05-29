# CONTEXT.md — EMerge

## Project Overview

EMerge is a Python-based Finite Element Method (FEM) electromagnetic simulation library for the time-harmonic Helmholtz formulation. It is designed for simulating electromagnetic wave phenomena entirely from within Python scripts — geometry creation, material assignment, meshing, boundary conditions, solving, and post-processing — without any external configuration files or GUI tools.

- **Author:** Robert Fennis
- **Repository:** https://github.com/FennisRobert/EMerge
- **Website:** https://www.emerge-software.com
- **License:** GPL-2.0+ (with CC0 for the materials database `lib.py`)
- **Language:** Python 3.10–3.13 (100%)
- **Package name on PyPI:** `emerge`
- **Current version:** 2.5.6

## What EMerge Can Simulate

- RF filters (combline, stepped impedance, coupled-line, waveguide bandpass)
- Signal propagation through PCBs (microstrip, stripline, differential pairs)
- Antennas (patch, helix, Vivaldi, inverted-F, horn arrays)
- Periodic structures and phased arrays (rectangular and hexagonal cells)
- Radar cross-section (RCS) via scattered field formulation
- Optical/photonic systems
- Steady-state heat conduction (including RF-coupled thermal analysis)

## Repository Layout

```
EMerge/
├── pyproject.toml                    # Build config (hatchling), dependencies, version
├── uv.lock                           # Lock file (uv package manager)
├── .python-version                   # Python version pin
├── README.md
├── LICENSE                           # GPL-2.0+ / CC0 dual license
├── THIRD_PARTY_LICENSES.md
├── UMFPACK_Install_windows.md        # Windows solver setup guide
├── examples/                         # 22+ demo scripts + heat conduction demos
│   ├── demo0_parallel_plate.py
│   ├── demo1_stepped_imp_filter.py   # PCB layout interface
│   ├── demo2_combline_filter.py      # Modeler class, method chaining
│   ├── demo3_coupled_line_filter.py
│   ├── demo4_patch_antenna.py        # Lumped port, far-field, 3D display
│   ├── demo5_revolve.py
│   ├── demo6_striplines_with_vias.py
│   ├── demo7_periodic_cells.py       # HexCell, periodic BC
│   ├── demo8_waveguide_bpf_synthesis.py
│   ├── demo9_dielectric_resonator.py
│   ├── demo10_sgh.py                 # Standard gain horn
│   ├── demo11_lumped_element_filter.py
│   ├── demo12_mode_alignment.py
│   ├── demo13_helix_antenna.py
│   ├── demo14_boundary_selection.py
│   ├── demo15_strip_slotline_transition.py
│   ├── demo16_differential_common_mode.py
│   ├── demo17_step_import.py         # STEP CAD file import
│   ├── demo18_plotting_and_visualization.py
│   ├── demo19_vivaldi_antenna.py
│   ├── demo20_optimization.py        # Built-in optimizer loop
│   ├── demo21_inverted_F_antenna.py
│   ├── demo22_RCS.py                 # Scattered field / RCS
│   ├── DielectricRod.step            # Sample CAD file
│   └── heatconduction/
│       ├── demo1_basic_simulation.py
│       ├── demo2_chip_heating.py
│       ├── demo3_RF_heating.py
│       └── demo4_black_body_radiation.py
└── src/
    ├── __init__.py
    ├── emerge/                        # Main package
    │   ├── __init__.py                # Public API surface, env var setup, version
    │   ├── __main__.py
    │   ├── core.py                    # Legacy API entry (older version of __init__)
    │   ├── cli.py                     # CLI: `emerge` command (init, update, install-solver, etc.)
    │   ├── ext.py                     # Extensions
    │   ├── integrals.py               # Integration utilities (exposed as `em.intf`)
    │   ├── plot.py                    # Re-export of emsutil plotting
    │   ├── read.py                    # File reading utilities
    │   ├── write.py                   # File writing utilities
    │   ├── auxilliary/
    │   │   └── touchstone.py          # Touchstone (S-parameter) file I/O
    │   ├── beta/                      # Experimental features
    │   │   ├── dxf.py                 # DXF import
    │   │   ├── gerber.py              # Gerber PCB file import
    │   │   └── _gerber/               # Gerber parsing internals
    │   └── _emerge/                   # Internal implementation (private)
    │       ├── __init__.py
    │       ├── bc.py                  # Base boundary condition classes
    │       ├── cacherun.py            # Simulation caching
    │       ├── cleanup.py             # Cleanup routines
    │       ├── compiled/              # Numba/JIT compiled routines
    │       ├── const.py               # Internal constants
    │       ├── coord.py               # Coordinate/Line classes
    │       ├── cs.py                  # CoordinateSystem, Axis, Plane definitions
    │       ├── dataset.py             # SimulationDataset
    │       ├── elements/              # FEM element definitions (Nedelec edge elements)
    │       ├── emerge_update.py       # Self-update mechanism
    │       ├── file.py                # Serialization (save/load)
    │       ├── geo/                   # Geometry module (see below)
    │       ├── geometry.py            # Base GeoObject, GeoVolume, GeoSurface, select()
    │       ├── howto.py               # Interactive help system
    │       ├── install_check.py       # Post-import installation verification
    │       ├── logsettings.py         # Loguru log configuration
    │       ├── mesh3d.py              # 3D mesh data structures
    │       ├── mesher.py              # Mesher class, Algorithm2D/3D
    │       ├── mth/                   # Math utilities (norm, dot, cross, coax formulas)
    │       ├── optim.py               # Optimizer class for parameter optimization
    │       ├── periodic.py            # RectCell, HexCell for periodic simulations
    │       ├── physics/               # Physics engines (see below)
    │       ├── plot/                  # Visualization (PyVista-based)
    │       ├── projects/              # Project scaffolding
    │       ├── selection.py           # Selection, FaceSelection, DomainSelection, EdgeSelection
    │       ├── settings.py            # Simulation settings
    │       ├── simmodel.py            # Simulation class — the central orchestrator (1836 lines)
    │       ├── simstate.py            # SimState — shared state container
    │       ├── simulation_data.py     # Data containers
    │       ├── solve_interfaces/      # Solver interface adapters
    │       ├── solver.py              # All solver classes (2174+ lines)
    │       └── system.py              # System utilities
    └── emerge_config/                 # Sister package for pre-import configuration
        ├── __init__.py
        └── config.py                  # Thread count config (for Apple Accelerate, etc.)
```

## Architecture Overview

### Central Class: `Simulation`

`emerge.Simulation` (in `_emerge/simmodel.py`) is the main orchestrator. A typical simulation script:

1. Creates `model = em.Simulation('ModelName')`
2. Builds geometry using `em.geo.*` primitives
3. Calls `model.commit_geometry()` to finalize the geometry in GMSH
4. Sets frequency via `model.mw.set_frequency()` or `model.mw.set_frequency_range()`
5. Generates the mesh via `model.generate_mesh()`
6. Defines boundary conditions via `model.mw.bc.*`
7. Runs the solver via `model.mw.run_sweep()` (or `run_scattered()`, `run_adaptive_sweep()`)
8. Post-processes results from the returned `MWData` object

Key properties/sub-objects of `Simulation`:
- `.mw` → `Microwave3D` — the microwave physics engine
- `.hc` → `HeatConduction3D` — heat conduction physics engine
- `.mesher` → `Mesher` — mesh control (face/boundary/domain sizes, algorithms)
- `.modeler` → `Modeler` — geometry creation helper with method chaining and parameter series
- `.display` → `PVDisplay` — PyVista-based 3D visualization
- `.select` → `Selector` — geometry selection queries
- `.opt` → `Optimizer` — parameter optimization loop
- `.settings` → `Settings` — simulation configuration
- `.state` → `SimState` — internal shared state
- `.data` → `SimulationDataset` — stored simulation results

### Geometry System (`_emerge/geo/`)

All geometry is defined through GMSH's OpenCASCADE kernel. The `geo` module provides:

**Primitives** (`shapes.py`): `Box`, `Cylinder`, `CoaxCylinder`, `Sphere`, `HalfSphere`, `Plate`, `XYPlate`, `Cone`

**Polygon-based** (`polybased.py`): `XYPolygon`, `GeoPrism`, `Disc`, `Curve`

**Operations** (`operations.py`): `subtract`, `add`, `embed`, `remove`, `rotate`, `mirror`, `translate`, `intersect`, `unite`, `extrude`, `stretch`, `expand_surface`, `stick`, `bounding_box`, `change_coordinate_system`

**Specialized**:
- `PCBNew` / `PCB` / `PCBLayer` (`pcb.py`) — PCB layout with trace routing via method chaining
- `Horn` (`horn.py`) — parametric horn antennas
- `STEPItems` (`step.py`) — STEP CAD file import
- `open_region` / `open_pml_region` (`open_region.py`) — open radiation boundaries
- `pmlbox` (`pmlbox.py`) — PML absorbing boundaries

Geometries are subclasses of `GeoObject` (defined in `geometry.py`) which can be `GeoVolume` or `GeoSurface`. Every geometry object exposes face selections like `.top`, `.bottom`, `.left`, `.right`, `.front`, `.back`, `.boundary()`, and `.face(name)`.

### Selection System (`selection.py`)

Selections identify mesh faces, edges, or domains for boundary condition assignment. Types:
- `FaceSelection` — 2D surface selections (for ports, BCs)
- `DomainSelection` — 3D volume selections
- `EdgeSelection` — 1D edge selections

Selections support addition (`sel1 + sel2`) to combine. The `select()` function creates selections from geometry objects. The `Selector` class (on `model.select`) provides query methods like `.face.near(x, y, z)`.

### Coordinate System (`cs.py`)

- `CoordinateSystem` / `CS` — full 3D coordinate system with origin and three axes
- `GCS` — Global Coordinate System
- Pre-defined axes: `XAX`, `YAX`, `ZAX`
- Pre-defined planes: `XYPLANE`, `XZPLANE`, `YZPLANE` (and reverses)
- `cs()` — convenience constructor
- `Anchor` — position anchoring

### Boundary Conditions

All BCs are defined through `model.mw.bc` (`MWBoundaryConditionSet`). Available types:

**Metallic**: `PEC` (perfect electric conductor — default on unassigned surfaces), `PMC` (perfect magnetic conductor)

**Ports** (excitation/measurement):
- `ModalPort` — eigenmode-based waveguide port (TEM, TE, TM modes)
- `RectangularWaveguide` — specialized waveguide port
- `CoaxPort` — coaxial cable port
- `LumpedPort` — lumped element port (voltage gap)
- `UserDefinedPort` — custom field distribution port
- `FloquetPort` — for periodic structures

**Absorbing**: `AbsorbingBoundary` — first-order absorbing BC

**Scattering**: `ScatteredField` — plane wave excitation for RCS computations

**Other**: `LumpedElement`, `SurfaceImpedance`, `Periodic`

### Solver System (`solver.py`)

Direct solvers (the primary path):
- `SolverPardiso` — Intel MKL PARDISO (Windows x86, fastest)
- `SolverAASDS` — Apple Accelerate sparse (macOS ARM, recommended)
- `SolverUMFPACK` — SuiteSparse UMFPACK
- `SolverMUMPS` — MUMPS multi-threaded
- `SolverSuperLU` — SciPy SuperLU (fallback, always available)
- `SolverCuDSS` — NVIDIA cuDSS GPU solver (experimental)

Iterative solvers: `SolverBicgstab`, `SolverGMRES`, `SolverGCROTMK`, `SolverCG`, `SolverCHOLMOD`

Eigenvalue solvers: `SolverLAPACK`, `SolverARPACK`, `SmartARPACK`

The `EMSolver` enum provides named presets. `SolveRoutine` configures solver pipelines. `AutomaticRoutine` auto-selects the best available solver.

### Physics Engines

**Microwave** (`physics/microwave/`):
- `Microwave3D` — full-wave frequency-domain solver using Nedelec (edge) elements
- `MWData` — result container with `.scalar` (S-parameters) and `.field` (E/H fields)
- Field post-processing: `.cutplane()`, `.grid()`, `.boundary()`, `.farfield_2d()`, `.farfield_3d()`, `.scalar()`, `.vector()`
- S-parameter modeling via Vector Fitting algorithm
- Supports parallel frequency sweeps, adaptive frequency sweeps, adaptive mesh refinement

**Heat Conduction** (`physics/heatconduction/`):
- `HeatConduction3D` — steady-state thermal solver
- Can couple with microwave results (RF heating)

### External Dependencies

**Required** (installed via pip): `numpy`, `scipy`, `gmsh` (≥4.13), `loguru`, `numba`, `matplotlib`, `pyvista`, `emsutil` (companion utilities package by the same author), `joblib`, `msgpack`, `psutil`

**Platform-specific**: `mkl` (x86 only)

**Optional**: `scikit-umfpack`, `nvidia-cudss-cu12` + `cupy`, `ezdxf` (DXF import), `pygerber` (Gerber import)

`emsutil` is a key companion package providing: `Material`, `MatProperty`, `FreqDependent`, `CoordDependent`, material libraries (`isola`, `rogers`, `const`, `lib`), plotting functions, physical constants (`C0`, `MU0`, `EPS0`, `Z0`), and theming.

### CLI

The `emerge` command (entry point in `cli.py`) provides subcommands for project scaffolding, solver installation, and updates. Command classes are auto-registered via `__init_subclass__`.

### Key Patterns and Conventions

1. **All dimensions in SI units** — meters, Hertz, etc. Users typically define `mm = 0.001` at the top of scripts.

2. **Method chaining** — The PCB layout interface and display system support fluent method chaining.

3. **Geometry is committed before meshing** — `model.commit_geometry()` must be called after all geometry is defined and before `model.generate_mesh()`.

4. **Face access on geometry objects** — `box.top`, `box.bottom`, `box.left`, `box.right`, `box.front`, `box.back`, `box.boundary()`, `box.face('+z')`, `box.face('back', tool=other_geo)`.

5. **Materials** — Assigned via `.set_material()` on geometry objects. PEC is the default for unassigned metal surfaces. `em.Material(er=value)` for dielectrics. Pre-defined materials in `em.lib`, `em.isola`, `em.rogers`.

6. **Periodic simulations** require a `PeriodicCell` (`RectCell` or `HexCell`) set on the simulation before meshing.

7. **Data access** — After `model.mw.run_sweep()`, results are in `data.scalar` (S-parameters) and `data.field` (field solutions). Use `.grid` for structured parameter sweeps. Use `.model_S()` for Vector Fitting interpolation.

8. **Display pipeline** — `model.display.add_object()`, `.add_field()`, `.add_portmode()`, `.animate()`, `.show()`.

9. **Optimization** — `model.opt.add_param(name, initial, bounds)`, then iterate with `for params in model.opt.run(max_iter=N)`.

10. **Serialization** — Supports `joblib` (legacy) and `msgpack` (safe) for saving/loading simulation state.

11. **GMSH integration** — GMSH is initialized/finalized automatically by the `Simulation` class. Signal handlers and atexit hooks ensure cleanup.

12. **Environment variables** — Thread counts for MKL, OpenBLAS, VECLIB, Numba, and OMP are set in `__init__.py`. Override before importing `emerge` or use `emerge_config`.
