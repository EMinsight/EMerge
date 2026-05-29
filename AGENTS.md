# AGENTS.md — EMerge

Instructions for AI coding agents (Claude Code, Copilot, Cursor, etc.) working on the EMerge codebase.

## Quick Orientation

EMerge is a Python FEM electromagnetic simulation library. The user writes a single Python script that defines geometry, meshes it, assigns boundary conditions, solves, and post-processes — all through the `emerge` package. There is no separate config file or GUI.

Read `CONTEXT.md` in this repo for full architectural details.

## Build & Run

```bash
# Install (basic)
pip install emerge

# Install from source (development)
git clone https://github.com/FennisRobert/EMerge.git
cd EMerge
pip install -e .

# Optional solver backends
pip install scikit-umfpack          # UMFPACK (Linux/macOS)
pip install emerge[cudss]           # NVIDIA GPU solver

# Run any example
python examples/demo0_parallel_plate.py

# CLI
emerge --help
```

**Python version:** 3.10–3.13. The build system is `hatchling`. The lock file uses `uv`.

**Testing:**
```bash
pytest                               # Runs tests in tests/ directory
```
Test config is in `pyproject.toml` under `[tool.pytest.ini_options]`. Tests use `importlib` import mode. The `trial_scripts` directory is ignored.

## Code Structure Rules

### Package layout

- `src/emerge/__init__.py` — the public API surface. All user-facing symbols are imported here.
- `src/emerge/_emerge/` — private implementation. Users never import from here directly.
- `src/emerge/core.py` — legacy API entry point (older version, kept for compatibility).
- `src/emerge_config/` — separate sister package for pre-import thread configuration.

### Key files by responsibility

| What you're changing | Where to look |
|---|---|
| Public API, imports, version | `src/emerge/__init__.py` |
| Simulation orchestration | `src/emerge/_emerge/simmodel.py` |
| Geometry primitives | `src/emerge/_emerge/geo/shapes.py` |
| Geometry operations (boolean, transforms) | `src/emerge/_emerge/geo/operations.py` |
| PCB layout system | `src/emerge/_emerge/geo/pcb.py` |
| Boundary conditions (microwave) | `src/emerge/_emerge/physics/microwave/microwave_bc.py` |
| Microwave solver engine | `src/emerge/_emerge/physics/microwave/microwave_3d.py` |
| FEM assembly | `src/emerge/_emerge/physics/microwave/assembly/` |
| Post-processing / data | `src/emerge/_emerge/physics/microwave/microwave_data.py` |
| Heat conduction physics | `src/emerge/_emerge/physics/heatconduction/` |
| Mesh generation | `src/emerge/_emerge/mesher.py`, `mesh3d.py` |
| Solver backends | `src/emerge/_emerge/solver.py` |
| Selection system | `src/emerge/_emerge/selection.py` |
| Coordinate systems | `src/emerge/_emerge/cs.py` |
| Periodic structures | `src/emerge/_emerge/periodic.py` |
| Optimization loop | `src/emerge/_emerge/optim.py` |
| Visualization | `src/emerge/_emerge/plot/` |
| CLI | `src/emerge/cli.py` |
| Serialization | `src/emerge/_emerge/file.py` |

### Companion package: `emsutil`

Material definitions (`Material`, `MatProperty`, `FreqDependent`, `CoordDependent`), material libraries (`lib`, `isola`, `rogers`), plotting utilities (`plot`, `plot_sp`, `smith`, `plot_ff`, `plot_ff_polar`), physical constants (`C0`, `MU0`, `EPS0`, `Z0`), and theming (`EMergeTheme`, `themes`) all live in the separate `emsutil` package. Do not duplicate these — import from `emsutil`.

## Coding Conventions

### Style

- No strict formatter enforced (no ruff/black config in repo). Follow the existing style: 4-space indentation, type hints on public methods, docstrings on public classes/methods.
- Copyright header at the top of every source file (GPL-2.0+).
- Cleanup date comment: `# Last Cleanup: YYYY-MM-DD`.
- Section dividers use `############################################################` comment blocks.

### Import patterns

- Internal modules use relative imports: `from .geometry import GeoObject`, `from ..mesher import Mesher`.
- Public `__init__.py` re-exports everything the user needs so they only write `import emerge as em`.
- Avoid circular imports — this is a known pain point (see the `_CalculationInterface` pattern in `selection.py`). If you add cross-module dependencies, be very careful about import order.

### GMSH integration

- All geometry goes through the GMSH Python API (`gmsh` module).
- GMSH is initialized/finalized by the `Simulation` class. Do not call `gmsh.initialize()` or `gmsh.finalize()` outside `simmodel.py`.
- The `Simulation` class registers atexit hooks and signal handlers for cleanup.

### Serialization

- Two backends: `joblib` (legacy default) and `msgpack` (preferred for new code).
- Saveable classes inherit from `emsutil.Saveable`.

### Threading and parallelism

- Environment variables for thread counts are set in `__init__.py` before any numeric library is imported.
- Frequency sweeps can run in parallel via `joblib` with `parallel=True` on `run_sweep()`.
- Be careful with GMSH — it is not thread-safe for geometry operations.

## Common Tasks

### Adding a new geometry primitive

1. Create the class in `src/emerge/_emerge/geo/shapes.py` (or a new file in `geo/`).
2. Inherit from `GeoVolume` or `GeoSurface` (from `geometry.py`).
3. Register face selections in `__init__` using the GMSH entity tags.
4. Export from `src/emerge/_emerge/geo/__init__.py`.
5. The public API in `__init__.py` imports `geo` as a module — the new class is automatically accessible as `em.geo.YourClass`.

### Adding a new boundary condition

1. Add the class in `src/emerge/_emerge/physics/microwave/microwave_bc.py`.
2. Inherit from `BoundaryCondition` or one of its subclasses (`RobinBC`, `PortBC`).
3. Register it in `MWBoundaryConditionSet` so it's accessible via `model.mw.bc.YourBC(...)`.
4. Implement the assembly contribution in `physics/microwave/assembly/`.

### Adding a new solver backend

1. Add a solver class in `src/emerge/_emerge/solver.py` inheriting from `Solver` (or `EigSolver`).
2. Implement `.solve(A, b)` (and optionally `.factorize(A)` + `.backsolve(b)`).
3. Add an entry to the `EMSolver` enum.
4. Update `AutomaticRoutine` if the solver should be auto-selected.

### Adding a new example

1. Create `examples/demoNN_descriptive_name.py`.
2. Follow the pattern: import emerge, define dimensions, create Simulation, build geometry, commit, set frequency, mesh, assign BCs, solve, post-process, display.
3. Include a docstring header explaining what the example demonstrates.
4. Use `model.check_version("X.Y.Z")` for version compatibility.

## Things to Watch Out For

1. **Circular imports** — The geometry ↔ selection dependency is fragile. The `_CalculationInterface` hack in `selection.py` exists because `Selection` needs mesh data but `GeoObject` creates selections. Do not add direct imports between these modules.

2. **GMSH state** — GMSH maintains global state. Only one `Simulation` object should be active at a time. The `Simulation.__init__` handles re-initialization if GMSH is already running.

3. **Version string** — The version appears in `__init__.py`, `core.py`, and `pyproject.toml`. Keep them in sync. There is a `.bumpversion.toml` config for automated version bumping.

4. **SI units everywhere** — All internal calculations use SI (meters, Hz, S/m, etc.). User-facing convenience constants (`mm`, `GHz`, etc.) are defined in `core.py` but not in `__init__.py` — users define their own.

5. **`emsutil` dependency** — Material classes, physical constants, plotting, and the `Saveable` base class all come from `emsutil` (version ≥0.6.0, <0.7.0). Changes to material handling require coordinating with that package.

6. **Platform-dependent solvers** — MKL/PARDISO is x86-only. Apple Accelerate is macOS ARM only. UMFPACK requires system libraries. SuperLU is the universal fallback. Code that touches solvers must handle import failures gracefully.

7. **Large files** — `simmodel.py` (1836 lines), `solver.py` (2174+ lines), `microwave_3d.py` (2769 lines), and `microwave_bc.py` (1600+ lines) are the largest files. Navigate by method name rather than scrolling.

8. **The `core.py` vs `__init__.py` split** — `core.py` is an older version of the public API kept for backwards compatibility. New symbols go in `__init__.py`. The two files have diverged; `__init__.py` is authoritative.

9. **Numba JIT** — Performance-critical assembly code in `_emerge/compiled/` uses Numba. These functions have strict typing requirements. Test on both first-run (JIT compilation) and subsequent runs.

10. **Display system** — `PVDisplay` uses PyVista. The `.animate()` call toggles animation mode via method chaining. Display operations are non-blocking until `.show()` is called.
