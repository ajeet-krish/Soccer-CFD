# Project Context — Sports Aerodynamics

## Goal
Integrate SU2 high-fidelity RANS validation alongside existing ΦFlow simulations and restructure the portfolio as a standalone HTML website (`docs/`). Hosted on GitHub Pages.

## SU2 v8.4 Critical Knowledge (Harrier, Rosetta x86_64)
- **Binary**: `/Users/ajeet/SU2_CFD/bin/SU2_CFD` — x86_64 under Rosetta 2 on ARM Mac
- **Quarantine cleared** on all binaries in that directory

### Config option quirks (v8.4, incompressible INC_RANS)
- `VISCOSITY_MODEL= CONSTANT_VISCOSITY` is **required** for inc flows (Sutherland's law rejected)
- `MU_CONSTANT` accepts **non-dimensional** viscosity = `1/Re` (dimensional value causes wrong Re)
- `INC_DENSITY_MODEL= CONSTANT` (not `INC_DENSITY_TYPE`)
- `ITER=` not `MAX_ITER` or `ITERATIONS`
- `CONV_NUM_METHOD_FLOW= FDS` (only `FDS` works for incompressible; `ROE` rejected)
- `MUSCL_FLOW= NO` (required with FDS)
- No `MULTIGRID` option
- `MARKER_MONITORING= ( wall )` required for force output (Cd, Cl)
- `INC_VELOCITY_INIT= ( 1.0, 0.0, 0.0 )` space-separated
- `OUTPUT_FILES= (RESTART, PARAVIEW)` for `.vtu` output

### gmsh SU2 export bugs (fixed)
- SU2 format requires **surface physical groups** AND calling `gmsh.model.mesh.createTopology()` before `gmsh.write()`
- Without physical groups: zero elements in `.su2` file
- Without `createTopology()`: zero elements in `.su2` file
- 2D mesh: add physical groups for cylinder wall (dim=1) and farfield (dim=2)

## File Layout
```
docs/                          # GitHub Pages site (standalone HTML)
  index.html, theory.html, shot.html, tactical.html, su2-validation.html
  custom.css                   # Dark theme + responsive nav
  images/                      # Generated visualizations
su2_runs/
  cylinder_2d/                 # Phase 3 — 2D cylinder validation
    run_cylinder.py            # Mesh → config → SU2 → results pipeline
    results.json               # Structured results metadata
src/
  su2_runner.py                # SU2Config, MeshGenerator, SU2Solver, PyVista viz helpers
  build_site.py                # Nav-sync utility (optional)
SU2_INTEGRATION_PLAN.md        # Full 6-phase plan
AGENTS.md                      # This file
```

## Progress Status
| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (SU2, gmsh, PyVista install) | ✅ |
| 1 | Website (5 HTML pages, dark theme, KaTeX, Prism) | ✅ |
| 2 | SU2 Runner Module (config, meshgen, solver, viz) | ✅ |
| 3a | 2D cylinder first run | ✅ Cd=0.683 |
| 3b | Analysis & comparison with ΦFlow (~1.2) | ✅ 43% diff explained |
| 4 | 3D sphere drag crisis (Re sweep) | 🔜 Next |
| 5 | Smooth vs textured sphere | ⏳ |
| 6 | Deformed ball, altitude effects | ⏳ |

## Key Results
- **SU2 2D cylinder @ Re=40k (RANS SST)**: Cd=0.683, Cl=-0.032
- **ΦFlow 2D cylinder @ Re=40k (laminar)**: Cd≈1.2
- **43% difference**: Fully turbulent RANS delays separation (~110° vs ~80°), produces narrower wake and lower drag. This is physically meaningful, not a bug — both models bracket real sub-critical behavior.

## Visualizations Generated
- `docs/images/cylinder_pressure_field.png`
- `docs/images/cylinder_velocity_field.png`
- `docs/images/cylinder_cd_convergence.png`

## Config template for 2D inc cylinder
SOLVER= INC_RANS
KIND_TURB_MODEL= SST
INC_DENSITY_MODEL= CONSTANT
INC_ENERGY_EQUATION= NO
INC_VELOCITY_INIT= ( 1.0, 0.0, 0.0 )
INC_DENSITY_INIT= 1.2886
INC_NONDIM= INITIAL_VALUES
REYNOLDS_NUMBER= 40000
REYNOLDS_LENGTH= 0.6
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= 2.5e-05
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_MONITORING= ( wall )
MARKER_FAR= ( farfield )
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= NO
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
CFL_NUMBER= 0.5
ITER= 3000
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
TABULAR_FORMAT= CSV
OUTPUT_FILES= (RESTART, PARAVIEW)

## Reference Commands
```bash
# Run cylinder validation
uv run python su2_runs/cylinder_2d/run_cylinder.py

# Run SU2 directly (from workdir)
cd "/Users/ajeet/Projects/Sports Aerodynamics/su2_runs/cylinder_2d" && \
  /Users/ajeet/SU2_CFD/bin/SU2_CFD cylinder.cfg

# Generate viz
uv run python /tmp/generate_viz.py
```
