# Project Context — Sports Aerodynamics

## Goal
Integrate SU2 high-fidelity RANS/URANS validation alongside existing ΦFlow simulations and publish as a standalone HTML portfolio on GitHub Pages.

## SU2 v8.4 Critical Knowledge (Harrier, Rosetta x86_64)
- **Binary**: `/Users/ajeet/SU2_CFD/bin/SU2_CFD` — x86_64 under Rosetta 2 on ARM Mac
- **Quarantine cleared** on all binaries in that directory

### Config option quirks (v8.4, incompressible INC_RANS / INC_NAVIER_STOKES)
- `VISCOSITY_MODEL= CONSTANT_VISCOSITY` is **required** for inc flows (Sutherland's law rejected)
- `MU_CONSTANT` with `INC_NONDIM= INITIAL_VALUES`: non-dimensional = `1/Re`
- `MU_CONSTANT` with `INC_NONDIM= DIMENSIONAL`: dimensional = `ρ·U·D/Re` (e.g. 0.00267875 for Re=120)
- `INC_DENSITY_MODEL= CONSTANT` (not `INC_DENSITY_TYPE`)
- `ITER=` not `MAX_ITER` or `ITERATIONS`
- `CONV_NUM_METHOD_FLOW= FDS` (only `FDS` works for incompressible; `ROE` rejected)
- `MUSCL_FLOW= NO` for RANS (required with FDS); `MUSCL_FLOW= YES` for laminar NS
- `SLOPE_LIMITER_FLOW= NONE` with MUSCL=YES for laminar shedding
- No `MULTIGRID` option
- `MARKER_MONITORING= ( wall )` required for force output (Cd, Cl)
- `INC_VELOCITY_INIT= ( 1.0, 0.0, 0.0 )` space-separated
- `OUTPUT_FILES= (RESTART, PARAVIEW)` for `.vtu` output
- `HISTORY_OUTPUT= (TIME_ITER, INNER_ITER, RMS_RES, AERO_COEFF, CUR_TIME)` — use AERO_COEFF group, not individual fields
- `TIME_DOMAIN= YES`, `TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER`, `TIME_STEP`, `TIME_ITER`, `INNER_ITER`
- `OUTPUT_WRT_FREQ= 1` for per-timestep VTU output
- Unsteady: remove `ITER` from config; remove `ITER` from HISTORY_OUTPUT fields
- Laminar shedding: `INC_NONDIM= DIMENSIONAL` (INITIAL_VALUES too dissipative)
- Moving wall: `SURFACE_MOVEMENT= MOVING_WALL`, `MARKER_MOVING= ( wall )`, `SURFACE_ROTATION_RATE= 0.0 0.0 <ωz>`
- LOW_MACH_PREC (not LOW_MACH_PRECONDITIONER) for compressible NS at M<0.1 with ROE

### gmsh SU2 export bugs (fixed)
- SU2 format requires **surface physical groups** AND calling `gmsh.model.mesh.createTopology()` before `gmsh.write()`
- Without physical groups: zero elements in `.su2` file
- Without `createTopology()`: zero elements in `.su2` file
- 2D mesh: add physical groups for cylinder wall (dim=1) and farfield (dim=2)
- gmsh Box field: `VIn` (size inside box) / `VOut` (size outside), not `VFull`

## File Layout
```
docs/                          # GitHub Pages site (standalone HTML)
  index.html, theory.html, shot.html, tactical.html, su2-validation.html
  custom.css                   # Dark theme + responsive nav
  images/                      # All visualizations (PNG, MP4, HTML)
su2_runs/
  cylinder_2d/                 # Phase 3 — 2D cylinder validation
    run_cylinder.py            # Steady-state mesh → config → SU2 → results
    run_unsteady.py            # Unsteady laminar Re=120 (no-spin + magnus)
    generate_viz.py            # All plots, animations, 3D HTML generator
    cylinder_fine.su2          # Wake-refined mesh (44,710 nodes, 89,422 tri)
    output_nospin_lam/         # 200 per-timestep VTU + history CSV (no spin)
    output_magnus_lam/         # 200 per-timestep VTU + history CSV (magnus)
src/
  su2_runner.py                # SU2Config, MeshGenerator (wake refine), SU2Solver, PyVista viz
  build_site.py                # Nav-sync utility (optional)
SU2_INTEGRATION_PLAN.md        # Full 6-phase plan
AGENTS.md                      # This file
```

## Visualizations Generated
| File | Type | Description |
|------|------|-------------|
| `cylinder_mesh.png` | PNG | Full domain mesh (224K) |
| `cylinder_mesh_zoom.png` | PNG | Cylinder close-up with wake refinement (1.3M) |
| `cylinder_nospin_pressure.png` | PNG | No-spin pressure field (final t=30s) |
| `cylinder_magnus_pressure.png` | PNG | Magnus pressure field (asymmetric dipole visible) |
| `cylinder_compare_pressure.png` | PNG | Side-by-side pressure comparison |
| `cylinder_nospin_velocity.png` | PNG | No-spin velocity magnitude |
| `cylinder_magnus_velocity.png` | PNG | Magnus velocity magnitude (wake deflection) |
| `cylinder_compare_velocity.png` | PNG | Side-by-side velocity comparison |
| `cylinder_nospin_pressure.mp4` | MP4 | 10s animation, pressure, no-spin (3.3M) |
| `cylinder_magnus_pressure.mp4` | MP4 | 10s animation, pressure, magnus (3.2M) |
| `cylinder_nospin_velocity.mp4` | MP4 | 10s animation, velocity, no-spin (3.0M) |
| `cylinder_magnus_velocity.mp4` | MP4 | 10s animation, velocity, magnus (3.1M) |
| `cylinder_compare_flow.mp4` | MP4 | 10s side-by-side comparison (4.3M) |
| `cylinder_nospin_3d.html` | HTML | Interactive 3D pressure field, no-spin (2.9M) |
| `cylinder_magnus_3d.html` | HTML | Interactive 3D pressure field, magnus (2.9M) |
| `cylinder_unsteady_cl_comparison.png` | PNG | Cl(t) evolution, all 4 cases |
| `cylinder_pressure_field.png` | PNG | Steady RANS pressure (Phase 3) |
| `cylinder_velocity_field.png` | PNG | Steady RANS velocity (Phase 3) |
| `cylinder_cd_convergence.png` | PNG | Steady RANS Cd convergence (Phase 3) |
| `cylinder_streamlines.png` | PNG | Steady RANS streamlines |
| `cylinder_nospin_streamlines.png` | PNG | Unsteady laminar streamline snapshot |
| `cylinder_magnus_streamlines.png` | PNG | Unsteady magnus streamline snapshot |
| `cylinder_pressure_comparison.png` | PNG | Steady: no-spin vs magnus pressure |
| `cylinder_streamline_comparison.png` | PNG | Steady: no-spin vs magnus streamlines |

## Progress Status
| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (SU2, gmsh, PyVista, trame install) | ✅ |
| 1 | Website (5 HTML pages, dark theme, KaTeX, Prism, video/iframe CSS) | ✅ |
| 2 | SU2 Runner Module (config, meshgen with wake refine, solver, viz) | ✅ |
| 3a | 2D cylinder steady RANS | ✅ Cd=0.683 |
| 3b | Analysis & comparison with ΦFlow (~1.2) | ✅ 43% diff explained |
| 3+ | Unsteady URANS SST + laminar NS (no-spin vs magnus) | ✅ Both regimes done |
| 4 | 3D sphere drag crisis (Re sweep) | ✅ Cd≈0.87 constant (no drag crisis) |
| 5 | Smooth vs textured sphere | 🔜 Next |
| 6 | Deformed ball, altitude effects | ⏳ |

## Key Results
- **SU2 2D cylinder @ Re=40k (RANS SST)**: Cd=0.683, Cl=-0.032
- **ΦFlow 2D cylinder @ Re=40k (laminar)**: Cd≈1.2
- **43% difference**: Fully turbulent RANS delays separation (~110° vs ~80°), produces narrower wake and lower drag
- **Unsteady URANS SST**: No shedding (eddy viscosity damps instability); Cl ±0.03
- **Unsteady laminar NS @ Re=120**: St=0.16 (Lit: ~0.17); Cl ±0.05 (mesh-limited amplitude); magnus bias ≈ -0.15
- **3D sphere SST**: Cd≈0.87 constant (no drag crisis); 282k tets, y⁺≈13 insufficient
- **Steady magnus (S=0.2)**: Cl=-0.172 (steady bias)

## Config templates
### Steady RANS SST (Re=40k, Phase 3)
`INC_NONDIM= INITIAL_VALUES`, `MU_CONSTANT= 2.5e-05`, `MUSCL_FLOW= NO`

### Unsteady Laminar NS (Re=120, Phase 3+)
`INC_NONDIM= DIMENSIONAL`, `MU_CONSTANT= 0.00267875` (dimensional), `INC_DENSITY_INIT= 1.2886`, `INC_VELOCITY_INIT= (1.0, 0.0, 0.0)`, `MUSCL_FLOW= YES`, `SLOPE_LIMITER_FLOW= NONE`

## Reference Commands
```bash
# Run cylinder validation
uv run python su2_runs/cylinder_2d/run_cylinder.py

# Run SU2 directly (from workdir)
su2_runs/cylinder_2d/ && /Users/ajeet/SU2_CFD/bin/SU2_CFD cylinder.cfg

# Generate viz
uv run python su2_runs/cylinder_2d/generate_viz.py

# Run unsteady laminar
uv run python su2_runs/cylinder_2d/run_unsteady.py
```
