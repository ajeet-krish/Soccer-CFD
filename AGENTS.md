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
  index.html, theory.html, shot.html, tactical.html
  cfd.html, code.html
  custom.css                   # Dark theme + responsive nav
  images/
    phiflow_cylinder_2d/       # ΦFlow cylinder visualizations (MP4)
    tactical/                  # Tactical simulations
      formation_pressure/      # 4 formations, Gaussian pressure fields (PNG)
      formation_lanes/         # 5-lane pressure analysis per formation (PNG)
      formation_transition/    # Offence to Defence transition videos (MP4)
    su2_cylinder_2d/           # SU2 2D cylinder (PNG, MP4)
      re120/                   #   Re=120 laminar (no-spin + magnus)
      re200/                   #   Re=200 laminar
      re500/                   #   Re=500 laminar
      steady_rans/             #   Steady RANS comparison images
      comparisons/             #   Re sweep comparison animations
      analysis/                #   Cp(θ) + separation angle analysis
      mesh/                    #   Mesh visualizations
    su2_sphere/                # SU2 3D sphere (PNG)
su2_runs/
  cylinder_2d/                 # Phase 3+ — 2D cylinder validation
    run_cylinder.py            # Steady-state mesh → config → SU2 → results
    run_unsteady.py            # Unsteady laminar Re=120/200/500 (no-spin + magnus)
    generate_viz.py            # Static plots + animation renderer
    analyze_cylinder.py        # Cp(θ) extraction + separation angle detection
    cylinder.cfg               # Steady RANS SST config (coarse mesh)
    cylinder_magnus.cfg        # Steady RANS magnus config
    cylinder_fine_rans.cfg     # (attempted) Fine-mesh RANS config
    output_*_lam_re*_fine/     # 6 dirs, each 42MB (1 VTU + 1 DAT + history CSV)
  sphere_3d/                   # Phase 4+5 — 3D sphere
    run_sphere.py              # Smooth sphere Re sweep
    run_roughness.py           # 4 balls × 5 Re roughness sweep
    viz_sphere.py              # PyVista visualization pipeline
    sphere.su2                 # 282K tetrahedral mesh
  tactical_formations/
    generate_formation_pressure.py  # Gaussian pressure field generator
src/
  su2_runner.py                # SU2Config (incl wall_roughness), MeshGenerator, SU2Solver, PyVista viz
  build_site.py                # Nav-sync utility (optional)
linkedin_posts/                # LinkedIn content drafts
AGENTS.md                      # This file
```

## Visualizations Generated
| File | Type | Description |
|------|------|-------------|
| `phiflow_cylinder_2d/pressure_comparison.mp4` | MP4 | ΦFlow sidecar: Magnus vs Knuckleball pressure |
| `phiflow_cylinder_2d/velocity_comparison.mp4` | MP4 | ΦFlow sidecar: Magnus vs Knuckleball velocity |
| `phiflow_cylinder_2d/phiflow_knuckleball_velocity_frame095.png` | PNG | ΦFlow knuckleball velocity at t=85.5s (interactive slider) |
| `phiflow_cylinder_2d/phiflow_magnus_velocity_frame095.png` | PNG | ΦFlow magnus velocity at t=85.5s (interactive slider) |
| `su2_cylinder_2d/mesh/cylinder_mesh.png` | PNG | Full domain mesh (224K) |
| `su2_cylinder_2d/mesh/cylinder_mesh_zoom.png` | PNG | Cylinder close-up with wake refinement (1.3M) |
| `su2_cylinder_2d/re120/nospin/cylinder_nospin_pressure_re120.png` | PNG | No-spin pressure at t=89.7s, Re=120 |
| `su2_cylinder_2d/re120/magnus/cylinder_magnus_pressure_re120.png` | PNG | Magnus pressure at t=89.7s, Re=120 |
| `su2_cylinder_2d/re120/nospin/cylinder_nospin_velocity_re120.png` | PNG | No-spin velocity, Re=120 |
| `su2_cylinder_2d/re120/magnus/cylinder_magnus_velocity_re120.png` | PNG | Magnus velocity, Re=120 |
| `su2_cylinder_2d/re*/nospin/cylinder_nospin_pressure_re*.png` | PNG | Per-Re no-spin pressure, Re=120/200/500 |
| `su2_cylinder_2d/re*/magnus/cylinder_magnus_pressure_re*.png` | PNG | Per-Re magnus pressure, Re=120/200/500 |
| `su2_cylinder_2d/steady_rans/cylinder_pressure_field.png` | PNG | Steady RANS pressure (Phase 3) |
| `su2_cylinder_2d/steady_rans/cylinder_velocity_field.png` | PNG | Steady RANS velocity (Phase 3) |
| `su2_cylinder_2d/steady_rans/cylinder_cd_convergence.png` | PNG | Steady RANS Cd convergence (Phase 3) |
| `su2_cylinder_2d/steady_rans/cylinder_streamlines.png` | PNG | Steady RANS streamlines |
| `su2_cylinder_2d/steady_rans/cylinder_pressure_comparison.png` | PNG | Steady RANS: no-spin vs magnus pressure |
| `su2_cylinder_2d/steady_rans/cylinder_streamline_comparison.png` | PNG | Steady RANS: no-spin vs magnus streamlines |
| `su2_cylinder_2d/analysis/cylinder_cp_comparison.png` | PNG | Cp(θ) overlay all 6 cases with separation angles |
| `su2_cylinder_2d/comparisons/cylinder_nospin_pressure_re_comparison.mp4` | MP4 | Re sweep: no-spin pressure (Re=120/200/500) |
| `su2_cylinder_2d/comparisons/cylinder_magnus_pressure_re_comparison.mp4` | MP4 | Re sweep: magnus pressure (Re=120/200/500) |
| `su2_cylinder_2d/comparisons/cylinder_nospin_velocity_re_comparison.mp4` | MP4 | Re sweep: no-spin velocity (Re=120/200/500) |
| `su2_cylinder_2d/comparisons/cylinder_magnus_velocity_re_comparison.mp4` | MP4 | Re sweep: magnus velocity (Re=120/200/500) |
| `su2_sphere/sphere_cd_re.png` | PNG | Smooth sphere Cd(Re) with experimental overlay |
| `su2_sphere/sphere_roughness_cd_re.png` | PNG | Roughness sweep Cd(Re): 4 ball types |
| `su2_sphere/sphere_roughness_cd_vs_ks.png` | PNG | Cd vs kₛ/D at each Re |
| `tactical/formation_comparison.png` | PNG | 2x2 comparison grid, 4 formations, turbo colormap |
| `tactical/formation_pressure/433.png` | PNG | 4-3-3 Gaussian defensive pressure field |
| `tactical/formation_pressure/442.png` | PNG | 4-4-2 Gaussian defensive pressure field |
| `tactical/formation_pressure/4231.png` | PNG | 4-2-3-1 Gaussian defensive pressure field |
| `tactical/formation_pressure/352.png` | PNG | 3-5-2 Gaussian defensive pressure field |
| `tactical/formation_lanes/433.png` | PNG | 4-3-3 five-lane pressure analysis with percentages |
| `tactical/formation_lanes/442.png` | PNG | 4-4-2 five-lane pressure analysis with percentages |
| `tactical/formation_lanes/4231.png` | PNG | 4-2-3-1 five-lane pressure analysis with percentages |
| `tactical/formation_lanes/352.png` | PNG | 3-5-2 five-lane pressure analysis with percentages |
| `tactical/formation_transition/attacking_to_defending_433.mp4` | MP4 | 4-3-3 attacking→defending morph animation |
| `tactical/formation_transition/attacking_to_defending_442.mp4` | MP4 | 4-4-2 attacking→defending morph animation |
| `tactical/formation_transition/attacking_to_defending_4231.mp4` | MP4 | 4-2-3-1 attacking→defending morph animation |
| `tactical/formation_transition/attacking_to_defending_352.mp4` | MP4 | 3-5-2 attacking→defending morph animation |

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
| 5 | Textured sphere roughness sweep (4 balls × 5 Re) | ✅ No effect (Cd varies &lt;0.001) |
| 6 | γ-Reθ transition model study | ❌ Negative — see below |
| 7 | Deformed ball, altitude effects | ⏳ |

## Key Results
- **SU2 2D cylinder @ Re=40k (RANS SST)**: Cd=0.683, Cl=-0.032
- **ΦFlow 2D cylinder @ Re=40k (laminar)**: Cd≈1.2
- **43% difference**: Fully turbulent RANS delays separation (~110° vs ~80°), produces narrower wake and lower drag
- **Unsteady URANS SST**: No shedding (eddy viscosity damps instability); Cl ±0.03
- **Unsteady laminar NS @ Re=120**: St=0.16 (Lit: ~0.17); Cl ±0.05 (mesh-limited amplitude); magnus bias ≈ -0.15
- **3D sphere SST**: Cd≈0.87 constant (no drag crisis); 282k tets, y⁺≈13 insufficient
- **Steady magnus (S=0.2)**: Cl=-0.172 (steady bias)
- **Overlapping fullback**: 26.1% drag reduction (tandem bluff-body drafting)
- **Formation pressure maps**: 4-3-3 center 81% (wings open), 4-4-2 half-spaces 68% (balanced), 4-2-3-1 center 93% (congested middle, wings 32%), 3-5-2 center 100% (five-player central wall) with wings at 21% (extremely exposed flanks)
- **γ-Reθ on 2D cylinder — negative (3 attempts)**: (1) y+≈13 coarse mesh → Cd=0.645 (all 3 Re, identical to SST fully turbulent; sensors don't trigger). (2) y+≈1 structured mesh steady → Cd drifts 1.36→1.34/8k iters, never converges (mesh too well-resolved, von Kármán develops but steady solver suppresses it). (3) y+≈1 URANS → Cd decays 3.92→-0.04 over 300 steps, Cl=0 always (SST eddy viscosity precludes separation, no LSB forms). **Root cause**: γ-Reθ works for attached-flow transition only. Cylinder separation-induced transition requires DDES/LES — RANS-based transition models pre-emptively damp the separated shear layer, preventing LSB formation. Correct approaches for this toolchain: laminar NS (Re<500, St validated), steady RANS SST (Re>40k, Cd validated).

## Tactical Page Structure
```
h1 Tactical Positioning
  h2 Overview
  h2 Formation Pressure Maps
    h3 4-3-3
    h3 4-4-2
    h3 4-2-3-1
    h3 3-5-2
    h3 Offence to Defence Transition
  h2 Formation Aerodynamics
    (Overlapping fullback science only — tandem drag reduction, 26.1%)
```

## Phase 5: Textured Sphere Roughness Sweep

Four ball types compared via SU2 `WALL_ROUGHNESS` model:

| Ball | Year | Panels | Seam Depth | kₛ/D | Source |
|------|------|--------|-----------|------|--------|
| Smooth | — | ∞ | 0.0 mm | 0.0 | — |
| Jabulani | 2010 | 8 | ~1.5 mm | 0.007 | Goff et al. 2022 |
| Brazuca | 2014 | 6 | ~3.0 mm | 0.014 | Alam et al. 2016 |
| Trionda | 2026 | 4 | ~4.5 mm | 0.020 | MDPI Appl Sci 2026 |

- **Re sweep**: 1×10⁴, 1×10⁵, 3×10⁵, 6×10⁵, 1×10⁶ (4 balls × 5 Re = 20 runs)
- **Method**: `WALL_ROUGHNESS= ( wall, <kₛ> )` in SU2 config
- **Result (negative)**: Cd varies by &lt;0.001 across all ball types at all Re. RANS SST + y+≈13 = no roughness sensitivity. Three factors: (1) SST is fully turbulent, no laminar regime for roughness to trip; (2) y+≈13 is buffer layer, too high for wall-resolved (y+<1) and too low for wall-function roughness (y+>30); (3) separation point already fixed by turbulence model at ~120°.
- **Fix**: Transition model (γ-Reθ) + wall-resolved mesh (prism layers, y+<1). With these, smooth ball separates laminar at ~80° (high Cd), rough ball triggers transition → turbulent separation at ~110° (low Cd), reproducing the drag crisis.
- **Runner**: `su2_runs/sphere_3d/run_roughness.py`
- **Output**: `sphere_roughness_cd_re.png`, `sphere_roughness_cd_vs_ks.png`, `results_roughness.json`

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

# Extract Cp(θ) + separation angles
uv run python su2_runs/cylinder_2d/analyze_cylinder.py

# Run sphere roughness sweep (4 balls × 5 Re = 20 runs)
uv run python su2_runs/sphere_3d/run_roughness.py

# Generate tactical formation pressure maps
uv run python su2_runs/tactical_formations/generate_formation_pressure.py
```
