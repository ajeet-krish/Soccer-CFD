# The Physics of Play

**Aerodynamics in Summer Sports & FIFA World Cup 2026**

A dual-layer CFD portfolio combining fast ΦFlow (JAX) prototyping with SU2 (RANS) high-fidelity validation, built as a standalone HTML website.

[**View the portfolio site →**](https://ajeet-krish.github.io/Soccer-CFD/)

## Project Overview

| Layer | Tool | Role |
|-------|------|------|
| **Fast prototyping** | ΦFlow (2D, JAX) | Real-time sweeps, differentiable flow, tactical space optimization |
| **High-fidelity validation** | SU2 (3D RANS) | Production-grade aerodynamic coefficients, boundary layer analysis |

### ΦFlow Modules

| # | Module | Method | Key Result |
|---|--------|--------|-----------|
| 1 | **Shot Aerodynamics** | Navier-Stokes (256×128) | Magnus lift from rotating cylinder; knuckleball unsteady wake |
| 2 | **Tactical Positioning** | Navier-Stokes (256×128) | 42.2% drag reduction via wake shielding (overlapping fullback) |

### SU2 Validation

| # | Case | Status | Key Result |
|---|------|--------|-----------|
| 1 | **2D Cylinder** (Re=40k, RANS SST) | ✅ Complete | Cd=0.683 vs ΦFlow~1.2 — 43% diff explained |
| 2 | **3D Smooth Sphere** (Re sweep 10⁴→10⁶, RANS SST) | ✅ Complete | Cd≈0.87 flat across Re (no drag crisis — SST limitation) |
| 3 | **Textured Sphere Roughness Sweep** (4 balls × 5 Re) | ✅ Complete | Cd varies &lt;0.001 — roughness needs γ-Reθ + y+&lt;1 |

### Formation Pressure Maps

Each formation modelled as a superposition of Gaussian influence fields (one per defender), revealing defensive pressure patterns across five vertical lanes. Four formations compared: 4-3-3, 4-4-2, 4-2-3-1, 3-5-2. Animated offence-to-defence transitions show how space closes during the counter-attack.

## Simulation Details

### 1. Shot Aerodynamics — Magnus vs Knuckleball

A cylinder in crossflow at Re ≈ 2×10⁵, comparing a rotating case (ω = 10 rad/s) with a non-rotating case (ω = 0).

- **Domain**: 8.0 × 4.0 m, cylinder radius 0.3 m at (2.0, 2.0)
- **Grid**: 256 × 128, staggered (MAC)
- **Solver**: Semi-Lagrangian advection + CG pressure solve
- **Inlet**: Uniform 1.0 m/s from left (x⁻) boundary

### 2. Tactical Positioning — Overlapping Fullback

The fluid-dynamic basis of the overlapping run: two rectangular bluff bodies in tandem at Re ≈ 2×10⁴ (8&thinsp;m × 4&thinsp;m domain, U = 1&thinsp;m/s). The downstream player (fullback) enters the winger's low-pressure wake, producing **42.2% drag reduction** at 2&thinsp;m gap — the drafting effect familiar from cycling and motorsport, now quantified for soccer. A lateral offset of just 0.35&thinsp;m drops the benefit to 3.9%. A gap distance sweep (2&thinsp;m → 3&thinsp;m → 4&thinsp;m) shows monotonic decay: 42.2% → 37.3% → 27.8%.

| Parameter | Value |
|-----------|-------|
| Player width | 0.3 m |
| Player height | 1.8 m |
| Streamwise spacing | 2.0 m (inline), 0.35&thinsp;m lateral (offset) |
| Domain | 8.0 × 4.0 m |
| Grid | 256 × 128 |
| Inlet velocity | 1.0 m/s |
| Drag reduction (inline) | 42.2% |
| Drag reduction (offset) | 3.9% |

### 3. SU2 2D Cylinder Validation

SU2 RANS SST on a circular cylinder at Re=40k, matching the ΦFlow shot domain.

- **Mesh**: gmsh-generated, 2D quadrilateral-dominant
- **Solver**: INC_RANS, SST turbulence model, FDS convective scheme
- **Results**: Cd=0.683 (SU2) vs ~1.2 (ΦFlow) — difference explained by fully turbulent RANS delaying separation (~110° vs ~80°), producing a narrower wake

### 4. SU2 3D Sphere: Drag Crisis (Smooth)

Steady RANS (SST k-ω) on a smooth sphere sweeping Re = 10⁴ → 10⁶.

- **Mesh**: Coarse tetrahedral (45K nodes, 282K tets, y+ ~13)
- **Limitation**: SST is fully turbulent — no drag crisis captured (Cd ≈ 0.87 flat). Consistent with RANS behaviour.

### 5. SU2 3D Sphere: Textured Ball Roughness

Comparing four ball types via `WALL_ROUGHNESS` model across Re = 10⁴ → 10⁶:

| Ball | Year | Panels | Est. Seam Depth | kₛ/D | Source |
|------|------|--------|----------------|------|--------|
| Smooth (baseline) | — | ∞ | 0.0 mm | 0.0 | — |
| Jabulani | 2010 | 8 | ~1.5 mm | 0.007 | Goff et al. 2022 |
| Brazuca | 2014 | 6 | ~3.0 mm | 0.014 | Alam et al. 2016 |
| Trionda | 2026 | 4 | ~4.5 mm | 0.020 | MDPI Appl Sci 2026 |

RANS SST with `WALL_ROUGHNESS` produces no meaningful drag variation (Cd varies &lt;0.001 across all balls and Re). The roughness effect cannot be captured without a transition model (γ-Reθ) and a wall-resolved mesh (y+&lt;1, prism layers). This negative result is documented as a clear modeling-limit demonstration.

## Project Structure

```
src/
├── __init__.py
├── build_all.py           # ΦFlow orchestrator
├── build_site.py          # Nav-sync utility (optional)
├── domain.py              # Domain constants (grid, bounds, dt)
├── shot_aero.py           # Magnus + Knuckleball (cylinder in crossflow)
├── tactical.py            # Tactical positioning (overlapping fullback)
├── su2_runner.py          # SU2Config, MeshGenerator, SU2Solver, PyVista viz
└── utils.py               # Shared theme, paths, imports
su2_runs/
├── cylinder_2d/           # Phase 3 — 2D cylinder validation
│   ├── run_cylinder.py    # Mesh → config → SU2 → steady RANS
│   ├── run_unsteady.py    # Unsteady laminar NS (Re=120/200/500)
│   ├── generate_viz.py    # Static plots + animation pipeline
│   ├── analyze_cylinder.py  # Cp extraction + separation angles
│   ├── cylinder.cfg       # Steady RANS config (coarse mesh)
│   └── cylinder_magnus.cfg # Steady RANS magnus config
├── sphere_3d/             # Phase 4+5 — sphere drag crisis + roughness
│   ├── run_sphere.py      # Smooth sphere Re sweep
│   ├── run_roughness.py   # 4 balls × 5 Re roughness sweep
│   ├── viz_sphere.py      # PyVista visualization pipeline
│   └── sphere.su2         # Tetrahedral mesh (282K tets)
└── tactical_formations/
    ├── generate_formation_pressure.py  # Gaussian pressure field generator
    ├── wake_drafting.py                # ΦFlow wake drafting: solo/inline/offset comparison
    └── gap_sweep.py                    # Gap distance sweep: 2m/3m/4m inline
docs/                      # GitHub Pages site
├── index.html             # Landing page hub
├── theory.html            # Background fluid dynamics theory
├── shot.html              # Shot aerodynamics (ΦFlow + SU2)
├── tactical.html          # Tactical positioning & formation pressure maps
├── cfd.html               # CFD methodology & validation
├── code.html              # Jupyter-style code notebook
├── custom.css             # Dark terminal theme + responsive nav
└── images/                # Generated visualizations
    ├── phiflow_cylinder_2d/   # ΦFlow pressure/velocity comparisons
    ├── tactical/               # Formation pressure maps, transitions, drafting
    │   ├── formation_lanes/
    │   ├── formation_pressure/
    │   ├── formation_transition/
    │   └── drafting/           # Wake drafting: solo/inline/offset + gap sweep
    ├── su2_cylinder_2d/        # SU2 2D cylinder results
    │   ├── re120/              # Unsteady laminar @ Re=120 (no-spin + magnus)
    │   ├── re200/              # Unsteady laminar @ Re=200
    │   ├── re500/              # Unsteady laminar @ Re=500
    │   ├── steady_rans/        # Steady RANS comparison images
    │   ├── comparisons/        # Re sweep comparison animations
    │   ├── analysis/           # Cp(θ) + separation angle analysis
    │   └── mesh/               # Mesh visualization
    └── su2_sphere/             # SU2 3D sphere (smooth + roughness sweep)
assets/                    # ΦFlow frame cache (ignored)
linkedin_posts/            # LinkedIn content drafts (ignored)
```

## Getting Started

**ΦFlow simulations:**
```bash
uv sync
uv run python src/build_all.py
```

**SU2 validation (requires SU2 v8.4):**
```bash
# 2D cylinder steady RANS
uv run python su2_runs/cylinder_2d/run_cylinder.py

# 2D cylinder unsteady laminar (Re=120/200/500, no-spin + magnus)
uv run python su2_runs/cylinder_2d/run_unsteady.py

# Generate visualizations from VTU output
uv run python su2_runs/cylinder_2d/generate_viz.py

# Extract Cp(θ) + separation angles
uv run python su2_runs/cylinder_2d/analyze_cylinder.py

# 3D sphere smooth Re sweep
uv run python su2_runs/sphere_3d/run_sphere.py

# 3D sphere roughness sweep (4 balls × 5 Re)
uv run python su2_runs/sphere_3d/run_roughness.py

# PyVista 3D visualizations
uv run python su2_runs/sphere_3d/viz_sphere.py
```

**Tactical formation pressure maps:**
```bash
uv run python su2_runs/tactical_formations/generate_formation_pressure.py
```

**Overlap drafting analysis (ΦFlow, 256×128):**
```bash
# Solo/inline/offset comparison (13 outputs)
uv run python su2_runs/tactical_formations/wake_drafting.py

# Gap distance sweep (2m/3m/4m inline)
uv run python su2_runs/tactical_formations/gap_sweep.py
```

**Website (local preview):**
```bash
python -m http.server -d docs 8000
open http://localhost:8000
```

## Requirements

- **Python** 3.12, **uv** package manager
- **ffmpeg** — MP4 export (`brew install ffmpeg`)
- **ΦFlow** (≥3.4), JAX, matplotlib, numpy, tqdm, pyvista, gmsh, Pillow (PIL)
- **SU2 v8.4** — CFD solver (validation only, not needed for browsing)

## Key Results & Honest Limitations

- **2D Cylinder**: SU2 Cd=0.683 vs ΦFlow Cd≈1.2 at Re=40k. The 43% gap is physically meaningful: fully turbulent RANS delays separation, narrowing the wake.
- **3D Sphere (Smooth)**: SST k-ω produces Cd≈0.87 flat across Re. No drag crisis — transition model + wall-resolved mesh required.
- **Textured Sphere (4 balls × 5 Re)**: WALL_ROUGHNESS produces no effect (Cd varies &lt;0.001). Modeling limit: needs γ-Reθ + y+&lt;1.
- **Unsteady laminar NS (Re=120/200/500)**: St matches Williamson 1988 to within 5%. Cd matches Tritton 1959. Magnus Cl bias up to -0.39.
- **Surface Cp analysis**: Separation angles extracted: 87°→97°→103° (Re=120→200→500), confirming correct physical trend.
- **Overlapping Fullback**: 42.2% drag reduction at 2&thinsp;m gap (inline); 3.9% at 0.35&thinsp;m lateral offset. Gap sweep (2m/3m/4m): 42.2% → 37.3% → 27.8% — monotonic decay as trailer exits leader's wake. Each configuration generates velocity field animations with animated streamlines, static pressure images clipped to 99th percentile (seismic colormap), and drag-table output.
