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
| 2 | **Tactical Positioning** | Navier-Stokes (248×186) | 26.1% drag reduction via wake shielding (overlapping fullback) |

### SU2 Validation

| # | Case | Status | Key Result |
|---|------|--------|-----------|
| 1 | **2D Cylinder** (Re=40k, RANS SST) | ✅ Complete | Cd=0.683 vs ΦFlow~1.2 — 43% diff explained |
| 2 | **3D Smooth Sphere** (Re sweep, RANS SST) | ✅ Complete | Cd≈0.87 flat across Re (no drag crisis — SST limitation) |
| 3 | **Textured Sphere Roughness Sweep** (4 balls × 5 Re) | 🔜 Phase 5 | Roughness trends via WALL_ROUGHNESS model |

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

The fluid-dynamic basis of the overlapping run: two rectangular bluff bodies in tandem at Re = 2×10⁵. The downstream player (fullback) enters the winger's low-pressure wake, producing **26.1% drag reduction** — the drafting effect familiar from cycling and motorsport, now quantified for soccer.

| Parameter | Value |
|-----------|-------|
| Player width | 0.3 m |
| Player height | 1.8 m |
| Streamwise spacing | 2.0 m |
| Inlet velocity | 10 m/s |
| Drag reduction | 26.1% |

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

RANS SST with roughness shows separation-point shift trends (~5-15% Cd variation) but cannot capture the full drag crisis drop (0.47→0.07) — that requires a transition model (γ-Reθ) + wall-resolved mesh.

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
│   ├── run_cylinder.py    # Mesh → config → SU2 → results pipeline
│   ├── run_magnus.py
│   ├── run_unsteady.py
│   └── results.json
├── sphere_3d/             # Phase 4 — smooth sphere drag crisis
│   ├── run_sphere.py
│   ├── viz_sphere.py      # PyVista visualization pipeline
│   └── results.json
└── tactical_formations/
    └── generate_formation_pressure.py  # Gaussian pressure field generator
docs/                      # GitHub Pages site
├── index.html             # Landing page hub
├── theory.html            # Background fluid dynamics theory
├── shot.html              # Shot aerodynamics (ΦFlow + SU2)
├── tactical.html          # Tactical positioning & formation pressure maps
├── cfd.html               # CFD methodology & validation
├── code.html              # Jupyter-style code notebook
├── custom.css             # Dark terminal theme + responsive nav
└── images/                # Generated visualizations
    ├── phiflow_cylinder_2d/
    ├── tactical/
    │   ├── formation_lanes/
    │   ├── formation_pressure/
    │   └── formation_transition/
    ├── su2_cylinder_2d/
    └── su2_sphere/
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
uv run python su2_runs/cylinder_2d/run_cylinder.py
uv run python su2_runs/sphere_3d/run_sphere.py
uv run python su2_runs/sphere_3d/viz_sphere.py
```

**Tactical formation pressure maps:**
```bash
uv run python su2_runs/tactical_formations/generate_formation_pressure.py
```

**Website (local preview):**
```bash
python -m http.server -d docs 8000
open http://localhost:8000
```

## Requirements

- **Python** 3.12, **uv** package manager
- **ffmpeg** — MP4 export (`brew install ffmpeg`)
- **ΦFlow** (≥3.4), JAX, matplotlib, numpy, tqdm, pyvista, gmsh
- **SU2 v8.4** — CFD solver (validation only, not needed for browsing)

## Key Results & Honest Limitations

- **2D Cylinder**: SU2 Cd=0.683 vs ΦFlow Cd≈1.2 at Re=40k. The 43% gap is physically meaningful: fully turbulent RANS delays separation, narrowing the wake.
- **3D Sphere (Smooth)**: SST k-ω produces Cd≈0.87 flat across Re. No drag crisis. A transition model (γ-Reθ) + wall-resolved mesh would be required for crisis capture.
- **Textured Sphere**: Phase 5 will use roughness wall functions to show separation-point shift trends.
- **Overlapping Fullback**: 26.1% drag reduction quantified — the fluid-dynamic basis for why overlapping runs save energy.
