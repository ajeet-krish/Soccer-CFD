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
| 3 | **Tactical Stress Fields** | Gaussian superposition + continuum mechanics | Two-phase flow interface & Von Mises yield detection in 7v7 |
| 4 | **Team Heatmaps** | Gaussian influence fields + Darcy permeability | Passing lane identification via influence thresholding |

### SU2 Validation

| # | Case | Status | Key Result |
|---|------|--------|-----------|
| 1 | **2D Cylinder** (Re=40k, RANS SST) | ✅ Complete | Cd=0.683 vs ΦFlow~1.2 — 43% diff explained |
| 2 | **3D Smooth Sphere** (Re sweep, RANS SST) | ✅ Complete | Cd≈0.87 flat across Re (no drag crisis — SST limitation) |
| 3 | **Textured Sphere Roughness Sweep** (4 balls × 5 Re) | 🔜 Phase 5 | Roughness trends via WALL_ROUGHNESS model |

## Simulation Details

### 1. Shot Aerodynamics — Magnus vs Knuckleball

A cylinder in crossflow at Re ≈ 2×10⁵, comparing a rotating case (ω = 10 rad/s) with a non-rotating case (ω = 0).

- **Domain**: 8.0 × 4.0 m, cylinder radius 0.3 m at (2.0, 2.0)
- **Grid**: 256 × 128, staggered (MAC)
- **Solver**: Semi-Lagrangian advection + CG pressure solve
- **Inlet**: Uniform 1.0 m/s from left (x⁻) boundary

### 2. Tactical Positioning — Formation Aerodynamics

Players modelled as rectangular bluff bodies in crossflow (10 m/s inlet).

| Case | Configuration | Metric |
|------|-------------|--------|
| 1 — Isolated Winger | Single player at (2.0, 3.0) | Baseline drag |
| 2 — Midfield Press Wall | Three players at x = 2.0, y = {1.5, 3.0, 4.5} | Vorticity amplification |
| 3 — Overlapping Fullback | Two players in tandem at x = {2.0, 4.0}, y = 3.0 | 26.1% drag reduction |

### 3. SU2 2D Cylinder Validation

SU2 RANS SST on a circular cylinder at Re=40k, matching the ΦFlow shot domain.

- **Mesh**: gmsh-generated, 2D quadrilateral-dominant
- **Solver**: INC_RANS, SST turbulence model, FDS convective scheme
- **Results**: Cd=0.683 (SU2) vs ~1.2 (ΦFlow) — difference explained by fully turbulent RANS delaying separation (~110° vs ~80°), producing a narrower wake

### 4. SU2 3D Sphere: Drag Crisis (Smooth)

Steady RANS (SST k-ω) on a smooth sphere sweeping Re = 10⁴ → 10⁶.

- **Mesh**: Coarse tetrahedral (45K nodes, 282K tets, y+ ~13)
- **Limitation**: SST is fully turbulent — no drag crisis captured (Cd ≈ 0.87 flat). This is consistent with RANS behaviour and documented honestly

### 5. SU2 3D Sphere: Textured Ball Roughness (Planned)

Comparing four ball types (smooth, Jabulani, Brazuca, Trionda) via `WALL_ROUGHNESS` model across Re = 10⁴ → 10⁶. Seam depths from peer-reviewed literature.

## Project Structure

```
src/
├── build_all.py           # ΦFlow orchestrator
├── build_site.py          # Nav-sync utility
├── shot_aero.py           # Magnus + Knuckleball (cylinder in crossflow)
├── tactical.py            # Tactical positioning + stress fields + team heatmaps
├── domain.py              # Domain constants (grid, bounds, dt)
├── su2_runner.py          # SU2Config, MeshGenerator, SU2Solver, PyVista viz
└── utils.py               # Shared theme, paths, imports
su2_runs/
├── cylinder_2d/           # Phase 3 — 2D cylinder validation
│   ├── run_cylinder.py    # Mesh → config → SU2 → results pipeline
│   ├── run_magnus.py
│   ├── run_unsteady.py
│   └── results.json
└── sphere_3d/             # Phase 4 — smooth sphere drag crisis
    ├── run_sphere.py
    ├── viz_sphere.py      # PyVista visualization pipeline
    ├── results.json
    └── vol_solution.vtu   # Final field solution (Re=1e6)
docs/                      # GitHub Pages site
├── index.html             # Landing page hub
├── theory.html            # Background fluid dynamics theory
├── shot.html              # Shot aerodynamics
├── tactical.html          # Tactical positioning
├── su2-validation.html    # SU2 validation narrative
├── custom.css             # Dark theme + responsive nav
└── images/                # Generated visualizations
assets/                    # ΦFlow-generated MP4s and PNGs
```

## Getting Started

**ΦFlow simulations:**
```bash
uv sync
uv run python src/build_all.py
uv run jupyter notebook
```

**SU2 validation (requires SU2 v8.4 installed):**
```bash
# 2D cylinder
uv run python su2_runs/cylinder_2d/run_cylinder.py

# 3D sphere
uv run python su2_runs/sphere_3d/run_sphere.py

# Sphere visualizations
uv run python su2_runs/sphere_3d/viz_sphere.py
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
- **SU2 v8.4** — CFD solver (validation only, not needed for browsing the site)

## Key Results & Honest Limitations

- **2D Cylinder**: SU2 Cd=0.683 vs ΦFlow Cd≈1.2 at Re=40k. The 43% gap is physically meaningful: fully turbulent RANS delays separation, narrowing the wake.
- **3D Sphere (Smooth)**: SST k-ω produces Cd≈0.87 flat across Re. No drag crisis — this is a known limitation of fully turbulent RANS. A transition model (γ-Reθ) + wall-resolved mesh would be required for crisis capture.
- **Textured Sphere**: Phase 5 will use roughness wall functions to show separation-point shift trends. Expect ~5–15% Cd variation, not a drag crisis drop.
