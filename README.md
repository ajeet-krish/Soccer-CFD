# The Physics of Play

**Aerodynamics in Soccer & FIFA World Cup 2026**

[**View the portfolio site →**](https://ajeet-krish.github.io/Soccer-CFD/)

This project explores fluid dynamics through soccer, building CFD simulations with &Phi;Flow and SU2 to reveal the hidden structure in everyday play.

---

### 1. Theory — The Physics Behind the Simulations

What is CFD and how the simulations work under the hood: from the Navier-Stokes equations to the Magnus effect, Reynolds number, and the Strouhal number that governs vortex shedding.

Key topics: boundary layers, separation, the Magnus effect, Bernoulli's principle, and how these translate to soccer scenarios.

[→ Theory page](https://ajeet-krish.github.io/Soccer-CFD/theory.html)

---

### 2. CFD Methodology — How the Simulations Are Built

A two-solver pipeline designed for both speed and accuracy:

| Layer | Tool | Fidelity | Purpose |
|-------|------|----------|---------|
| Rapid prototyping | &Phi;Flow (2D) | Laminar, no turbulence model | 100+ parametric sweeps, identify interesting regimes |
| High-fidelity validation | SU2 (2D / 3D) | RANS SST $k$-$\omega$, unstructured mesh | Accurate $C_d$, $C_l$, boundary layer profiles |
| Visualization | PyVista | N/A | 3D pressure contours, stream ribbons, comparison renders |

All meshes are generated programmatically with gmsh's Python API, combining distance-based wall refinement with a wake refinement box for vortex resolution. The full mesh has 160k nodes and 321k triangular elements.

Six unsteady laminar cases (Re=120, 200, 500 × no-spin, Magnus) validate the solver against Williamson 1988 and Tritton 1959 — Strouhal number matches to within 5%. Separation angles extracted from surface $C_p$ curves confirm the physical trend (87° → 97° → 103°).

[→ CFD page](https://ajeet-krish.github.io/Soccer-CFD/cfd.html)

---

### 3. Shot Aerodynamics — Magnus vs Knuckleball

Animations comparing the Magnus effect (topspin) against the knuckleball (no-spin) on a 2D cylinder proxy for a soccer ball at Re ≈ 4×10⁴.

- **Magnus**: Rotating cylinder at $\omega = 10$ rad/s ($S = 3.0$). The asymmetric wake produces a downward force, bending the trajectory.
- **Knuckleball**: Non-rotating. Alternating vortex shedding creates an unsteady wake, producing the unpredictable "dancing" motion of a knuckleball free kick.
- **SU2 validation**: RANS SST predicts $C_d = 0.68$ (fully turbulent), while &Phi;Flow predicts $C_d \approx 1.2$ (laminar) — the 43% difference reflects the physical distinction between laminar and turbulent separation. Together they bracket the real behavior of a soccer ball in flight.

[→ Shooting page](https://ajeet-krish.github.io/Soccer-CFD/shot.html)

---

### 4. Tactical Positioning — Formation Pressure Maps

Players modeled as Gaussian influence fields to examine defensive pressure across five vertical lanes. Four formations compared:

| Formation | Center pressure | Flanks |
|-----------|----------------|--------|
| 4-3-3 | 81% | Wings open, wide attacking options |
| 4-4-2 | 68% | Balanced half-spaces |
| 4-2-3-1 | 93% | Congested middle, wings at 32% |
| 3-5-2 | 100% | Central wall, wings at 21% |

Animated offence-to-defence transitions show how space closes during the counter-attack.

[→ Tactical page](https://ajeet-krish.github.io/Soccer-CFD/tactical.html)

---

### 5. Overlap Run Analysis — Wake Drafting

CFD analysis of the overlapping fullback: two rectangular bluff bodies in tandem at Re ≈ 2×10⁴. The downstream player (fullback) enters the winger's low-pressure wake, producing significant drag reduction:

| Configuration | Gap | Drag reduction |
|--------------|-----|---------------|
| Inline | 2 m | 42.2% |
| Inline | 3 m | 37.3% |
| Inline | 4 m | 27.8% |
| Offset (0.35 m lateral) | 2 m | 3.9% |

The drafting effect — familiar from cycling and motorsport — is now quantified for soccer. A lateral offset of just 0.35 m drops the benefit from 42.2% to 3.9%, showing how critical the in-line overlap path is for conserving energy.

[→ Overlap Run page](https://ajeet-krish.github.io/Soccer-CFD/overlap.html)

---

## Key Results & Honest Limitations

- **2D Cylinder (Re=40k)**: SU2 Cd=0.683 vs ΦFlow Cd≈1.2 — the 43% gap is physically meaningful (laminar vs turbulent separation). Neither is perfect for real soccer balls; together they bracket the truth.
- **3D Sphere (Smooth, Re 10⁴→10⁶)**: SST k-ω produces Cd≈0.87 flat. No drag crisis — transition model + wall-resolved mesh required.
- **Textured Sphere (4 balls × 5 Re)**: WALL_ROUGHNESS produces no effect (Cd varies &lt;0.001). Modelling limit: needs γ-Reθ + y+&lt;1.
- **Unsteady laminar NS (Re=120/200/500)**: St matches Williamson 1988 to within 5%. Cd matches Tritton 1959. Magnus Cl bias up to -0.39.
- **Overlapping Fullback**: 42.2% drag reduction at 2 m gap (inline); 3.9% at 0.35 m lateral offset. Gap sweep: 42.2% → 37.3% → 27.8% — monotonic decay as trailer exits leader's wake.
- **Mesh limitation**: Shedding amplitude is underpredicted (±0.05 vs literature ±0.5) — numerical dissipation from unstructured triangles damps vortex cores. Structured O-grid or prism layers would recover the full amplitude.

---

## Getting Started

```bash
# Install dependencies
uv sync

# Run ΦFlow simulations (shot + tactical)
uv run python src/build_all.py

# SU2 2D cylinder validation (requires SU2 v8.4)
uv run python su2_runs/cylinder_2d/run_unsteady.py
uv run python su2_runs/cylinder_2d/analyze_cylinder.py

# SU2 3D sphere sweeps
uv run python su2_runs/sphere_3d/run_sphere.py
uv run python su2_runs/sphere_3d/run_roughness.py

# Tactical formation pressure maps
uv run python src/tactical_formations/generate_formation_pressure.py

# Overlap drafting analysis
uv run python src/tactical_formations/wake_drafting.py
uv run python src/tactical_formations/gap_sweep.py

# Preview the site locally
python -m http.server -d docs 8000
```

## Requirements

- Python 3.12, **uv** package manager
- **ffmpeg** — MP4 export (`brew install ffmpeg`)
- &Phi;Flow (≥3.4), JAX, matplotlib, numpy, tqdm, pyvista, gmsh, Pillow
- SU2 v8.4 "Harrier" — CFD solver (validation only, not needed for browsing)

---

## Project Structure

```
src/                       # Python source code
├── build_all.py           # ΦFlow orchestrator
├── shot_aero.py           # Magnus + Knuckleball simulations
├── tactical.py            # Overlapping fullback (ΦFlow)
├── su2_runner.py          # SU2 config, meshgen, solver, viz
├── domain.py              # Grid and domain constants
└── utils.py               # Shared helpers

su2_runs/                  # SU2 validation scripts and configs
├── cylinder_2d/           # 2D cylinder (steady + unsteady RANS)
├── sphere_3d/             # 3D sphere (smooth + roughness sweep)
└── tactical_formations/   # Formation pressure + drafting analysis

docs/                      # GitHub Pages site (7 HTML pages)
├── index.html             # Landing page
├── theory.html            # Fluid dynamics fundamentals
├── shot.html              # Shot aerodynamics
├── tactical.html          # Formation pressure maps
├── overlap.html           # Overlap run analysis
├── cfd.html               # CFD methodology & validation
├── code.html              # Code notebook
├── custom.css             # Dark terminal theme
└── images/                # All visualizations
    ├── phiflow_cylinder_2d/
    ├── tactical/           # Formation lanes, transitions, drafting
    ├── su2_cylinder_2d/    # Per-Re results, mesh, comparisons
    └── su2_sphere/         # Sphere drag curves
```
