# SU2 Integration Plan — Sports Aerodynamics

## Overview

Integrate SU2 (high-fidelity RANS CFD) alongside the existing ΦFlow simulations to create a validation pipeline:

| Layer | Tool | Role |
|-------|------|------|
| **Fast prototyping** | ΦFlow (2D) | Real-time sweeps, differentiable flow, tactical space optimization |
| **High-fidelity validation** | SU2 (3D RANS) | Production-grade aerodynamic coefficients (Cl, Cd), boundary layer resolution |

---

## Phases

### Phase 0 — Discovery & Infrastructure
- Check if SU2, gmsh, PyVista are installed
- Install anything missing
- Run SU2 test case to verify solver works
- Verify PyVista rendering

### Phase 1 — Website Foundation (Plain HTML+CSS+JS)
- Shared HTML layout (nav, footer, dark theme)
- KaTeX for LaTeX math
- Prism.js with copy-button for code blocks
- Hub page + skeleton module pages
- Replace Quarto output; live in `docs/` for GitHub Pages

### Phase 2 — SU2 Runner Module (`src/su2_runner.py`)
- Config class: generate `.cfg` files from Python parameters
- Mesh generation: wrap gmsh Python API (2D cylinder, 3D sphere)
- Solver invocation: spawn `SU2_CFD`, capture logs
- Results parser: extract Cl, Cd, CMz from history; convergence from residuals
- Callable from `build_all.py`

### Phase 3 — 2D Cylinder Validation (phiflow ↔ SU2 bridge)
- SU2 2D cylinder at Re ≈ 2×10⁵, matching phiflow domain
- Compare: Cd, Cp distribution, wake velocity profile
- Show phiflow approximation bias (e.g., phiflow Cd ≈ 1.2, SU2 Cd ≈ 0.9)
- PyVista renders: pressure + streamlines

### Phase 4 — 3D Sphere: Drag Crisis (Re Sweep)
- Steady RANS (SST k-ω) on smooth sphere
- Sweep Re: 10⁴ → 5×10⁵ → 10⁶
- Plot Cd(Re) against experimental data (Achenbach 1972)
- PyVista: 3D pressure contours, separation point viz

### Phase 5 — 3D Sphere: Roughness-Modeled Textured Ball Comparison

**Approach:** Surface roughness model via SU2 `WALL_ROUGHNESS` config option (equivalent sand-grain roughness). Captures boundary-layer modification from panel seams without complex mesh geometry. RANS SST + roughness shows separation-point shift trends but cannot capture the drag crisis drop — that requires a transition model (γ-Reθ) + wall-resolved mesh.

#### Ball Types Studied (With Citations)

| Ball | Year | Panels | Est. Seam Depth | Total Seam Length | kₛ/D | Source |
|------|------|--------|----------------|-------------------|------|--------|
| Smooth (baseline) | — | ∞ | 0.0 mm | — | 0.0 | — |
| Jabulani | 2010 | 8 | ~1.5 mm | ~2.0 m | 0.007 | Goff et al. 2022, *J Sports Eng Tech* |
| Brazuca | 2014 | 6 | ~3.0 mm | ~3.0 m | 0.014 | Alam et al. 2016, *Procedia Eng*; Goff et al. 2018 |
| Trionda | 2026 | 4 | ~4.5 mm | ~2.6 m | 0.020 | MDPI *Appl Sci* 2026, 16(6), 2808; COMSOL Blog 2026 |

**Rationale for selection:** These four span the full design space — smooth baseline → shallowest seams (Jabulani, notorious knuckleball) → moderate deep (Brazuca, most stable 6-panel) → deepest seams yet (Trionda, 2026 World Cup).

#### Reynolds Number Sweep
- Re = 1×10⁴ (deep subcritical), 1×10⁵ (approaching critical), 3×10⁵ (critical zone), 6×10⁵ (supercritical), 1×10⁶ (fully supercritical)
- 4 balls × 5 Re = 20 total runs (15 new beyond smooth baseline)

#### Implementation

1. **Code change** (`src/su2_runner.py:SU2Config`): Add `roughness_height: float = 0.0` field; emit `WALL_ROUGHNESS= ( wall, <value> )` when > 0
2. **Runner** (`su2_runs/sphere_3d/run_roughness.py`): Iterate ball types × Re values, run SU2, collect Cd/Cl
3. **Post-processing**: Multi-line Cd(Re) plot with one curve per ball type + Achenbach experimental overlay
4. **Visualization**: Cp(θ) comparison at Re = 3×10⁵ across roughness levels
5. **Website**: Update `docs/su2-validation.html` and `docs/shot.html` with results and narrative

#### Key Limitations (Documented for Portfolio)
- RANS SST is fully turbulent — roughness modifies wall function but does not trigger laminar-to-turbulent transition. Results show moderate Cd variation (~5–15%) from separation-point shift, not the dramatic 0.47→0.07 drag crisis.
- For true drag crisis capture, a transition model (Langtry-Menter γ-Reθ via `KIND_TRANS_MODEL= LM`) and wall-resolved mesh (prism layers, y+ < 1) would be required.

#### Visualization Files (Phase 4, Generated Pre-Roughness)
- `docs/images/sphere_pressure_surface.png` — 3D sphere Cp surface (isometric)
- `docs/images/sphere_pressure_plane.png` — Mid-plane Cp slice
- `docs/images/sphere_velocity_plane.png` — Mid-plane velocity magnitude
- `docs/images/sphere_streamlines.png` — Streamlines over velocity background
- `docs/images/sphere_surface_streamlines.png` — Combined Cp + 3D streamlines
- `docs/images/sphere_cd_re.png` — Cd(Re) comparison with experiment

#### Generation Script
```bash
su2_runs/sphere_3d/viz_sphere.py   # PyVista pipeline, run via `uv run python`
```

### Phase 6 — Portfolio Build & Deploy
- Generate final renders (.png, optional .html interactive exports)
- Write SU2 validation narrative into `su2-validation.html`
- Update `shot.html` with inline SU2 comparison
- Full site rebuild, local test, deploy to GitHub Pages

---

## Working Mode

One phase at a time. Each phase is proposed with detailed steps, reviewed, iterated if needed, then executed. No surprises.
