"""Phase 4: 3D sphere drag crisis — Re sweep using SU2 RANS SST."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, MeshGenerator, SU2Solver
from pathlib import Path

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_sphere"

# ── Reynolds numbers of interest ──
# Drag crisis for smooth sphere occurs around Re ~ 2-3×10^5
RE_VALUES = [1e4, 1e5, 2e5, 3e5, 5e5, 1e6]

# Frontal area for Cd normalization: A = π * D² / 4
SPHERE_DIAMETER = 0.22  # m
SPHERE_AREA = 3.14159265 * SPHERE_DIAMETER**2 / 4  # ~0.038 m²

# ── 1. Generate mesh (if not already present) ──
mesh_path = WORKDIR / "sphere.su2"
if not mesh_path.exists():
    print("=== Generating 3D sphere mesh ===")
    t0 = time.time()
    mesh_path = MeshGenerator.sphere_3d(
        radius=SPHERE_DIAMETER / 2,
        farfield_radius=3.0,
        cl_surface=0.004,
        cl_farfield=0.15,
        name=str(WORKDIR / "sphere"),
    )
    t1 = time.time()
    print(f"  Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes, {t1-t0:.1f}s)")
else:
    print(f"  Using existing mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")

# ── 2. Run SU2 at each Re ──
results = []

for Re in RE_VALUES:
    print(f"\n=== Re = {Re:.0e} ===")
    config = SU2Config.from_re(Re=Re, length=SPHERE_DIAMETER, incompressible=True)
    config.ref_area = SPHERE_AREA
    config.iterations = 400
    config.cfl_number = 0.5
    config.conv_residual_minval = -6
    config.screen_output = "WARNING"

    cfg_path = WORKDIR / f"sphere_Re{int(Re)}.cfg"
    config.write(cfg_path)

    solver = SU2Solver(workdir=WORKDIR)
    print(f"  Config written, launching SU2...")
    result = solver.run(cfg_path, mesh_path, timeout=1800)

    results.append({
        "Re": Re,
        "cd": result.cd,
        "cl": result.cl,
        "converged": result.converged,
        "iterations": result.iterations,
    })
    print(f"  SU2: Cd={result.cd:.4f}, Cl={result.cl:.4f}, converged={result.converged}")

# ── 3. Save structured results ──
output = {
    "case": "3D Sphere Drag Crisis",
    "phase": 4,
    "mesh": {"radius": 0.11, "farfield": 3.0, "file": str(mesh_path.name)},
    "results": results,
}
Path(WORKDIR / "results.json").write_text(json.dumps(output, indent=2))
print(f"\n=== Results saved to results.json ===")

# ── 4. Generate Cd(Re) plot ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Re_vals = np.array([r["Re"] for r in results])
Cd_vals = np.array([r["cd"] for r in results])

fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111111")
ax.set_facecolor("#1a1a1a")

ax.semilogx(Re_vals, Cd_vals, "o-", color="#f5a623", linewidth=2, markersize=8, label="SU2 RANS SST")

# Experimental reference: smooth sphere (based on Achenbach 1972, White FM)
Re_exp = np.array([1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 4e5, 5e5, 1e6])
Cd_exp = np.array([0.47, 0.47, 0.47, 0.47, 0.47, 0.45, 0.30, 0.10, 0.07, 0.07, 0.12])
ax.semilogx(Re_exp, Cd_exp, "--", color="#4ecdc4", linewidth=1.5, alpha=0.8, label="Experimental (smooth)")

# Sub-critical vs super-critical regime markers
ax.axvspan(1e4, 2e5, alpha=0.05, color="#4ecdc4", label="Sub-critical" if False else "")
ax.axvspan(2e5, 1e6, alpha=0.05, color="#f5a623", label="Super-critical" if False else "")
ax.text(3e4, 0.35, "Sub-critical\n(laminar BL)", color="#4ecdc4", fontsize=9, ha="center")
ax.text(6e5, 0.35, "Super-critical\n(turbulent BL)", color="#f5a623", fontsize=9, ha="center")

ax.set_xlabel("Reynolds Number $Re_D$", color="white")
ax.set_ylabel("Drag Coefficient $C_d$", color="white")
ax.set_title("Sphere Drag Crisis — SU2 RANS SST vs Experiment", color="white")
ax.tick_params(colors="white")
ax.legend(facecolor="#222", edgecolor="#333", labelcolor="white")
for s in ax.spines.values():
    s.set_color("#333")

plt.tight_layout()
plt.savefig(str(IMAGES / "sphere_cd_re.png"), facecolor="#111111")
print(f"  Plot saved: {IMAGES / 'sphere_cd_re.png'}")

# ── 5. Print comparison summary ──
print(f"\n{'='*60}")
print(f"{'Re':>10} | {'SU2 Cd':>8} | {'Exp Cd':>8} | {'Note':>25}")
print(f"{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*25}")
for i, Re in enumerate(Re_vals):
    idx = np.argmin(np.abs(Re_exp - Re))
    exp_cd = Cd_exp[idx]
    note = ""
    if Cd_vals[i] > exp_cd * 1.2:
        note = "over-predicts drag"
    elif Cd_vals[i] < exp_cd * 0.8:
        note = "under-predicts drag"
    else:
        note = "good match"
    print(f"{Re:>10.0e} | {Cd_vals[i]:>8.4f} | {exp_cd:>8.3f} | {note:>25}")
print(f"{'='*60}")
