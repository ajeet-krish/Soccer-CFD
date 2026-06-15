"""Unsteady 2D cylinder: no-spin vs Magnus at Re=120 (laminar).
Each case writes to its own output directory for clean VTU preservation.
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, SU2Solver, MeshGenerator
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images"

SPIN_RATE = 0.4  # S = ω·R/U∞ = 0.2
INNER_ITER = 15
DT = 0.15
N_TIME = 200

mesh_path = WORKDIR / "cylinder_fine.su2"
if not mesh_path.exists():
    print("=== Generating refined mesh with wake refinement ===")
    mesh_path = MeshGenerator.cylinder_2d(
        radius=0.3, farfield_radius=20.0,
        cl_cylinder=0.008, cl_farfield=1.0,
        cl_wake=0.04, wake_length=15.0, wake_width=2.0,
        name=str(WORKDIR / "cylinder_fine"),
    )
    print(f"  Mesh: {mesh_path}")


def make_config(reynolds: int, rotation_rate: float,
                dt: float, n_time: int, out_dir: Path) -> Path:
    U = 1.0
    rho = 1.2886
    D = 0.6
    mu = rho * U * D / reynolds
    cfg_text = f"""SOLVER= INC_NAVIER_STOKES
KIND_TURB_MODEL= NONE
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
INC_DENSITY_MODEL= CONSTANT
INC_ENERGY_EQUATION= NO
INC_DENSITY_INIT= {rho}
INC_VELOCITY_INIT= ( {U}, 0.0, 0.0 )
INC_NONDIM= DIMENSIONAL
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {mu}
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_MONITORING= ( wall )
MARKER_FAR= ( farfield )
{f'SURFACE_MOVEMENT= MOVING_WALL' if abs(rotation_rate) > 1e-10 else '% No moving wall'}
{f'MARKER_MOVING= ( wall )' if abs(rotation_rate) > 1e-10 else ''}
{f'SURFACE_MOTION_ORIGIN= 0.0 0.0 0.0' if abs(rotation_rate) > 1e-10 else ''}
{f'SURFACE_ROTATION_RATE= 0.0 0.0 {rotation_rate}' if abs(rotation_rate) > 1e-10 else ''}
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= NONE
TIME_DISCRE_FLOW= EULER_IMPLICIT
CFL_NUMBER= 1.0
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= {dt}
MAX_TIME= {n_time * dt}
TIME_ITER= {n_time}
INNER_ITER= {INNER_ITER}
OUTPUT_WRT_FREQ= 1
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
SCREEN_OUTPUT= (WARNING)
HISTORY_OUTPUT= ( TIME_ITER, INNER_ITER, RMS_RES, AERO_COEFF, CUR_TIME)
TABULAR_FORMAT= CSV
OUTPUT_FILES= (RESTART, PARAVIEW)
MGLEVEL= 0
"""
    cfg_path = out_dir / "cylinder.cfg"
    cfg_path.write_text(cfg_text)
    return cfg_path


def read_su2_history(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        for raw_row in reader:
            entry = {}
            for raw_key, raw_val in raw_row.items():
                key = raw_key.strip().strip('"')
                val = raw_val.strip()
                try:
                    entry[key] = float(val)
                except ValueError:
                    entry[key] = val
            rows.append(entry)
    return rows


def per_time_step(rows: list[dict], dt: float) -> tuple:
    best = {}
    for row in rows:
        ti = int(row.get("Time_Iter", 0))
        inner = int(row.get("Inner_Iter", 0))
        cl = row.get("CL", 0.0)
        cd = row.get("CD", 0.0)
        if ti not in best or inner > best[ti][0]:
            best[ti] = (inner, cl, cd)
    if not best:
        return np.array([]), np.array([]), np.array([])
    sorted_ti = sorted(best.keys())
    t = np.array([ti * dt for ti in sorted_ti])
    cl = np.array([best[ti][1] for ti in sorted_ti])
    cd = np.array([best[ti][2] for ti in sorted_ti])
    return t, cl, cd


# ── Run both laminar cases ──
cases = [
    ("No Spin", 0.0, "nospin_lam"),
    ("Magnus (S=0.2)", SPIN_RATE, "magnus_lam"),
]

results = {}
for label, rate, suffix in cases:
    out_dir = WORKDIR / f"output_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
    cfg_path = make_config(120, rate, DT, N_TIME, out_dir)
    # Copy mesh to output dir (SU2 needs it local alongside config)
    mesh_local = out_dir / mesh_path.name
    if not mesh_local.exists():
        shutil.copy2(mesh_path, mesh_local)

    solver = SU2Solver(workdir=out_dir)
    result = solver.run(cfg_path, mesh_local, timeout=1800)
    results[label] = result

    # Rename history to preserve it
    hist_src = out_dir / "history.csv"
    if hist_src.exists():
        hist_src.rename(out_dir / "history_unsteady.csv")

    print(f"  Done. Final Cd={result.cd:.4f}  Cl={result.cl:.4f}")

# ── Extract Cl(t) and plot ──
time_data = {}
cl_data = {}
cd_data = {}
for label, _, suffix in cases:
    hist = WORKDIR / f"output_{suffix}" / "history_unsteady.csv"
    rows = read_su2_history(hist)
    t, cl, cd = per_time_step(rows, DT)
    time_data[label] = t
    cl_data[label] = cl
    cd_data[label] = cd
    if len(t) > 0:
        print(f"  {label}: {len(t)} steps, Cl ∈ [{cl.min():.3f}, {cl.max():.3f}]")

# Cl(t) plot
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"No Spin": "#2ecc71", "Magnus (S=0.2)": "#f39c12"}
for label, _, _ in cases:
    t = time_data[label]
    cl = cl_data[label]
    if len(t) == 0:
        continue
    ax.plot(t, cl, color=colors[label], label=label, linewidth=1.5)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Lift Coefficient Cl", fontsize=12)
ax.set_title("Unsteady 2D Cylinder @ Re=120 — No Spin vs Magnus (S=0.2)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
fig.tight_layout()
fig.savefig(IMAGES / "cylinder_unsteady_cl_comparison.png", dpi=150)
print(f"Saved Cl(t) plot")
plt.close()

# Summary & save results
summary = {}
for label, _, suffix in cases:
    r = results[label]
    t = time_data[label]
    cl = cl_data[label]
    cd = cd_data[label]
    print(f"\n  {label}: Cd={r.cd:.4f}  Cl={r.cl:.4f}")
    if len(cl) > 0:
        print(f"    Cl: min={cl.min():.4f}  max={cl.max():.4f}  mean={cl.mean():.4f}")
    entry = {
        "case": label, "reynolds": 120,
        "spin_rate": SPIN_RATE if "magnus" in suffix else 0.0,
        "cd_final": r.cd, "cl_final": r.cl,
        "converged": r.converged,
    }
    if len(cl) > 0:
        entry.update({"cl_min": float(cl.min()), "cl_max": float(cl.max()),
                      "cl_mean": float(cl.mean()), "cl_std": float(cl.std())})
    if len(cd) > 0:
        entry.update({"cd_min": float(cd.min()), "cd_max": float(cd.max()),
                      "cd_mean": float(cd.mean()), "cd_std": float(cd.std())})
    summary[suffix] = entry

Path(WORKDIR / "results_unsteady.json").write_text(json.dumps(summary, indent=2))
print(f"\nResults saved")
