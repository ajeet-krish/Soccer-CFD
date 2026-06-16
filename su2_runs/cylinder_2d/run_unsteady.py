"""Unsteady 2D cylinder: Re sweep [120, 200, 500] × no-spin vs Magnus (laminar).
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
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_cylinder_2d"

SPIN_RATE = 0.4  # S = ω·R/U∞ = 0.2
INNER_ITER = 15
DT = 0.15
N_TIME = 200
RE_VALUES = [120, 200, 500]

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


# ── Build case list: (label, spin_name, spin_rate, re, dir_suffix) ──
spin_cases = [
    ("No Spin", 0.0, "nospin"),
    ("Magnus (S=0.2)", SPIN_RATE, "magnus"),
]

all_cases = []  # (label, spin_rate, out_dir_name, re)
for spin_label, rate, spin_name in spin_cases:
    for re in RE_VALUES:
        dir_name = f"output_{spin_name}_lam_re{re}"
        all_cases.append((f"{spin_label} @ Re={re}", rate, dir_name, re))

# ── Run each case if not already completed ──
results = {}
for label, rate, dir_name, re in all_cases:
    out_dir = WORKDIR / dir_name
    hist_path = out_dir / "history_unsteady.csv"

    # Skip if already completed
    if hist_path.exists():
        print(f"\n  [SKIP] {label} — already exists in {dir_name}/")
        rows = read_su2_history(hist_path)
        t, cl, cd = per_time_step(rows, DT)
        results[dir_name] = {"t": t, "cl": cl, "cd": cd,
                             "converged": True, "cd_final": float(cd[-1]) if len(cd) else 0,
                             "cl_final": float(cl[-1]) if len(cl) else 0}
        continue

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Output:  {dir_name}/")
    print(f"{'='*60}")
    cfg_path = make_config(re, rate, DT, N_TIME, out_dir)
    mesh_local = out_dir / mesh_path.name
    if not mesh_local.exists():
        shutil.copy2(mesh_path, mesh_local)

    solver = SU2Solver(workdir=out_dir)
    result = solver.run(cfg_path, mesh_local, timeout=3600)  # 1hr timeout
    results[dir_name] = {"t": np.array([]), "cl": np.array([]), "cd": np.array([]),
                         "converged": result.converged,
                         "cd_final": result.cd, "cl_final": result.cl}

    hist_src = out_dir / "history.csv"
    if hist_src.exists():
        hist_src.rename(out_dir / "history_unsteady.csv")

    print(f"  Done. Final Cd={result.cd:.4f}  Cl={result.cl:.4f}")

    # Reload history for this case
    rows = read_su2_history(hist_path)
    t, cl, cd = per_time_step(rows, DT)
    results[dir_name] = {"t": t, "cl": cl, "cd": cd,
                         "converged": result.converged,
                         "cd_final": result.cd, "cl_final": result.cl}

# ── Cl(t) overlay plot: all 6 curves ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

line_styles = {120: "-", 200: "--", 500: ":"}
colors = {"nospin": "#2ecc71", "magnus": "#f39c12"}
spin_names_for_plot = {"nospin": "No Spin", "magnus": "Magnus"}

for spin_label, rate, spin_name in spin_cases:
    for re in RE_VALUES:
        dir_name = f"output_{spin_name}_lam_re{re}"
        d = results.get(dir_name)
        if d is None or len(d["t"]) == 0:
            continue
        ax.plot(d["t"], d["cl"], color=colors[spin_name],
                linestyle=line_styles[re], linewidth=1.5,
                label=f"{spin_names_for_plot[spin_name]} @ Re={re}")

ax.set_xlabel("Time (s)", fontsize=12, color="white")
ax.set_ylabel("Lift Coefficient Cl", fontsize=12, color="white")
ax.set_title("Unsteady 2D Cylinder — Re Sweep: No Spin vs Magnus", fontsize=13, color="white")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.tick_params(colors="white")
fig.tight_layout()
fig.savefig(IMAGES / "cylinder_unsteady_cl_comparison.png", dpi=150, facecolor="#111111")
print(f"\nSaved Cl(t) overlay: cylinder_unsteady_cl_comparison.png")
plt.close()

# ── Summary ──
summary = {}
for spin_label, rate, spin_name in spin_cases:
    for re in RE_VALUES:
        dir_name = f"output_{spin_name}_lam_re{re}"
        d = results.get(dir_name)
        if d is None:
            continue
        cl = d["cl"]
        cd = d["cd"]
        print(f"\n  {spin_label} @ Re={re}: Cd={d['cd_final']:.4f}  Cl={d['cl_final']:.4f}")
        entry = {
            "case": spin_label, "reynolds": re,
            "spin_rate": rate,
            "cd_final": d["cd_final"], "cl_final": d["cl_final"],
            "converged": d["converged"],
        }
        if len(cl) > 0:
            entry.update({"cl_min": float(cl.min()), "cl_max": float(cl.max()),
                          "cl_mean": float(cl.mean()), "cl_std": float(cl.std())})
        if len(cd) > 0:
            entry.update({"cd_min": float(cd.min()), "cd_max": float(cd.max()),
                          "cd_mean": float(cd.mean()), "cd_std": float(cd.std())})
        summary[dir_name] = entry

Path(WORKDIR / "results_unsteady.json").write_text(json.dumps(summary, indent=2))
print(f"\nResults saved to results_unsteady.json")
