"""URANS γ-Reθ at Re=100k on structured O-grid (y+≈1).
Single case to test if transition model captures laminar separation bubble.
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, SU2Solver, MeshGenerator
from pathlib import Path
import csv
import numpy as np

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_cylinder_2d"

RE = 100_000
DT = 0.3
N_TIME = 300
INNER_ITER = 15

# ── Mesh ──
mesh_path = WORKDIR / "cylinder_structured.su2"
if not mesh_path.exists():
    print("=== Generating structured O-grid mesh ===")
    mesh_path = MeshGenerator.cylinder_2d_structured(
        radius=0.3, farfield_radius=20.0,
        n_radial=105, n_circum=720, growth=1.1,
        name=str(WORKDIR / "cylinder_structured"),
    )
    print(f"  Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")

OUT_DIR = WORKDIR / "output_transition_urans_re100k"
HIST_PATH = OUT_DIR / "history_urans.csv"

if HIST_PATH.exists():
    print(f"  [SKIP] Already done — {HIST_PATH}")
else:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_text = f"""SOLVER= INC_RANS
KIND_TURB_MODEL= SST
KIND_TRANS_MODEL= LM
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
SYSTEM_MEASUREMENTS= SI
INC_DENSITY_MODEL= CONSTANT
INC_ENERGY_EQUATION= NO
INC_DENSITY_INIT= 1.2886
INC_VELOCITY_INIT= ( 1.0, 0.0, 0.0 )
INC_NONDIM= INITIAL_VALUES
REYNOLDS_NUMBER= {RE}
REYNOLDS_LENGTH= 0.6
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {1.0/RE:.6e}
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_MONITORING= ( wall )
MARKER_FAR= ( farfield )
CONV_NUM_METHOD_FLOW= FDS
MUSCL_FLOW= NO
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
CFL_NUMBER= 1.0
TIME_DOMAIN= YES
TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER
TIME_STEP= {DT}
MAX_TIME= {N_TIME * DT}
TIME_ITER= {N_TIME}
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
MESH_FILENAME= {mesh_path.name}
"""
    cfg_path = OUT_DIR / "cylinder_urans_transition.cfg"
    cfg_path.write_text(cfg_text)

    mesh_local = OUT_DIR / mesh_path.name
    if not mesh_local.exists():
        shutil.copy2(mesh_path, mesh_local)

    print(f"\n{'='*60}")
    print(f"  URANS γ-Reθ @ Re={RE:,} — structured O-grid (y+≈1)")
    print(f"  {N_TIME} steps × {INNER_ITER} inner iters")
    print(f"{'='*60}")

    solver = SU2Solver(workdir=OUT_DIR)
    result = solver.run(cfg_path, mesh_local, timeout=14400)

    hist_src = OUT_DIR / "history.csv"
    if hist_src.exists():
        hist_src.rename(HIST_PATH)

    print(f"  Done. Converged={result.converged}")

# ── Parse time history ──
print("\n=== Post-processing ===")
with HIST_PATH.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Column names may have whitespace/quotes
def clean(k):
    return k.strip().strip('"').strip()
fields = [clean(h) for h in reader.fieldnames] if reader.fieldnames else []
reader2 = csv.DictReader(HIST_PATH.open(newline=""))
reader2.fieldnames = [clean(h) for h in reader2.fieldnames]

# Best value per time step (last inner iter)
best = {}
for row in reader2:
    ti = int(float(row.get("Time_Iter", 0)))
    ii = int(float(row.get("Inner_Iter", 0)))
    cd = float(row.get("DRAG", row.get("CD", 0)))
    cl = float(row.get("LIFT", row.get("CL", 0)))
    if ti not in best or ii > best[ti][0]:
        best[ti] = (ii, cd, cl)

sorted_ti = sorted(best.keys())
t = np.array([ti * DT for ti in sorted_ti])
cd = np.array([best[ti][1] for ti in sorted_ti])
cl = np.array([best[ti][2] for ti in sorted_ti])

# Skip transient (first 50 steps)
N_TRANSIENT = 50
steady = sorted_ti[N_TRANSIENT:]
t_s = t[N_TRANSIENT:]
cd_s = cd[N_TRANSIENT:]
cl_s = cl[N_TRANSIENT:]

print(f"  Time steps: {len(sorted_ti)} (transient skipped: {N_TRANSIENT})")
print(f"  Cd: mean={cd_s.mean():.4f} ± {(cd_s.max()-cd_s.min())/2:.4f}")
print(f"  Cl: mean={cl_s.mean():.4f}  range=[{cl_s.min():.3f}, {cl_s.max():.3f}]")
print(f"  Final Cd(t={t[-1]:.1f}s): {cd[-1]:.4f}")

# Strouhal from Cl spectrum
from scipy import signal as sg
if len(cl_s) > 50:
    freq, psd = sg.welch(cl_s - cl_s.mean(), fs=1/DT, nperseg=min(128, len(cl_s)//2))
    peak_idx = np.argmax(psd)
    st = freq[peak_idx] * 0.6 / 1.0  # St = f*D/U
    print(f"  Strouhal (from Cl spectrum): {st:.3f}")
else:
    st = 0

# Save results
results = {
    "reynolds": RE,
    "cd_mean": float(cd_s.mean()),
    "cd_std": float(cd_s.std()),
    "cd_min": float(cd_s.min()),
    "cd_max": float(cd_s.max()),
    "cd_final": float(cd[-1]),
    "cl_mean": float(cl_s.mean()),
    "cl_std": float(cl_s.std()),
    "cl_min": float(cl_s.min()),
    "cl_max": float(cl_s.max()),
    "strouhal": st,
    "n_time_steps": len(sorted_ti),
    "transient_skipped": N_TRANSIENT,
}
Path(WORKDIR / "results_transition_urans.json").write_text(json.dumps(results, indent=2))
print(f"\nResults saved to results_transition_urans.json")
