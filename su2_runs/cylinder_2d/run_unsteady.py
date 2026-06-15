"""Unsteady 2D cylinder: no-spin vs Magnus at two Reynolds numbers.
- Phase 1: Re=40k URANS SST (turbulent, shedding suppressed)
- Phase 2: Re=100  Laminar NS (clean von Kármán street)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, SU2Solver
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
CFL = 0.5

from src.su2_runner import MeshGenerator

# Generate finer mesh with wake refinement for vortex shedding
print("=== Generating refined mesh with wake refinement ===")
mesh_path = MeshGenerator.cylinder_2d(
    radius=0.3,
    farfield_radius=20.0,
    cl_cylinder=0.008,       # ~235 elems around cylinder
    cl_farfield=1.0,
    cl_wake=0.04,            # fine wake region (40x finer than farfield)
    wake_length=15.0,        # 15D downstream
    wake_width=2.0,          # 2D half-width
    name=str(WORKDIR / "cylinder_fine"),
)
print(f"  Mesh: {mesh_path}")


def make_config(reynolds: int, rotation_rate: float, dt: float,
                n_time: int, suffix: str, solver: str,
                laminar: bool = False) -> Path:
    if laminar:
        # Use compressible NS at low Mach (M=0.1) for ROE scheme
        config = SU2Config()
        config.solver = "NAVIER_STOKES"
        config.turbulence_model = "NONE"
        config.mach_number = 0.05  # very low Mach for incompressible-like flow
        config.reynolds_number = reynolds
        config.reynolds_length = 0.6
        config.conv_numerical_method_flow = "ROE"
        config.muscl_flow = "YES"
        config.cfl_number = 5.0
    else:
        config = SU2Config.from_re(Re=reynolds, length=0.6, incompressible=True)
        config.solver = solver
        config.cfl_number = CFL

    config.conv_residual_minval = -8
    config.screen_output = "WARNING"
    config.rotation_rate = rotation_rate
    config.time_domain = True
    config.time_marching = "DUAL_TIME_STEPPING-2ND_ORDER"
    config.time_step = dt
    config.max_time = n_time * dt
    config.time_iter = n_time
    config.inner_iter = INNER_ITER
    config.output_wrt_freq = 1

    cfg_path = WORKDIR / f"cylinder_unsteady_{suffix}.cfg"
    config.write(cfg_path)

    # Override numerics for laminar low-Mach compressible
    if laminar:
        txt = cfg_path.read_text()
        txt = txt.replace("SLOPE_LIMITER_FLOW= VENKATAKRISHNAN", "SLOPE_LIMITER_FLOW= NONE")
        txt += "\n% Low Mach preconditioner for ROE\n"
        txt += "LOW_MACH_PRECONDITIONER= YES\n"
        cfg_path.write_text(txt)
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


# ── Phase 1: Re=40k URANS SST ──
print("=" * 60)
print("  PHASE 1: Re=40k URANS SST — No Spin vs Magnus")
print("  (turbulence model suppresses shedding)")
print("=" * 60)

DT1 = 0.12
N1 = 200
results1 = {}  # {label: SU2Results}

for label, rate, suffix, solver in [
    ("No Spin (Re=40k SST)", 0.0, "nospin_sst", "INC_RANS"),
    ("Magnus (Re=40k SST)", SPIN_RATE, "magnus_sst", "INC_RANS"),
]:
    print(f"  Running: {label} ...")
    cfg_path = make_config(40000, rate, DT1, N1, suffix, solver)
    solver_obj = SU2Solver(workdir=WORKDIR)
    results1[label] = solver_obj.run(cfg_path, mesh_path, timeout=1800)
    hist_src = WORKDIR / "history.csv"
    if hist_src.exists():
        hist_src.rename(WORKDIR / f"history_unsteady_{suffix}.csv")

# ── Phase 2: Re=100 Laminar NS (no-spin + magnus) ──
print("\n" + "=" * 60)
print("  PHASE 2: Re=100 Laminar NS — No Spin vs Magnus")
print("  (clean von Kármán vortex street)")
print("=" * 60)

# St ≈ 0.17 at Re=100: T = D/(U·St) = 0.6/0.17 ≈ 3.53s
DT2 = 0.15  # ~T/24
N2 = 200    # ~8.5 periods
results2 = {}

for label, rate, suffix, solver in [
    ("No Spin (Re=100 Laminar)", 0.0, "nospin_lam", "INC_NAVIER_STOKES"),
    ("Magnus (Re=100 Laminar)", SPIN_RATE, "magnus_lam", "INC_NAVIER_STOKES"),
]:
    print(f"  Running: {label} ...")
    cfg_path = make_config(100, rate, DT2, N2, suffix, solver, laminar=True)
    solver_obj = SU2Solver(workdir=WORKDIR)
    results2[label] = solver_obj.run(cfg_path, mesh_path, timeout=1800)
    hist_src = WORKDIR / "history.csv"
    if hist_src.exists():
        hist_src.rename(WORKDIR / f"history_unsteady_{suffix}.csv")

# ── Extract Cl(t) ──
all_cases = [
    ("No Spin (Re=40k SST)", "nospin_sst", DT1, "#e74c3c"),
    ("Magnus (Re=40k SST)", "magnus_sst", DT1, "#3498db"),
    ("No Spin (Re=100 Laminar)", "nospin_lam", DT2, "#2ecc71"),
    ("Magnus (Re=100 Laminar)", "magnus_lam", DT2, "#f39c12"),
]

time_data = {}
cl_data = {}
cd_data = {}
results_all = {**results1, **results2}

for label, suffix, dt, _ in all_cases:
    hist = WORKDIR / f"history_unsteady_{suffix}.csv"
    rows = read_su2_history(hist)
    t, cl, cd = per_time_step(rows, dt)
    time_data[label] = t
    cl_data[label] = cl
    cd_data[label] = cd
    if len(t) > 0:
        print(f"  {label}: {len(t)} steps, Cl ∈ [{cl.min():.3f}, {cl.max():.3f}]")

# ── Plot: two subplots ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Re=40k SST
for label in ["No Spin (Re=40k SST)", "Magnus (Re=40k SST)"]:
    t = time_data[label]
    cl = cl_data[label]
    if len(t) == 0:
        continue
    c = "#e74c3c" if "No Spin" in label else "#3498db"
    ax1.plot(t, cl, color=c, label=label, linewidth=1.5)
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Cl", fontsize=11)
ax1.set_title("Re=40k — URANS SST", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

# Right: Re=100 Laminar
for label in ["No Spin (Re=100 Laminar)", "Magnus (Re=100 Laminar)"]:
    t = time_data[label]
    cl = cl_data[label]
    if len(t) == 0:
        continue
    c = "#2ecc71" if "No Spin" in label else "#f39c12"
    ax2.plot(t, cl, color=c, label=label, linewidth=1.5)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.set_ylabel("Cl", fontsize=11)
ax2.set_title("Re=100 — Laminar NS", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

fig.suptitle("Unsteady 2D Cylinder: No Spin vs Magnus Effect", fontsize=14, y=1.02)
fig.tight_layout()
plot_path = IMAGES / "cylinder_unsteady_cl_comparison.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {plot_path}")
plt.close()

# ── Summary ──
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for label, _, dt, _ in all_cases:
    r = results_all.get(label)
    if not r:
        continue
    t = time_data[label]
    cl = cl_data[label]
    print(f"\n  {label}:")
    print(f"    Cd={r.cd:.4f}  Cl={r.cl:.4f}  converged={r.converged}")
    if len(cl) > 0:
        print(f"    Cl: min={cl.min():.4f}  max={cl.max():.4f}  mean={cl.mean():.4f}  std={cl.std():.4f}")
    if len(t) > 4 and "Laminar" in label:
        n = len(cl)
        cl_detrended = cl - np.polyval(np.polyfit(t, cl, 1), t)
        fft_vals = np.fft.rfft(cl_detrended)
        fft_freqs = np.fft.rfftfreq(n, d=dt)
        peak_idx = np.argmax(np.abs(fft_vals[1:])) + 1
        peak_freq = fft_freqs[peak_idx]
        st = peak_freq * 0.6 / 1.0
        print(f"    Shedding freq: {peak_freq:.3f} Hz  St: {st:.3f}")

# Save results
summary = {}
for label, suffix, _, _ in all_cases:
    r = results_all.get(label)
    if not r:
        continue
    t = time_data[label]
    cl = cl_data[label]
    entry = {
        "case": label,
        "reynolds": 40000 if "sst" in suffix else 100,
        "spin_rate": SPIN_RATE if "magnus" in suffix else 0.0,
        "cd_final": r.cd,
        "cl_final": r.cl,
        "converged": r.converged,
    }
    if len(cl) > 0:
        entry.update({"cl_min": float(cl.min()), "cl_max": float(cl.max()),
                      "cl_mean": float(cl.mean()), "cl_std": float(cl.std())})
    summary[suffix] = entry

Path(WORKDIR / "results_unsteady.json").write_text(json.dumps(summary, indent=2))
print(f"\nResults saved to results_unsteady.json")
