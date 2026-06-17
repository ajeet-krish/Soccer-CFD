"""Phase 5: Textured sphere roughness sweep — 4 ball types × 5 Re."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.su2_runner import SU2Config, SU2Solver

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_sphere"

# Ball types: (name, year, panels, seam_depth_mm, k_s_D)
# k_s = seam depth in meters
# D = 0.22 m (FIFA regulation size 5)
BALLS = [
    {"name": "Smooth",   "year": None, "panels": None, "seam_depth_mm": 0.0,  "k_s_D": 0.0},
    {"name": "Jabulani", "year": 2010, "panels": 8,    "seam_depth_mm": 1.5,  "k_s_D": 0.007},
    {"name": "Brazuca",  "year": 2014, "panels": 6,    "seam_depth_mm": 3.0,  "k_s_D": 0.014},
    {"name": "Trionda",  "year": 2026, "panels": 4,    "seam_depth_mm": 4.5,  "k_s_D": 0.020},
]

SPHERE_DIAMETER = 0.22  # m
SPHERE_AREA = np.pi * SPHERE_DIAMETER**2 / 4  # 0.038 m²

# Re sweep — covers subcritical regime where RANS SST is most meaningful
RE_VALUES = [1e4, 1e5, 3e5, 6e5, 1e6]

TIMEOUT_PER_RUN = 1800  # 30 minutes per run


def run_roughness_sweep() -> dict:
    """Run all ball × Re combinations and return structured results."""
    mesh_path = WORKDIR / "sphere.su2"
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Sphere mesh not found at {mesh_path}. Run run_sphere.py first."
        )

    all_results = {}
    total = len(BALLS) * len(RE_VALUES)
    completed = 0

    for ball in BALLS:
        name = ball["name"]
        k_s = ball["k_s_D"] * SPHERE_DIAMETER  # dimensional roughness height (m)
        ball_key = name.lower()
        all_results[ball_key] = {"ball": ball, "runs": []}

        for Re in RE_VALUES:
            completed += 1
            print(f"\n[{completed}/{total}] {name} @ Re = {Re:.0e} (k_s = {k_s:.6f} m)")

            config = SU2Config.from_re(Re=Re, length=SPHERE_DIAMETER, incompressible=True)
            config.ref_area = SPHERE_AREA
            config.iterations = 400
            config.cfl_number = 0.5
            config.conv_residual_minval = -6
            config.screen_output = "WARNING"
            config.wall_roughness = k_s

            cfg_path = WORKDIR / f"rough_{ball_key}_Re{int(Re)}.cfg"
            config.write(cfg_path)

            solver = SU2Solver(workdir=WORKDIR)
            t0 = time.time()
            result = solver.run(cfg_path, mesh_path, timeout=TIMEOUT_PER_RUN)
            elapsed = time.time() - t0

            entry = {
                "Re": Re,
                "k_s_D": ball["k_s_D"],
                "k_s_m": k_s,
                "cd": result.cd,
                "cl": result.cl,
                "converged": result.converged,
                "iterations": result.iterations,
                "elapsed_s": round(elapsed, 1),
            }
            all_results[ball_key]["runs"].append(entry)
            print(f"  SU2: Cd={result.cd:.4f}, Cl={result.cl:.4f}, "
                  f"converged={result.converged}, {elapsed:.0f}s")

    return all_results


def plot_cd_re(all_results: dict, save_path: Path) -> None:
    """Cd vs Re overlay for all ball types + experimental smooth sphere."""
    # Experimental data (Achenbach 1972, White FM)
    Re_exp = np.array([1e4, 2e4, 5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 4e5, 5e5, 1e6])
    Cd_exp = np.array([0.47, 0.47, 0.47, 0.47, 0.47, 0.45, 0.30, 0.10, 0.07, 0.07, 0.12])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111111")
    ax.set_facecolor("#1a1a1a")

    # Experimental reference
    ax.semilogx(Re_exp, Cd_exp, "--", color="#4ecdc4", linewidth=1.5, alpha=0.7,
                label="Experimental (smooth sphere)")

    # Colormap for ball types
    colors = ["#ffffff", "#f5a623", "#e74c3c", "#9b59b6"]
    markers = ["o", "s", "^", "D"]

    for idx, (ball_key, data) in enumerate(all_results.items()):
        runs = sorted(data["runs"], key=lambda r: r["Re"])
        Re_vals = np.array([r["Re"] for r in runs])
        Cd_vals = np.array([r["cd"] for r in runs])
        ball_name = data["ball"]["name"]
        ax.semilogx(Re_vals, Cd_vals, marker=markers[idx], color=colors[idx],
                    linewidth=2, markersize=7, label=ball_name)

    # Regime markers
    ax.axvspan(1e4, 2e5, alpha=0.05, color="#4ecdc4")
    ax.axvspan(2e5, 1e6, alpha=0.05, color="#f5a623")
    ax.text(3e4, 0.90, "Sub-critical", color="#4ecdc4", fontsize=9, ha="center")
    ax.text(6e5, 0.90, "Super-critical", color="#f5a623", fontsize=9, ha="center")

    ax.set_xlabel("Reynolds Number $Re_D$", color="white")
    ax.set_ylabel("Drag Coefficient $C_d$", color="white")
    ax.set_title("Sphere Roughness Sweep — SU2 RANS SST", color="white")
    ax.set_ylim(0.0, 1.2)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222", edgecolor="#333", labelcolor="white")
    for s in ax.spines.values():
        s.set_color("#333")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), facecolor="#111111", dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_cd_vs_roughness(all_results: dict, save_path: Path) -> None:
    """Cd vs k_s/D at each Re — shows sensitivity to roughness."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111111")
    ax.set_facecolor("#1a1a1a")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(RE_VALUES)))

    for i, Re in enumerate(RE_VALUES):
        ks_vals = []
        cd_vals = []
        for ball_key, data in all_results.items():
            for run in data["runs"]:
                if abs(run["Re"] - Re) / Re < 0.01:
                    ks_vals.append(run["k_s_D"])
                    cd_vals.append(run["cd"])
                    break
        # Sort by k_s
        order = np.argsort(ks_vals)
        ax.plot(np.array(ks_vals)[order], np.array(cd_vals)[order],
                marker="o", color=colors[i], linewidth=2, markersize=7,
                label=f"Re = {Re:.0e}")

    ax.set_xlabel("Roughness $k_s / D$", color="white")
    ax.set_ylabel("Drag Coefficient $C_d$", color="white")
    ax.set_title("Cd Sensitivity to Surface Roughness", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222", edgecolor="#333", labelcolor="white")
    for s in ax.spines.values():
        s.set_color("#333")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), facecolor="#111111", dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


def print_summary(all_results: dict) -> None:
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"{'Ball':<12} {'Re':>10} | {'Cd':>8} | {'Cl':>8} | {'Converged':>10} | {'k_s/D':>8}")
    print(f"{'-'*12} {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
    for ball_key, data in all_results.items():
        name = data["ball"]["name"]
        for run in sorted(data["runs"], key=lambda r: r["Re"]):
            print(f"{name:<12} {run['Re']:>10.0e} | {run['cd']:>8.4f} | "
                  f"{run['cl']:>8.4f} | {str(run['converged']):>10} | {run['k_s_D']:>8.4f}")
    print(f"{'='*80}")


def main() -> None:
    print("=" * 60)
    print("Phase 5: Textured Sphere Roughness Sweep")
    print(f"  {len(BALLS)} ball types × {len(RE_VALUES)} Re = {len(BALLS) * len(RE_VALUES)} runs")
    print("=" * 60)

    for ball in BALLS:
        k_s = ball["k_s_D"] * SPHERE_DIAMETER
        print(f"  {ball['name']:<10} k_s/D={ball['k_s_D']:.3f}  k_s={k_s:.6f}m")

    t_start = time.time()
    all_results = run_roughness_sweep()
    elapsed_total = time.time() - t_start

    # Save JSON
    output = {
        "phase": 5,
        "description": "Textured sphere roughness sweep — 4 ball types × 5 Re",
        "mesh": "sphere.su2 (45k nodes, 282k tets, y+≈13)",
        "method": "WALL_ROUGHNESS (dimensional k_s)",
        "elapsed_total_s": round(elapsed_total, 1),
        "balls": all_results,
    }
    results_path = WORKDIR / "results_roughness.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {results_path}")

    print_summary(all_results)

    # Generate plots
    plot_cd_re(all_results, IMAGES / "sphere_roughness_cd_re.png")
    plot_cd_vs_roughness(all_results, IMAGES / "sphere_roughness_cd_vs_ks.png")

    print(f"\n  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    main()
