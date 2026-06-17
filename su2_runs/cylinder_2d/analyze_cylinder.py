"""
analyze_cylinder.py — Extract Cp(theta), separation angles, and surface pressure
from the final timestep of each fine-mesh laminar NS case.

Usage:  uv run python su2_runs/cylinder_2d/analyze_cylinder.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_cylinder_2d"

RHO = 1.2886      # kg/m³ (dimensional, from config)
U_INF = 1.0       # m/s
P_DYN = 0.5 * RHO * U_INF**2  # 0.6443 Pa

CYL_R = 0.3       # cylinder radius (m)
R_TOL = 0.001     # tolerance for wall node detection


def discover_cases():
    """Return list of (dir_name, spin_name, re) sorted."""
    cases = []
    for d in sorted(WORKDIR.glob("output_*_lam_re*_fine")):
        parts = d.name.split("_")
        spin = parts[1]
        re = int(parts[3].replace("re", ""))
        cases.append((d, spin, re))
    return cases


def load_wall_data(case_dir):
    """Load final VTU and extract wall-node pressure + coordinates + Cf."""
    vtu_files = sorted(case_dir.glob("vol_solution_*.vtu"))
    if not vtu_files:
        print(f"  [SKIP] No VTU files in {case_dir.name}")
        return None
    vtu = pv.read(str(vtu_files[-1]))
    pts = np.array(vtu.points[:, :2])
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    mask = np.abs(dist - CYL_R) < R_TOL
    if not mask.any():
        print(f"  [SKIP] No wall nodes found in {case_dir.name}")
        return None
    x_wall = pts[mask, 0]
    y_wall = pts[mask, 1]
    theta = np.arctan2(y_wall, x_wall)  # radians, [-π, π]
    p = np.array(vtu.point_data["Pressure"][mask])
    cf = np.array(vtu.point_data["Skin_Friction_Coefficient"][mask])  # (N, 3)
    return theta, p, cf, x_wall, y_wall


def compute_cp(p):
    """Pressure coefficient: Cp = (p - p_ref) / (0.5 * ρ * U²)."""
    return p / P_DYN


def sort_by_theta(theta, *arrays):
    """Sort all arrays by ascending theta."""
    idx = np.argsort(theta)
    return (theta[idx],) + tuple(a[idx] for a in arrays)


def find_separation_angle(theta, cp):
    """
    Estimate separation angle from Cp distribution.

    For laminar cylinder flow:
      - Suction peak (min Cp) occurs at θ ≈ 65°–75° (front shoulder)
      - After the peak, adverse pressure gradient (Cp rising)
      - Separation occurs where dCp/dθ drops → Cp plateaus

    We locate the suction peak in the front half (θ < 90°), then
    find where the recovery slope drops below 20% of the max recovery slope.
    """
    deg = np.degrees(theta)
    # Top side only (θ > 0)
    top = (deg > 15) & (deg < 160)
    if not top.any():
        return None
    deg_t = deg[top]
    cp_t = cp[top]
    # Suction peak in the front half only
    front = deg_t < 85
    if not front.any():
        return None
    i_min = np.argmin(cp_t[front])  # index within front subset
    # Map back to full top array
    i_min_global = np.where(front)[0][i_min]
    if i_min_global >= len(deg_t) - 5:
        return None
    # After suction peak, compute slopes over a moving window
    window = min(5, (len(deg_t) - i_min_global) // 3)
    if window < 2:
        return None
    slopes = []
    for i in range(i_min_global + 1, len(deg_t) - window):
        dcp = cp_t[i + window] - cp_t[i - window]
        dt = deg_t[i + window] - deg_t[i - window]
        if dt > 0:
            slopes.append(abs(dcp / dt))
        else:
            slopes.append(0)
    if not slopes:
        return None
    # Find where slope drops below 20% of max in the recovery region
    max_slope = max(slopes)
    threshold = 0.20 * max_slope
    for i, s in enumerate(slopes):
        if s < threshold:
            sep_idx = i_min_global + 1 + i
            sep_deg = float(deg_t[sep_idx])
            if 60 < sep_deg < 130:
                return sep_deg
    return None


def plot_cp_comparison(all_data):
    """2-panel figure: full Cp(θ) and zoomed rear separation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})
    colors = {"nospin": {"120": "#4fc3f7", "200": "#29b6f6", "500": "#0288d1"},
              "magnus": {"120": "#ef5350", "200": "#e53935", "500": "#b71c1c"}}
    styles = {"nospin": "-", "magnus": "--"}

    for (theta, cp, spin, re, sep_angle) in all_data:
        color = colors[spin][str(re)]
        style = styles[spin]
        label = f"{'No Spin' if spin == 'nospin' else 'Magnus'} Re={re}"
        deg = np.degrees(theta)
        ax1.plot(deg, cp, style, color=color, label=label, linewidth=1.0, alpha=0.8)
        ax2.plot(deg, cp, style, color=color, label=label, linewidth=1.0, alpha=0.8)
        if sep_angle is not None:
            ax1.axvline(sep_angle, color=color, linestyle=":", alpha=0.5)
            ax2.axvline(sep_angle, color=color, linestyle=":", alpha=0.5)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="#44475a", linewidth=0.5)
        ax.axvline(0, color="#44475a", linewidth=0.5)
        ax.set_xlabel("Angle θ (degrees)", color="white")
        ax.set_ylabel("$C_p$", color="white")
        ax.tick_params(colors="white")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    ax1.set_title("$C_p(\\theta)$ — All Cases", color="white", fontsize=12)
    ax2.set_title("Rear Surface (Separation Detail)", color="white", fontsize=12)
    ax2.set_xlim(60, 180)
    ax2.set_ylim(-0.5, 0.5)

    fig.patch.set_facecolor("#282a36")
    fig.tight_layout()
    out = IMAGES / "analysis" / "cylinder_cp_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#282a36")
    print(f"  Saved: {out}")
    plt.close()


def main():
    print("=" * 60)
    print("  CYLINDER CP ANALYSIS — LAMINAR RE SWEEP")
    print("=" * 60)

    cases = discover_cases()
    if not cases:
        print("  No cases found.")
        sys.exit(1)

    all_data = []
    for case_dir, spin, re in cases:
        print(f"\n  Processing: {case_dir.name}")
        result = load_wall_data(case_dir)
        if result is None:
            continue
        theta, p, cf, x_wall, y_wall = result
        cp = compute_cp(p)
        theta, cp, cf, x_wall, y_wall = sort_by_theta(theta, cp, cf, x_wall, y_wall)

        sep_angle = find_separation_angle(theta, cp)
        if sep_angle is not None:
            print(f"    Wall nodes: {len(theta)}")
            print(f"    Cp range: {cp.min():.3f} to {cp.max():.3f}")
            print(f"    Separation angle: θ ≈ {sep_angle:.1f}°")
        else:
            print(f"    Separation angle not found")

        all_data.append((theta, cp, spin, re, sep_angle))

    if all_data:
        plot_cp_comparison(all_data)

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
