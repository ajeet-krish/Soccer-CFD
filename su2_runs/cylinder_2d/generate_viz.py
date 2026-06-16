"""Generate visualizations from unsteady laminar Re sweep (120, 200, 500).
Produces: mesh PNG, static 2D field plots PNG, full-length animated MP4,
multi-Re comparison MP4s, 3D interactive HTML. All plots match ΦFlow style.
"""
import sys, os, re as re_mod
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.contour import QuadContourSet
import pyvista as pv

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_cylinder_2d"
DT = 0.15
CYL_R = 0.3
FPS = 10  # match ΦFlow → 200 frames = 20s video = full 30s sim at 1.5x

# ΦFlow-matched viewport (tight zoom around cylinder, ~6D × 3D)
XLIM = (-1.0, 5.0)
YLIM = (-1.5, 1.5)

# ── Discover output directories ──
def discover_cases():
    """Return list of (case_dir, spin_name, re) for all output_*_lam_re* dirs."""
    cases = []
    pattern = re_mod.compile(r"output_(nospin|magnus)_lam_re(\d+)")
    for d in sorted(WORKDIR.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            cases.append((d.name, m.group(1), int(m.group(2))))
    return cases


# ── VTU helpers ──

def load_vtu(step: int, case_dir: str) -> pv.DataSet:
    path = WORKDIR / case_dir / f"vol_solution_{step:05d}.vtu"
    return pv.read(str(path))


def get_2d_data(vtu: pv.DataSet) -> tuple:
    pts = vtu.points
    x, y = pts[:, 0], pts[:, 1]
    p = vtu["Pressure"]
    vel = vtu["Velocity"]
    u, v = vel[:, 0], vel[:, 1]
    vel_mag = np.sqrt(u**2 + v**2)
    return x, y, p, u, v, vel_mag


def make_regular_grid(x, y, nx=200, ny=100):
    margin = 0.5
    xs = np.linspace(x.min() + margin, x.max() - margin, nx)
    ys = np.linspace(y.min() + margin, y.max() - margin, ny)
    return np.meshgrid(xs, ys)


def interp_to_grid(x, y, values, Xg, Yg):
    from scipy.interpolate import griddata
    points = np.column_stack([x, y])
    return griddata(points, values, (Xg, Yg), method="linear")


def _mask_cylinder(xg, yg):
    r = np.sqrt(xg**2 + yg**2)
    return r < CYL_R * 1.05


def get_step_range(case_dir: str, start=0, end=200):
    steps = []
    for s in range(start, end):
        path = WORKDIR / case_dir / f"vol_solution_{s:05d}.vtu"
        if path.exists():
            steps.append(s)
    return steps


def _setup_ax(ax, title=""):
    """Apply ΦFlow-matched dark styling."""
    fig = ax.figure
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, color="white")
    return fig, ax


# ═══════════════════════════════════════════════════════════
# 1. MESH VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_mesh():
    print("--- Mesh visualization ---")
    # Use first available case
    cases = discover_cases()
    if not cases:
        print("  [SKIP] No cases found")
        return
    case_dir = cases[0][0]
    vtu = load_vtu(0, case_dir)

    for zoom_name, zoom_factor, filename in [
        ("Full domain", 0.6, "cylinder_mesh.png"),
        ("Zoomed", 6.0, "cylinder_mesh_zoom.png"),
    ]:
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
        plotter.add_mesh(vtu, show_edges=True, color="lightblue", opacity=0.6,
                         lighting=False, label="Mesh")
        edges = vtu.separate_cells().extract_feature_edges()
        plotter.add_mesh(edges, color="grey", line_width=0.3)
        plotter.view_xy()
        plotter.camera.zoom(zoom_factor)
        plotter.screenshot(str(IMAGES / filename))
        print(f"  Saved: {filename}")
        plotter.close()


# ═══════════════════════════════════════════════════════════
# 2. STATIC 2D FIELD PLOTS (last timestep of each case)
# ═══════════════════════════════════════════════════════════

def static_plots():
    print("\n--- Static 2D field plots ---")
    cases = discover_cases()
    if not cases:
        print("  [SKIP] No cases found")
        return

    # Pressure: individual per (spin, re)
    for case_dir, spin_name, re in cases:
        steps = get_step_range(case_dir)
        if not steps:
            continue
        step = min(150, max(steps))
        vtu = load_vtu(step, case_dir)
        x, y, p, u, v, vm = get_2d_data(vtu)
        Xg, Yg = make_regular_grid(x, y)
        Pg = interp_to_grid(x, y, p, Xg, Yg)

        fig, ax = plt.subplots(figsize=(8, 4))
        _setup_ax(ax, title=f"{'No Spin' if spin_name == 'nospin' else 'Magnus'} — Pressure @ Re={re} (t={step*DT:.1f}s)")
        pg_masked = np.ma.masked_where(_mask_cylinder(Xg, Yg), Pg)
        cntr = ax.contourf(Xg, Yg, pg_masked, levels=40, cmap="magma", extend="both")
        ax.contour(Xg, Yg, pg_masked, levels=10, colors="k", linewidths=0.3, alpha=0.3)
        circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
        ax.add_patch(circle)
        cbar = fig.colorbar(cntr, ax=ax, label="Pressure (Pa)", shrink=0.85)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        cbar.outline.set_edgecolor("white")
        fig.tight_layout()
        out = IMAGES / f"cylinder_{spin_name}_pressure_re{re}.png"
        fig.savefig(out, dpi=150, facecolor="#111111")
        print(f"  Saved: {out.name}")
        plt.close()

    # Velocity: individual per (spin, re) with streamlines
    for case_dir, spin_name, re in cases:
        steps = get_step_range(case_dir)
        if not steps:
            continue
        step = min(150, max(steps))
        vtu = load_vtu(step, case_dir)
        x, y, p, u, v, vm = get_2d_data(vtu)
        Xg, Yg = make_regular_grid(x, y)
        Ug = interp_to_grid(x, y, u, Xg, Yg)
        Vg = interp_to_grid(x, y, v, Xg, Yg)

        fig, ax = plt.subplots(figsize=(8, 4))
        label = "No Spin" if spin_name == "nospin" else "Magnus"
        _setup_ax(ax, title=f"{label} — Velocity @ Re={re} (t={step*DT:.1f}s)")
        vg_masked = np.ma.masked_where(_mask_cylinder(Xg, Yg), np.sqrt(Ug**2 + Vg**2))
        cntr = ax.contourf(Xg, Yg, vg_masked, levels=40, cmap="inferno", extend="both")
        stride = 3
        streamline_color = "white" if spin_name == "nospin" else "black"
        ax.streamplot(Xg[::stride, ::stride], Yg[::stride, ::stride],
                      Ug[::stride, ::stride], Vg[::stride, ::stride],
                      color=streamline_color, linewidth=0.6, density=0.8, arrowsize=0.6)
        circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
        ax.add_patch(circle)
        cbar = fig.colorbar(cntr, ax=ax, label="Velocity (m/s)", shrink=0.85)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        cbar.outline.set_edgecolor("white")
        fig.tight_layout()
        out = IMAGES / f"cylinder_{spin_name}_velocity_re{re}.png"
        fig.savefig(out, dpi=150, facecolor="#111111")
        print(f"  Saved: {out.name}")
        plt.close()


# ═══════════════════════════════════════════════════════════
# 3. ANIMATED MP4 — Full-length (200 frames, 20s @ 10fps)
# ═══════════════════════════════════════════════════════════

def animate_flow(case_dir: str, spin_name: str, re: int,
                 field: str = "pressure"):
    """Full-length animation of pressure or velocity field."""
    steps = get_step_range(case_dir)
    if len(steps) < 2:
        print(f"  [SKIP] {case_dir} ({field}) — not enough steps")
        return
    N = len(steps)
    label = "No Spin" if spin_name == "nospin" else "Magnus"
    print(f"\n  Animating {label} Re={re} ({field}) — {N} frames @ {FPS}fps = {N/FPS:.0f}s")

    vtu0 = load_vtu(steps[0], case_dir)
    x0, y0, p0, u0, v0, vm0 = get_2d_data(vtu0)
    Xg, Yg = make_regular_grid(x0, y0)
    cyl_mask = _mask_cylinder(Xg, Yg)

    cmap = "magma" if field == "pressure" else "inferno"
    cbar_label = "Pressure (Pa)" if field == "pressure" else "Velocity (m/s)"

    fig, ax = plt.subplots(figsize=(8, 4))
    _setup_ax(ax, title=f"{label} — {field.title()} Field @ Re={re}")

    # First frame data
    vtu = load_vtu(steps[0], case_dir)
    x, y, p, u, v, vm = get_2d_data(vtu)
    if field == "pressure":
        vals = interp_to_grid(x, y, p, Xg, Yg)
        do_streamlines = False
    else:
        vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
        Ug = interp_to_grid(x, y, u, Xg, Yg)
        Vg = interp_to_grid(x, y, v, Xg, Yg)
        do_streamlines = True
        sl_color = "white" if spin_name == "nospin" else "black"

    vals_masked = np.ma.masked_where(cyl_mask, vals)
    cntr = ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")
    circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
    ax.add_patch(circle)

    if do_streamlines:
        stride = 3
        sl = ax.streamplot(Xg[::stride, ::stride], Yg[::stride, ::stride],
                           Ug[::stride, ::stride], Vg[::stride, ::stride],
                           color=sl_color, linewidth=0.6, density=0.8, arrowsize=0.6)

    cbar = fig.colorbar(cntr, ax=ax, label=cbar_label, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    cbar.outline.set_edgecolor("white")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                        color="white", bbox=dict(boxstyle="round", fc="0.2", ec="none"))
    fig.tight_layout()

    suffix = f"{spin_name}_{field}_re{re}"
    out_path = IMAGES / f"cylinder_{suffix}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=3000)

    with writer.saving(fig, str(out_path), dpi=150):
        for i, step in enumerate(steps):
            vtu = load_vtu(step, case_dir)
            x, y, p, u, v, vm = get_2d_data(vtu)
            if field == "pressure":
                vals = interp_to_grid(x, y, p, Xg, Yg)
            else:
                vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
                Ug = interp_to_grid(x, y, u, Xg, Yg)
                Vg = interp_to_grid(x, y, v, Xg, Yg)
            vals_masked = np.ma.masked_where(cyl_mask, vals)

            for c in ax.collections[:]:
                if isinstance(c, QuadContourSet):
                    c.remove()

            cntr = ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")

            # Re-draw streamlines each frame (remove old line collections)
            if do_streamlines:
                for coll in ax.lines[:]:
                    coll.remove()
                stride = 3
                sl = ax.streamplot(Xg[::stride, ::stride], Yg[::stride, ::stride],
                                   Ug[::stride, ::stride], Vg[::stride, ::stride],
                                   color=sl_color, linewidth=0.6, density=0.8, arrowsize=0.6)

            t = step * DT
            time_text.set_text(f"t = {t:.1f} s")
            writer.grab_frame()

    plt.close()
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════
# 4. RE COMPARISON ANIMATION — 3-panel (Re=120 | 200 | 500)
# ═══════════════════════════════════════════════════════════

def animate_re_comparison(spin_name: str, field: str = "pressure"):
    """3-panel side-by-side animation comparing Re=120, 200, 500."""
    label = "No Spin" if spin_name == "nospin" else "Magnus"
    cases = []
    for re in [120, 200, 500]:
        dir_name = f"output_{spin_name}_lam_re{re}"
        steps = get_step_range(dir_name)
        if len(steps) > 1:
            cases.append((dir_name, re, steps))

    if len(cases) < 2:
        print(f"  [SKIP] Re comparison {label} ({field}) — < 2 cases available")
        return

    # Use common steps across all cases
    common_steps = sorted(set.intersection(*[set(s) for _, _, s in cases]))
    if len(common_steps) < 2:
        print(f"  [SKIP] Re comparison {label} ({field}) — no common steps")
        return
    N = len(common_steps)
    print(f"\n  Re comparison: {label} ({field}) — {len(cases)} Re values, {N} frames @ {FPS}fps")

    # Pre-compute grid from first available case
    vtu0 = load_vtu(common_steps[0], cases[0][0])
    x0, y0, p0, u0, v0, vm0 = get_2d_data(vtu0)
    Xg, Yg = make_regular_grid(x0, y0)
    cyl_mask = _mask_cylinder(Xg, Yg)

    cmap = "magma" if field == "pressure" else "inferno"
    cbar_label = "Pressure (Pa)" if field == "pressure" else "Velocity (m/s)"
    re_colors = {120: "#ff6b6b", 200: "#ffd93d", 500: "#6bcb77"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(f"{label} — Re Sweep: {field.title()} Field", fontsize=14,
                 color="white", y=1.02)

    # Pre-load first frame for each panel
    data_2d = {}
    for dir_name, re, steps in cases:
        vtu = load_vtu(common_steps[0], dir_name)
        x, y, p, u, v, vm = get_2d_data(vtu)
        if field == "pressure":
            vals = interp_to_grid(x, y, p, Xg, Yg)
            us, vs = None, None
        else:
            vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
            us = interp_to_grid(x, y, u, Xg, Yg)
            vs = interp_to_grid(x, y, v, Xg, Yg)
        data_2d[re] = (vals, us, vs)

    # Initialize axes
    time_texts = []
    for idx, (dir_name, re, steps) in enumerate(cases):
        ax = axes[idx]
        ax.set_facecolor("#111111")
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Re = {re}", fontsize=12, color=re_colors[re])

        vals_masked = np.ma.masked_where(cyl_mask, data_2d[re][0])
        cntr = ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")
        circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
        ax.add_patch(circle)

        if field == "velocity":
            stride = 3
            sl_color = "white" if spin_name == "nospin" else "black"
            ax.streamplot(Xg[::stride, ::stride], Yg[::stride, ::stride],
                          data_2d[re][1][::stride, ::stride],
                          data_2d[re][2][::stride, ::stride],
                          color=sl_color, linewidth=0.6, density=0.8, arrowsize=0.6)

        tt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8,
                     color="white", bbox=dict(boxstyle="round", fc="0.2", ec="none"))
        time_texts.append(tt)

    fig.tight_layout()
    # Add single colorbar
    cbar = fig.colorbar(axes[0].collections[0] if axes[0].collections else cntr,
                        ax=axes, label=cbar_label, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    cbar.outline.set_edgecolor("white")

    suffix = f"{spin_name}_{field}_re_comparison"
    out_path = IMAGES / f"cylinder_{suffix}.mp4"
    writer = FFMpegWriter(fps=FPS, bitrate=5000)

    with writer.saving(fig, str(out_path), dpi=150):
        for i, step in enumerate(common_steps):
            for idx, (dir_name, re, steps) in enumerate(cases):
                ax = axes[idx]
                vtu = load_vtu(step, dir_name)
                x, y, p, u, v, vm = get_2d_data(vtu)
                if field == "pressure":
                    vals = interp_to_grid(x, y, p, Xg, Yg)
                else:
                    vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
                vals_masked = np.ma.masked_where(cyl_mask, vals)

                for c in ax.collections[:]:
                    if isinstance(c, QuadContourSet):
                        c.remove()
                ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")

                if field == "velocity":
                    for coll in ax.lines[:]:
                        coll.remove()
                    us = interp_to_grid(x, y, u, Xg, Yg)
                    vs = interp_to_grid(x, y, v, Xg, Yg)
                    stride = 3
                    sl_color = "white" if spin_name == "nospin" else "black"
                    ax.streamplot(Xg[::stride, ::stride], Yg[::stride, ::stride],
                                  us[::stride, ::stride], vs[::stride, ::stride],
                                  color=sl_color, linewidth=0.6, density=0.8, arrowsize=0.6)

                t = step * DT
                time_texts[idx].set_text(f"t = {t:.1f} s")

            writer.grab_frame()

    plt.close()
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GENERATING VISUALIZATIONS — LAMINAR RE SWEEP")
    print("=" * 60)

    cases = discover_cases()
    if not cases:
        print("\n  No output_*_lam_re* directories found.")
        print("  Run run_unsteady.py first.")
        sys.exit(1)

    print(f"\n  Found {len(cases)} case directories:")
    for d, s, r in cases:
        steps = len(get_step_range(d))
        print(f"    {d}/ — {steps} steps")

    # 1. Mesh
    plot_mesh()

    # 2. Static 2D plots (per case)
    static_plots()

    # 3. Per-case animations (pressure + velocity)
    for case_dir, spin_name, re in cases:
        animate_flow(case_dir, spin_name, re, field="pressure")
        animate_flow(case_dir, spin_name, re, field="velocity")

    # 4. Re comparison animations (3-panel)
    for spin_name in ["nospin", "magnus"]:
        animate_re_comparison(spin_name, field="pressure")
        animate_re_comparison(spin_name, field="velocity")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
