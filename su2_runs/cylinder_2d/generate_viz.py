"""Generate visualizations from unsteady laminar Re=120 cylinder runs.
Produces: mesh PNG, static 2D field plots PNG, animated 2D MP4, 3D interactive HTML.
"""
import sys, os
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
IMAGES = WORKDIR.parent.parent / "docs" / "images"
DT = 0.15
CYL_R = 0.3
FS = 1.0  # uniform farfield stream

# ── Helpers ──

def load_vtu(step: int, case_dir: str) -> pv.DataSet:
    """Load a VTU file for a given time step and case."""
    path = WORKDIR / case_dir / f"vol_solution_{step:05d}.vtu"
    return pv.read(str(path))


def get_2d_data(vtu: pv.DataSet) -> tuple:
    """Extract 2D coordinates, pressure, velocity from VTU."""
    pts = vtu.points
    x, y = pts[:, 0], pts[:, 1]
    p = vtu["Pressure"]
    vel = vtu["Velocity"]
    u, v = vel[:, 0], vel[:, 1]
    vel_mag = np.sqrt(u**2 + v**2)
    return x, y, p, u, v, vel_mag


def make_regular_grid(x, y, nx=200, ny=100):
    """Create a regular grid for streamplot, clipped to domain bounds."""
    margin = 0.5
    xs = np.linspace(x.min() + margin, x.max() - margin, nx)
    ys = np.linspace(y.min() + margin, y.max() - margin, ny)
    return np.meshgrid(xs, ys)


def interp_to_grid(x, y, values, Xg, Yg):
    """Linear interpolation of scattered data to a regular grid via PyVista."""
    # Build a temporary mesh from points for griddata
    from scipy.interpolate import griddata
    points = np.column_stack([x, y])
    return griddata(points, values, (Xg, Yg), method="linear")


def _mask_cylinder(xg, yg):
    """Mask points inside the cylinder."""
    r = np.sqrt(xg**2 + yg**2)
    return r < CYL_R * 1.05


def plot_2d_pressure(ax, x, y, p, xg, yg, pg, title="", cmap="RdBu_r"):
    """Pressure contourf on an axis."""
    pg_masked = np.ma.masked_where(_mask_cylinder(xg, yg), pg)
    cntr = ax.contourf(xg, yg, pg_masked, levels=40, cmap=cmap, extend="both")
    ax.contour(xg, yg, pg_masked, levels=10, colors="k", linewidths=0.3, alpha=0.3)
    # Cylinder
    circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-3, 3)
    ax.set_title(title, fontsize=11)
    return cntr


def plot_2d_velocity(ax, x, y, vel_mag, u, v, xg, yg, ug, vg, title="", cmap="viridis"):
    """Velocity magnitude contourf + streamlines on an axis."""
    vg_masked = np.ma.masked_where(_mask_cylinder(xg, yg), np.sqrt(ug**2 + vg**2))
    cntr = ax.contourf(xg, yg, vg_masked, levels=40, cmap=cmap, extend="both")
    # Streamlines — use finer stride for sparser lines
    stride = 3
    ax.streamplot(xg[::stride, ::stride], yg[::stride, ::stride],
                  ug[::stride, ::stride], vg[::stride, ::stride],
                  color="white", linewidth=0.6, density=0.8, arrowsize=0.6)
    circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-3, 3)
    ax.set_title(title, fontsize=11)
    return cntr


def get_step_range(case_dir: str, start=0, end=200):
    """List of available VTU step numbers."""
    steps = []
    for s in range(start, end):
        path = WORKDIR / case_dir / f"vol_solution_{s:05d}.vtu"
        if path.exists():
            steps.append(s)
    return steps


# ═══════════════════════════════════════════════════════════
# 1. MESH VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_mesh():
    print("--- Mesh visualization ---")
    # Load a VTU to get the mesh geometry (first step of any case)
    vtu = load_vtu(0, "output_nospin_lam")

    # Full domain view
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.add_mesh(vtu, show_edges=True, color="lightblue", opacity=0.6,
                     lighting=False, label="Mesh")
    edges = vtu.separate_cells().extract_feature_edges()
    plotter.add_mesh(edges, color="grey", line_width=0.3)
    plotter.view_xy()
    plotter.camera.zoom(0.6)
    plotter.screenshot(str(IMAGES / "cylinder_mesh.png"))
    print(f"  Saved: cylinder_mesh.png")
    plotter.close()

    # Zoom to cylinder
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.add_mesh(vtu, show_edges=True, color="lightblue", opacity=0.6,
                     lighting=False, label="Mesh")
    edges = vtu.separate_cells().extract_feature_edges()
    plotter.add_mesh(edges, color="grey", line_width=0.3)
    plotter.view_xy()
    plotter.camera.zoom(6.0)
    plotter.screenshot(str(IMAGES / "cylinder_mesh_zoom.png"))
    print(f"  Saved: cylinder_mesh_zoom.png")
    plotter.close()


# ═══════════════════════════════════════════════════════════
# 2. STATIC 2D FIELD PLOTS
# ═══════════════════════════════════════════════════════════

def static_plots():
    print("\n--- Static 2D field plots ---")
    # Use last timestep for each case
    case_steps = [("output_nospin_lam", "No Spin"),
                  ("output_magnus_lam", "Magnus")]
    data = {}
    for case_dir, label in case_steps:
        steps = get_step_range(case_dir)
        vtu = load_vtu(steps[-1], case_dir) if steps else load_vtu(0, case_dir)
        x, y, p, u, v, vm = get_2d_data(vtu)
        Xg, Yg = make_regular_grid(x, y)
        Pg = interp_to_grid(x, y, p, Xg, Yg)
        Ug = interp_to_grid(x, y, u, Xg, Yg)
        Vg = interp_to_grid(x, y, v, Xg, Yg)
        data[label] = (x, y, p, u, v, vm, Xg, Yg, Pg, Ug, Vg)

    # ── Pressure: individual + side-by-side ──
    for label, suffix in [("No Spin", "nospin"), ("Magnus", "magnus")]:
        d = data[label]
        fig, ax = plt.subplots(figsize=(8, 4))
        cntr = plot_2d_pressure(ax, d[0], d[1], d[2], d[6], d[7], d[8],
                                title=f"{label} — Pressure Field (Re=120)")
        cbar = fig.colorbar(cntr, ax=ax, label="Pressure (Pa)", shrink=0.85)
        fig.tight_layout()
        fig.savefig(IMAGES / f"cylinder_{suffix}_pressure.png", dpi=150)
        plt.close()
        print(f"  Saved: cylinder_{suffix}_pressure.png")

    # Side-by-side pressure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    d1 = data["No Spin"]
    d2 = data["Magnus"]
    c1 = plot_2d_pressure(ax1, d1[0], d1[1], d1[2], d1[6], d1[7], d1[8],
                          title="No Spin — Pressure")
    c2 = plot_2d_pressure(ax2, d2[0], d2[1], d2[2], d2[6], d2[7], d2[8],
                          title="Magnus (S=0.2) — Pressure")
    fig.colorbar(c1, ax=[ax1, ax2], label="Pressure (Pa)", shrink=0.85)
    fig.suptitle("Pressure Field Comparison @ Re=120", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES / "cylinder_compare_pressure.png", dpi=150)
    plt.close()
    print(f"  Saved: cylinder_compare_pressure.png")

    # ── Velocity: individual + side-by-side ──
    for label, suffix in [("No Spin", "nospin"), ("Magnus", "magnus")]:
        d = data[label]
        fig, ax = plt.subplots(figsize=(8, 4))
        cntr = plot_2d_velocity(ax, d[0], d[1], d[5], d[3], d[4],
                                d[6], d[7], d[9], d[10],
                                title=f"{label} — Velocity + Streamlines")
        cbar = fig.colorbar(cntr, ax=ax, label="Velocity (m/s)", shrink=0.85)
        fig.tight_layout()
        fig.savefig(IMAGES / f"cylinder_{suffix}_velocity.png", dpi=150)
        plt.close()
        print(f"  Saved: cylinder_{suffix}_velocity.png")

    # Side-by-side velocity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    c1 = plot_2d_velocity(ax1, d1[0], d1[1], d1[5], d1[3], d1[4],
                           d1[6], d1[7], d1[9], d1[10],
                           title="No Spin — Velocity + Streamlines")
    c2 = plot_2d_velocity(ax2, d2[0], d2[1], d2[5], d2[3], d2[4],
                           d2[6], d2[7], d2[9], d2[10],
                           title="Magnus (S=0.2) — Velocity + Streamlines")
    fig.colorbar(c1, ax=[ax1, ax2], label="Velocity (m/s)", shrink=0.85)
    fig.suptitle("Velocity Field Comparison @ Re=120", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES / "cylinder_compare_velocity.png", dpi=150)
    plt.close()
    print(f"  Saved: cylinder_compare_velocity.png")


# ═══════════════════════════════════════════════════════════
# 3. ANIMATED 2D PHIFLOW-STYLE (MP4, 10s each)
# ═══════════════════════════════════════════════════════════

def animate_flow(case_dir: str, label: str, suffix: str,
                 field: str = "pressure", fps: int = 20):
    """Generate a 10s MP4 animation of pressure or velocity field."""
    steps = get_step_range(case_dir)
    if len(steps) < 2:
        print(f"  [SKIP] Not enough steps for {label}")
        return
    N = len(steps)
    duration = N * DT
    print(f"\n  Animating {label} ({field}, {N} frames, {duration:.1f}s @ {fps}fps)")

    # Pre-load all VTU data (or load on-demand for memory)
    vtu0 = load_vtu(steps[0], case_dir)
    x0, y0, p0, u0, v0, vm0 = get_2d_data(vtu0)
    Xg, Yg = make_regular_grid(x0, y0)
    cyl_mask = _mask_cylinder(Xg, Yg)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = "RdBu_r" if field == "pressure" else "viridis"
    cbar_label = "Pressure (Pa)" if field == "pressure" else "Velocity (m/s)"

    # First frame
    vtu = load_vtu(steps[0], case_dir)
    x, y, p, u, v, vm = get_2d_data(vtu)
    if field == "pressure":
        vals = interp_to_grid(x, y, p, Xg, Yg)
    else:
        vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
    vals_masked = np.ma.masked_where(cyl_mask, vals)
    cntr = ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")
    circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 10)
    ax.set_ylim(-3, 3)
    ax.set_title(f"{label} — {'Pressure' if field == 'pressure' else 'Velocity'} Field", fontsize=11)
    cbar = fig.colorbar(cntr, ax=ax, label=cbar_label, shrink=0.85)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                        color="white", bbox=dict(boxstyle="round", fc="0.2", ec="none"))
    fig.tight_layout()

    writer = FFMpegWriter(fps=fps, bitrate=3000)
    out_path = IMAGES / f"cylinder_{suffix}_{field}.mp4"

    with writer.saving(fig, str(out_path), dpi=150):
        for i, step in enumerate(steps):
            vtu = load_vtu(step, case_dir)
            x, y, p, u, v, vm = get_2d_data(vtu)
            if field == "pressure":
                vals = interp_to_grid(x, y, p, Xg, Yg)
            else:
                vals = interp_to_grid(x, y, np.sqrt(u**2 + v**2), Xg, Yg)
            vals_masked = np.ma.masked_where(cyl_mask, vals)

            # Remove previous contour
            for c in ax.collections[:]:
                if isinstance(c, QuadContourSet):
                    c.remove()
            cntr = ax.contourf(Xg, Yg, vals_masked, levels=40, cmap=cmap, extend="both")

            t = step * DT
            time_text.set_text(f"t = {t:.1f} s")

            writer.grab_frame()

    plt.close()
    print(f"  Saved: {out_path.name}")


def animate_comparison(fps: int = 20):
    """Side-by-side MP4: no-spin vs magnus pressure field."""
    steps_ns = get_step_range("output_nospin_lam")
    steps_mg = get_step_range("output_magnus_lam")
    steps = sorted(set(steps_ns) & set(steps_mg))
    if len(steps) < 2:
        print("  [SKIP] Not enough overlapping steps")
        return
    N = len(steps)
    print(f"\n  Animating side-by-side comparison ({N} frames @ {fps}fps)")

    vtu0 = load_vtu(steps[0], "output_nospin_lam")
    x0, y0, p0, u0, v0, vm0 = get_2d_data(vtu0)
    Xg, Yg = make_regular_grid(x0, y0)
    cyl_mask = _mask_cylinder(Xg, Yg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Pressure Field: No Spin vs Magnus (S=0.2) @ Re=120", fontsize=13, y=1.02)

    cmap = "RdBu_r"

    def init_ax(ax, idx):
        d = load_vtu(steps[0], ["output_nospin_lam", "output_magnus_lam"][idx])
        x, y, p, u, v, vm = get_2d_data(d)
        pg = interp_to_grid(x, y, p, Xg, Yg)
        vals = np.ma.masked_where(cyl_mask, pg)
        cntr = ax.contourf(Xg, Yg, vals, levels=40, cmap=cmap, extend="both")
        circle = plt.Circle((0, 0), CYL_R, color="0.85", ec="k", lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 10)
        ax.set_ylim(-3, 3)
        ax.set_title(["No Spin", "Magnus (S=0.2)"][idx], fontsize=11)
        return cntr

    c1 = init_ax(ax1, 0)
    c2 = init_ax(ax2, 1)
    fig.colorbar(c1, ax=[ax1, ax2], label="Pressure (Pa)", shrink=0.85)
    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10,
                         color="white", bbox=dict(boxstyle="round", fc="0.2", ec="none"))
    fig.tight_layout()

    writer = FFMpegWriter(fps=fps, bitrate=4000)
    out_path = IMAGES / "cylinder_compare_flow.mp4"

    with writer.saving(fig, str(out_path), dpi=150):
        for i, step in enumerate(steps):
            for idx, (case_dir, ax) in enumerate([
                ("output_nospin_lam", ax1), ("output_magnus_lam", ax2)
            ]):
                vtu = load_vtu(step, case_dir)
                x, y, p, u, v, vm = get_2d_data(vtu)
                pg = interp_to_grid(x, y, p, Xg, Yg)
                vals = np.ma.masked_where(cyl_mask, pg)
                for c in ax.collections[:]:
                    if isinstance(c, QuadContourSet):
                        c.remove()
                ax.contourf(Xg, Yg, vals, levels=40, cmap=cmap, extend="both")
            time_text.set_text(f"t = {step * DT:.1f} s")
            writer.grab_frame()

    plt.close()
    print(f"  Saved: cylinder_compare_flow.mp4")


# ═══════════════════════════════════════════════════════════
# 4. 3D INTERACTIVE HTML (PyVista → Three.js)
# ═══════════════════════════════════════════════════════════

def export_3d_html(case_dir: str, label: str, suffix: str):
    """Export a 3D interactive HTML view of the pressure field."""
    print(f"\n--- 3D interactive: {label} ---")
    steps = get_step_range(case_dir)
    if not steps:
        print(f"  [SKIP] No VTU files for {label}")
        return
    vtu = load_vtu(steps[-1], case_dir)  # last timestep

    # Create a copy with pressure as the warp scalar
    mesh = vtu.copy()
    # Warp by pressure (height) for visual impact
    warp = mesh.warp_by_scalar("Pressure", factor=0.001)

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.add_mesh(warp, scalars="Pressure", cmap="RdBu_r",
                     show_edges=False, lighting=True,
                     scalar_bar_args={"title": "Pressure (Pa)"})
    plotter.view_xy()
    plotter.camera.zoom(0.8)
    plotter.camera_position = [
        (8, 0, 6),   # camera position
        (0, 0, 0),   # focal point
        (0, 1, 1),   # view up
    ]
    plotter.show_bounds(grid="back", location="outer", all_edges=True)

    out_path = IMAGES / f"cylinder_{suffix}_3d.html"
    plotter.export_html(str(out_path))
    print(f"  Saved: {out_path.name}")
    plotter.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    # 1. Mesh
    plot_mesh()

    # 2. Static 2D plots
    static_plots()

    # 3. Animated 2D MP4s
    # Pressure animations
    animate_flow("output_nospin_lam", "No Spin", "nospin", field="pressure")
    animate_flow("output_magnus_lam", "Magnus", "magnus", field="pressure")
    # Velocity animations
    animate_flow("output_nospin_lam", "No Spin", "nospin", field="velocity")
    animate_flow("output_magnus_lam", "Magnus", "magnus", field="velocity")
    # Side-by-side comparison
    animate_comparison()

    # 4. 3D interactive HTML
    export_3d_html("output_nospin_lam", "No Spin", "nospin")
    export_3d_html("output_magnus_lam", "Magnus", "magnus")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
