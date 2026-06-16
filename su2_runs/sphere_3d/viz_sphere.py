"""Generate 3D sphere flow visualization images (pressure, velocity, streamlines)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pyvista as pv
from pathlib import Path

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_sphere"
VTU_PATH = WORKDIR / "vol_solution.vtu"

SPHERE_RADIUS = 0.11

# ── Settings ──
pv.OFF_SCREEN = True
pv.set_plot_theme("dark")
BG = "#111111"
TITLE_COLOR = "white"
CMAP_CP = "coolwarm"
CMAP_VEL = "turbo"
RESOLUTION = (1920, 1080)


def load_solution():
    return pv.read(str(VTU_PATH))


def sphere_surface(mesh, theta=60, phi=60):
    """Create sampled sphere surface with volume data mapped onto it."""
    sphere = pv.Sphere(radius=SPHERE_RADIUS, theta_resolution=theta,
                       phi_resolution=phi)
    return sphere.sample(mesh)


def get_midplane(mesh, normal=(0, 1, 0)):
    return mesh.slice(normal=normal, origin=(0, 0, 0))


def generate_streamlines(mesh, n_seeds=15):
    x_start = -0.30
    yz = np.linspace(-0.18, 0.18, n_seeds)
    YY, ZZ = np.meshgrid(yz, yz)
    points = np.column_stack([np.full(YY.size, x_start), YY.ravel(), ZZ.ravel()])
    dist = np.linalg.norm(points, axis=1)
    points = points[dist > SPHERE_RADIUS * 1.1]
    seeds = pv.PolyData(points)
    return mesh.streamlines_from_source(
        seeds,
        vectors="Velocity",
        integration_direction="forward",
        max_length=3.0,
        initial_step_length=0.005,
        max_steps=1000,
    )


def plot_pressure_surface(sphere_surf):
    """3D sphere surface colored by Cp."""
    p = pv.Plotter(window_size=RESOLUTION)
    p.set_background(BG)
    p.add_mesh(sphere_surf, scalars="Pressure_Coefficient", cmap=CMAP_CP,
               smooth_shading=True,
               scalar_bar_args=dict(title="Cp", color=TITLE_COLOR,
                                    title_font_size=14, label_font_size=10))
    p.add_text("SU2 3D Sphere — Pressure Coefficient (Re = 1×10⁶)",
               position="upper_edge", color=TITLE_COLOR, font_size=14)
    p.camera_position = [(0.35, 0.25, 0.20), (0, 0, 0), (0, 0, 1)]
    p.show(screenshot=str(IMAGES / "sphere_pressure_surface.png"))
    p.close()


def plot_midplane_pressure(mesh):
    """Mid-plane slice with Cp field."""
    plane = get_midplane(mesh)
    p = pv.Plotter(window_size=RESOLUTION)
    p.set_background(BG)
    p.add_mesh(plane, scalars="Pressure_Coefficient", cmap=CMAP_CP,
               scalar_bar_args=dict(title="Cp", color=TITLE_COLOR,
                                    title_font_size=14, label_font_size=10))
    outline = pv.Sphere(radius=SPHERE_RADIUS)
    p.add_mesh(outline, color="white", opacity=0.25, style="wireframe",
               line_width=2)
    p.add_text("SU2 3D Sphere — Pressure Field, Mid-Plane (Re = 1×10⁶)",
               position="upper_edge", color=TITLE_COLOR, font_size=14)
    p.show(screenshot=str(IMAGES / "sphere_pressure_plane.png"))
    p.close()


def plot_midplane_velocity(mesh):
    """Mid-plane slice with velocity magnitude."""
    plane = get_midplane(mesh)
    plane["Velocity_Mag"] = np.linalg.norm(plane.point_data["Velocity"], axis=1)
    p = pv.Plotter(window_size=RESOLUTION)
    p.set_background(BG)
    p.add_mesh(plane, scalars="Velocity_Mag", cmap=CMAP_VEL,
               scalar_bar_args=dict(title="Velocity", color=TITLE_COLOR,
                                    title_font_size=14, label_font_size=10))
    outline = pv.Sphere(radius=SPHERE_RADIUS)
    p.add_mesh(outline, color="white", opacity=0.25, style="wireframe",
               line_width=2)
    p.add_text("SU2 3D Sphere — Velocity Magnitude, Mid-Plane (Re = 1×10⁶)",
               position="upper_edge", color=TITLE_COLOR, font_size=14)
    p.show(screenshot=str(IMAGES / "sphere_velocity_plane.png"))
    p.close()


def plot_streamlines(mesh):
    """Streamlines over velocity magnitude background."""
    plane = get_midplane(mesh)
    plane["Velocity_Mag"] = np.linalg.norm(plane.point_data["Velocity"], axis=1)
    streamlines = generate_streamlines(mesh, n_seeds=12)
    p = pv.Plotter(window_size=RESOLUTION)
    p.set_background(BG)
    p.add_mesh(plane, scalars="Velocity_Mag", cmap=CMAP_VEL, opacity=0.35,
               scalar_bar_args=dict(title="Velocity", color=TITLE_COLOR,
                                    title_font_size=14, label_font_size=10))
    p.add_mesh(streamlines, line_width=2.0, color="#f5a623")
    outline = pv.Sphere(radius=SPHERE_RADIUS)
    p.add_mesh(outline, color="white", style="wireframe", line_width=2)
    p.add_text("SU2 3D Sphere — Streamlines + Velocity (Re = 1×10⁶)",
               position="upper_edge", color=TITLE_COLOR, font_size=14)
    p.show(screenshot=str(IMAGES / "sphere_streamlines.png"))
    p.close()


def plot_combined(sphere_surf, mesh):
    """3D surface Cp + streamlines wrapping around ball."""
    streamlines = generate_streamlines(mesh, n_seeds=10)
    p = pv.Plotter(window_size=RESOLUTION)
    p.set_background(BG)
    p.add_mesh(sphere_surf, scalars="Pressure_Coefficient", cmap=CMAP_CP,
               smooth_shading=True,
               scalar_bar_args=dict(title="Cp", color=TITLE_COLOR,
                                    title_font_size=14, label_font_size=10))
    p.add_mesh(streamlines, line_width=1.5, color="#f5a623", opacity=0.7)
    p.add_text("SU2 3D Sphere — Cp + Streamlines (Re = 1×10⁶)",
               position="upper_edge", color=TITLE_COLOR, font_size=14)
    p.camera_position = [(0.40, 0.20, 0.25), (0, 0, 0), (0, 0, 1)]
    p.show(screenshot=str(IMAGES / "sphere_surface_streamlines.png"))
    p.close()


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    print("Loading solution...")
    mesh = load_solution()
    print(f"  Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    print("Sampling sphere surface...")
    surf = sphere_surface(mesh)
    print(f"  Surface: {surf.n_points} points, {surf.n_cells} triangles")
    cp = surf.point_data["Pressure_Coefficient"]
    print(f"  Cp range: {cp.min():.4f} to {cp.max():.4f}")

    print("\n--- Generating visualizations ---\n")

    print("[1/5] 3D pressure surface...")
    plot_pressure_surface(surf)

    print("[2/5] Mid-plane pressure field...")
    plot_midplane_pressure(mesh)

    print("[3/5] Mid-plane velocity field...")
    plot_midplane_velocity(mesh)

    print("[4/5] Streamlines...")
    plot_streamlines(mesh)

    print("[5/5] Combined: Cp + streamlines...")
    plot_combined(surf, mesh)

    print(f"\nAll images saved to {IMAGES}/")


if __name__ == "__main__":
    main()
