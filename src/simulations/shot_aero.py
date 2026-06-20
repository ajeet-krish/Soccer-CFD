"""Shot Aerodynamics — Magnus vs Knuckleball cylinder simulation."""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi.jax.flow import Obstacle, StaggeredGrid, CenteredGrid, fluid, advect, vec, field, ZERO_GRADIENT

from src.core.utils import setup_theme, ASSETS, IMAGES
from src.core.domain import BOX, BOUNDARY, CENTER, RADIUS, CYLINDER, CENTERED, DT, N_STEPS, SAVE_EVERY

setup_theme()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from tqdm import trange


def run_cylinder(obstacle, label):
    v0 = StaggeredGrid((1.0, 0.0), BOUNDARY, x=256, y=128, bounds=BOX)
    v, _ = fluid.make_incompressible(v0, ())
    frames = []
    v = v0
    for i in trange(N_STEPS, desc=label):
        v = advect.semi_lagrangian(v, v, dt=DT)
        v, p = fluid.make_incompressible(v, obstacle)
        if i % SAVE_EVERY == 0:
            vc = v @ CENTERED
            u_c = np.array(vc.vector[0].values.native("x", "y"))
            v_c = np.array(vc.vector[1].values.native("x", "y"))
            vel_mag = field.vec_length(v)
            p_np = np.array(p.values.native("x", "y"))
            frames.append({
                "pressure": p_np,
                "u": u_c,
                "v": v_c,
                "velocity": np.array(vel_mag.values.native("x", "y")),
                "step": i,
            })
    return frames


def animate_velocity_comparison(frames_magnus, frames_knuckle):
    n_frames = min(len(frames_magnus), len(frames_knuckle))
    fm = frames_magnus[:n_frames]
    fk = frames_knuckle[:n_frames]

    x_s = np.linspace(0, 8.0, 256)
    y_s = np.linspace(0, 4.0, 128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig.suptitle("Velocity Field — Magnus vs Knuckleball  (U = 1.0 m/s, Re = 4\u00d710\u2074)", color="white", fontsize=14, y=0.98)

    titles = [
        "Magnus (\u03c9 = 10 rad/s)",
        "Knuckleball (\u03c9 = 0 rad/s)",
    ]
    frames_list = [fm, fk]
    imgs = []

    for ax, title, frames in zip([ax1, ax2], titles, frames_list):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 3.5)

        im = ax.imshow(
            frames[0]["velocity"].T,
            origin="lower", cmap="inferno", aspect="auto",
            extent=[0, 8, 0, 4], vmin=0, vmax=1.5,
        )
        ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))
        ax.streamplot(
            x_s, y_s, frames[0]["u"].T, frames[0]["v"].T,
            color="black",
            linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6,
        )
        imgs.append(im)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cbar = fig.colorbar(imgs[0], ax=[ax1, ax2], fraction=0.04, pad=0.02)
    cbar.set_label("Velocity magnitude (m/s)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        for ax, frames in zip([ax1, ax2], frames_list):
            ax.clear()
            ax.set_facecolor("#111111")
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(
                frames[idx]["velocity"].T,
                origin="lower", cmap="inferno", aspect="auto",
                extent=[0, 8, 0, 4], vmin=0, vmax=1.5,
            )
            ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))
            ax.streamplot(
                x_s, y_s, frames[idx]["u"].T, frames[idx]["v"].T,
                color="black",
                linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6,
            )
        return (ax1, ax2)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    path = IMAGES / "phiflow_cylinder_2d" / "velocity_comparison.mp4"
    anim.save(str(path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def animate_pressure_comparison(frames_magnus, frames_knuckle):
    n_frames = min(len(frames_magnus), len(frames_knuckle))
    fm = frames_magnus[:n_frames]
    fk = frames_knuckle[:n_frames]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig.suptitle("Pressure Field — Magnus vs Knuckleball  (U = 1.0 m/s, Re = 4\u00d710\u2074)", color="white", fontsize=14, y=0.98)

    titles = [
        "Magnus (\u03c9 = 10 rad/s)",
        "Knuckleball (\u03c9 = 0 rad/s)",
    ]
    frames_list = [fm, fk]
    imgs = []

    for ax, title, frames in zip([ax1, ax2], titles, frames_list):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 3.5)

        p0 = frames[0]["pressure"]
        im = ax.imshow(p0.T, origin="lower", cmap="Spectral", aspect="auto", extent=[0, 8, 0, 4])
        ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))
        imgs.append(im)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cbar = fig.colorbar(imgs[0], ax=[ax1, ax2], fraction=0.04, pad=0.02)
    cbar.set_label("Pressure (Pa)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        for ax, frames, im in zip([ax1, ax2], frames_list, imgs):
            data = frames[idx]["pressure"]
            im.set_data(data.T)
            im.set_clim(data.min(), data.max())
        return imgs

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
    path = IMAGES / "phiflow_cylinder_2d" / "pressure_comparison.mp4"
    anim.save(str(path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def _animate_single(frames, output_path, *, title, cmap, vmin, vmax, cbar_label,
                    data_fn, streamlines_fn=None):
    """Generic single-panel animation for pressure, velocity, or vorticity."""
    n = len(frames)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), facecolor="#111111")
    fig.suptitle(title, color="white", fontsize=13, y=0.96)

    ax.set_facecolor("#111111")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])

    d0 = data_fn(frames[0])
    im = ax.imshow(d0.T, origin="lower", cmap=cmap, aspect="auto",
                   extent=[0, 8, 0, 4], vmin=vmin, vmax=vmax)
    ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))
    if streamlines_fn:
        streamlines_fn(ax, frames[0])

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        ax.clear()
        ax.set_facecolor("#111111")
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_xticks([])
        ax.set_yticks([])

        d = data_fn(frames[idx])
        ax.imshow(d.T, origin="lower", cmap=cmap, aspect="auto",
                   extent=[0, 8, 0, 4], vmin=vmin, vmax=vmax)
        ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))
        if streamlines_fn:
            streamlines_fn(ax, frames[idx])
        return (ax,)

    anim = FuncAnimation(fig, update, frames=n, interval=80, blit=False)
    anim.save(str(output_path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def animate_magnus_pressure(frames):
    all_p = np.concatenate([f["pressure"].ravel() for f in frames[1:]])
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "magnus_pressure.mp4",
                    title="Magnus Effect — Pressure Field  (U = 1.0 m/s, Re = 4\u00d710\u2074)",
                    cmap="Spectral", vmin=float(all_p.min()), vmax=float(all_p.max()),
                    cbar_label="Pressure (Pa)",
                    data_fn=lambda f: f["pressure"])


def animate_magnus_velocity(frames):
    x_s = np.linspace(0, 8.0, 256)
    y_s = np.linspace(0, 4.0, 128)
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "magnus_velocity.mp4",
                    title="Magnus Effect — Velocity Field  (U = 1.0 m/s, Re = 4\u00d710\u2074)",
                    cmap="inferno", vmin=0, vmax=1.5,
                    cbar_label="Velocity magnitude (m/s)",
                    data_fn=lambda f: f["velocity"],
                    streamlines_fn=lambda ax, f: ax.streamplot(
                        x_s, y_s, f["u"].T, f["v"].T,
                        color="black",
                        linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6))


def animate_magnus_vorticity(frames):
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "magnus_vorticity.mp4",
                    title="Magnus Effect — Vorticity Field",
                    cmap="RdBu", vmin=-4, vmax=4,
                    cbar_label="Vorticity ω_z",
                    data_fn=lambda f: np.gradient(f["v"], axis=1) - np.gradient(f["u"], axis=0))


def animate_knuckle_pressure(frames):
    all_p = np.concatenate([f["pressure"].ravel() for f in frames[1:]])
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "knuckleball_pressure_individual.mp4",
                    title="Knuckleball — Pressure Field  (U = 1.0 m/s, Re = 4\u00d710\u2074)",
                    cmap="Spectral", vmin=float(all_p.min()), vmax=float(all_p.max()),
                    cbar_label="Pressure (Pa)",
                    data_fn=lambda f: f["pressure"])


def animate_knuckle_velocity(frames):
    x_s = np.linspace(0, 8.0, 256)
    y_s = np.linspace(0, 4.0, 128)
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "knuckleball_velocity.mp4",
                    title="Knuckleball — Velocity Field  (U = 1.0 m/s, Re = 4\u00d710\u2074)",
                    cmap="inferno", vmin=0, vmax=1.5,
                    cbar_label="Velocity magnitude (m/s)",
                    data_fn=lambda f: f["velocity"],
                    streamlines_fn=lambda ax, f: ax.streamplot(
                        x_s, y_s, f["u"].T, f["v"].T,
                        color="white",
                        linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6))


def animate_knuckle_vorticity(frames):
    _animate_single(frames, IMAGES / "phiflow_cylinder_2d" / "knuckleball_vorticity.mp4",
                    title="Knuckleball — Vorticity Field",
                    cmap="RdBu", vmin=-4, vmax=4,
                    cbar_label="Vorticity ω_z",
                    data_fn=lambda f: np.gradient(f["v"], axis=1) - np.gradient(f["u"], axis=0))


CACHE_FILE = ASSETS / "_frames_cache.pkl"
PHIFLOW_IMAGES = IMAGES / "phiflow_cylinder_2d"


def extract_static_frames(frames_magnus, frames_knuckle, output_dir=None):
    """Extract static PNG frames at frame 095 (last frame)."""
    if output_dir is None:
        output_dir = PHIFLOW_IMAGES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = [95]

    x_s = np.linspace(0, 8.0, 256)
    y_s = np.linspace(0, 4.0, 128)

    def _extract(frames, slug, title, field, data_key, cmap, vmin, vmax, cbar_label):
        for idx in frame_indices:
            if idx >= len(frames):
                continue
            data = frames[idx][data_key]
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#111111")
            ax.set_facecolor("#111111")
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto",
                           extent=[0, 8, 0, 4], vmin=vmin, vmax=vmax)
            ax.add_patch(Circle(CENTER, RADIUS, facecolor="black", edgecolor="white", linewidth=2.0))

            # Streamlines for velocity fields
            if field == "velocity":
                u = frames[idx]["u"]
                v = frames[idx]["v"]
                stride = 2
                streamline_color = "black" if slug == "magnus" else "white"
                ax.streamplot(x_s, y_s, u.T, v.T, color=streamline_color, linewidth=0.7,
                              density=1.2, arrowstyle="->", arrowsize=0.5)

            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(cbar_label, color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

            ax.set_title(f"{title} — {field} Field", color="white", fontsize=10)

            fig.tight_layout()

            out_path = output_dir / f"phiflow_{slug}_{field.lower()}_frame{idx:03d}.png"
            fig.savefig(out_path, dpi=150, facecolor="#111111", bbox_inches="tight")
            print(f"  Saved: {out_path.name}")
            plt.close(fig)

    # Pressure fields
    all_p_m = np.concatenate([f["pressure"].ravel() for f in frames_magnus[1:]])
    all_p_k = np.concatenate([f["pressure"].ravel() for f in frames_knuckle[1:]])
    pmin = min(float(all_p_m.min()), float(all_p_k.min()))
    pmax = max(float(all_p_m.max()), float(all_p_k.max()))

    _extract(frames_magnus, "magnus", "Magnus", "pressure", "pressure", "Spectral", pmin, pmax, "Pressure (Pa)")
    _extract(frames_knuckle, "knuckleball", "Knuckleball", "pressure", "pressure", "Spectral", pmin, pmax, "Pressure (Pa)")

    # Velocity fields
    _extract(frames_magnus, "magnus", "Magnus (Spin)", "velocity", "velocity", "inferno", 0, 1.5, "Velocity (m/s)")
    _extract(frames_knuckle, "knuckleball", "Knuckleball (No Spin)", "velocity", "velocity", "inferno", 0, 1.5, "Velocity (m/s)")

    print(f"\nExtracted {len(frame_indices) * 4} static PNGs to {output_dir}/ (frame {frame_indices[0]})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shot aerodynamics simulation & visualization")
    parser.add_argument("--extract", action="store_true", help="Extract static frames only (no MP4s)")
    args = parser.parse_args()

    obstacle_spin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=10.0)
    obstacle_nospin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=0.0)

    cache = Path(CACHE_FILE)
    if cache.exists():
        import pickle
        with open(cache, "rb") as f:
            data = pickle.load(f)
        frames_magnus = data["Magnus"]
        frames_knuckle = data["Knuckleball"]
        print(f"Loaded {len(frames_magnus)} Magnus + {len(frames_knuckle)} Knuckleball frames from cache")
    else:
        print("Magnus (ω = 10 rad/s)")
        frames_magnus = run_cylinder(obstacle_spin, "Magnus")

        print("\nKnuckleball (ω = 0 rad/s)")
        frames_knuckle = run_cylinder(obstacle_nospin, "Knuckleball")

        import pickle
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump({"Magnus": frames_magnus, "Knuckleball": frames_knuckle}, f)
        print(f"Cached frames to {cache}")

    print(f"\nFrames: {len(frames_magnus)} each, u shape: {frames_magnus[0]['u'].shape}")

    if args.extract:
        print("\n" + "=" * 60)
        print("STATIC FRAME EXTRACTION (t = 6-10s, every 5th frame)")
        print("=" * 60)
        PHIFLOW_IMAGES.mkdir(parents=True, exist_ok=True)
        extract_static_frames(frames_magnus, frames_knuckle, PHIFLOW_IMAGES)
    else:
        # Individual animations
        print("\n--- Individual Magnus animations ---")
        animate_magnus_pressure(frames_magnus)
        animate_magnus_velocity(frames_magnus)

        print("\n--- Individual Knuckleball animations ---")
        animate_knuckle_pressure(frames_knuckle)
        animate_knuckle_velocity(frames_knuckle)

        # Side-by-side comparisons
        print("\n--- Side-by-side comparisons ---")
        animate_velocity_comparison(frames_magnus, frames_knuckle)
        animate_pressure_comparison(frames_magnus, frames_knuckle)
