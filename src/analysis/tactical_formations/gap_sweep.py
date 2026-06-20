"""Gap distance sweep for inline drafting (2m, 3m, 4m).
Generates:
  - Drag table (printed + results_gap.json)
  - Static velocity+streamlines PNGs per gap at t~25s
  - Individual velocity+streamlines MP4s per gap
  - Stacked 3×1 comparison MP4 (velocity only, no streamlines)

Usage:
    uv run python src/tactical_formations/gap_sweep.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tqdm import trange
from pathlib import Path

from phi import geom
from phi.jax.flow import Obstacle, StaggeredGrid, CenteredGrid, fluid, advect, vec, field, ZERO_GRADIENT

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "tactical"
DRAFTING_DIR = IMAGES / "drafting"
DRAFTING_DIR.mkdir(parents=True, exist_ok=True)

BOX = geom.Box(x=8.0, y=4.0)
BOUNDARY = {"x-": vec(x=1.0, y=0.0), "x+": ZERO_GRADIENT, "y": 0}
NX, NY = 256, 128
_centered = CenteredGrid(0, x=NX, y=NY, bounds=BOX)

HW, HH = 0.15, 0.25

def runner_at(cx, cy):
    return geom.Box(x=(cx - HW, cx + HW), y=(cy - HH, cy + HH))

gap_cases = {
    2: [
        Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0)),
        Obstacle(runner_at(4.0, 2.0), velocity=vec(x=0.0, y=0.0)),
    ],
    3: [
        Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0)),
        Obstacle(runner_at(5.0, 2.0), velocity=vec(x=0.0, y=0.0)),
    ],
    4: [
        Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0)),
        Obstacle(runner_at(6.0, 2.0), velocity=vec(x=0.0, y=0.0)),
    ],
}

x_g = np.linspace(0, 8.0, NX)
y_g = np.linspace(0, 4.0, NY)
XX, YY = np.meshgrid(x_g, y_g, indexing="ij")

def face_drag_mask(cx, cy):
    up = (XX > cx - HW * 1.5) & (XX < cx - HW * 0.8) & (YY > cy - HH) & (YY < cy + HH)
    dn = (XX > cx + HW * 0.8) & (XX < cx + HW * 1.5) & (YY > cy - HH) & (YY < cy + HH)
    return up, dn

def box_drag(p_np, cx, cy):
    up, dn = face_drag_mask(cx, cy)
    return float(p_np[up].mean() - p_np[dn].mean()) if up.sum() and dn.sum() else 0.0

DT = 0.3
N_STEPS = 160
SAVE_EVERY = 2

def run_case(obstacles, label, drag_pos):
    v0 = StaggeredGrid((1.0, 0.0), BOUNDARY, x=NX, y=NY, bounds=BOX)
    v, _ = fluid.make_incompressible(v0, ())
    v = v0
    frames = []
    drag_series = []

    for i in trange(N_STEPS, desc=label):
        v = advect.semi_lagrangian(v, v, dt=DT)
        v, p = fluid.make_incompressible(v, obstacles)

        if i % SAVE_EVERY == 0:
            p_np = np.array(p.values.native("x", "y"))
            vel_mag = field.vec_length(v)
            vel_np = np.array(vel_mag.values.native("x", "y"))
            vc = v @ _centered
            u_c = np.array(vc.vector[0].values.native("x", "y"))
            v_c = np.array(vc.vector[1].values.native("x", "y"))
            frames.append({
                "velocity": vel_np, "u": u_c, "v": v_c, "pressure": p_np, "step": i,
            })
            drag_series.append(box_drag(p_np, *drag_pos))

    avg_drag = np.mean(drag_series[-len(drag_series)//5:]) if drag_series else 0.0
    return frames, avg_drag, drag_series

results = {}
frames_map = {}
for gap_m in sorted(gap_cases.keys()):
    obstacles = gap_cases[gap_m]
    trailer_x = 2.0 + gap_m
    label = f"Gap {gap_m}m"
    print(f"\n--- Gap {gap_m}m ---")
    frames, drag, _ = run_case(obstacles, label, (trailer_x, 2.0))
    frames_map[gap_m] = frames
    results[gap_m] = round(drag, 6)

print("\n" + "=" * 50)
print("Gap Sweep Drag Results")
print("=" * 50)
for gap_m in sorted(results):
    print(f"  {gap_m}m gap:  drag = {results[gap_m]:.6f}")

results_file = WORKDIR / "results_gap.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {results_file}")

n_frames = min(len(frames_map[g]) for g in frames_map)
for g in frames_map:
    while len(frames_map[g]) > n_frames:
        frames_map[g].pop()

gap_list = sorted(gap_cases.keys())
gap_labels = [f"{g}m Runner Gap" for g in gap_list]
gap_fnames = [f"gap_{g}m" for g in gap_list]
T25_IDX = 42
t25 = frames_map[gap_list[0]][T25_IDX]["step"] * DT

def _draw_runners(ax, runners):
    for cx, cy in runners:
        ax.add_patch(Rectangle((cx - HW, cy - HH), 2*HW, 2*HH,
                                facecolor="black", edgecolor="white", lw=1.5))

def _style_ax(ax, title, fontsize=13):
    ax.set_facecolor("#111111")
    ax.set_title(title, color="white", fontsize=fontsize)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])

def _cbar(fig, im, ax, label):
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

# ── 1. Static velocity+streamlines PNGs per gap ──
print(f"\n── 1. Static velocity+streamlines (t={t25:.1f}s) ──")
for gap_m, fms, glab, gfname in zip(gap_list,
    [frames_map[g] for g in gap_list], gap_labels, gap_fnames):
    runners = [(2.0, 2.0), (2.0 + gap_m, 2.0)]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{glab} - Velocity Field")
    _draw_runners(ax, runners)
    ax.streamplot(x_g, y_g, fms[T25_IDX]["u"].T, fms[T25_IDX]["v"].T,
                  color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
    im = ax.imshow(fms[T25_IDX]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    _cbar(fig, im, ax, "Velocity magnitude (m/s)")
    fig.tight_layout()
    path = DRAFTING_DIR / f"drafting_{gfname}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved {path}")
    plt.close(fig)

# ── 2. Individual velocity+streamlines MP4s ──
print("\n── 2. Individual velocity+streamlines videos ──")
for gap_m, fms, glab, gfname in zip(gap_list,
    [frames_map[g] for g in gap_list], gap_labels, gap_fnames):
    runners_list = [(2.0, 2.0), (2.0 + gap_m, 2.0)]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{glab} - Velocity Field")
    _draw_runners(ax, runners_list)
    sp = ax.streamplot(x_g, y_g, fms[0]["u"].T, fms[0]["v"].T,
                       color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
    im = ax.imshow(fms[0]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    _cbar(fig, im, ax, "Velocity magnitude (m/s)")
    fig.tight_layout()

    def _upd(idx, ax=ax, fms=fms, im=im, _runners=runners_list):
        for coll in list(ax.collections):
            coll.remove()
        for p in list(ax.patches):
            p.remove()
        for cx, cy in _runners:
            ax.add_patch(Rectangle((cx - HW, cy - HH), 2*HW, 2*HH,
                                    facecolor="black", edgecolor="white", lw=1.5))
        ax.streamplot(x_g, y_g, fms[idx]["u"].T, fms[idx]["v"].T,
                      color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
        im.set_data(fms[idx]["velocity"].T)

    anim = FuncAnimation(fig, _upd, frames=n_frames, interval=80, blit=False)
    path = DRAFTING_DIR / f"drafting_{gfname}.mp4"
    anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

# ── 3. Stacked comparison video (3 panels, velocity only) ──
print("\n── 3. Stacked comparison video (velocity only, no streamlines) ──")
fig, axes = plt.subplots(3, 1, figsize=(8, 10), facecolor="#111111")
fig.suptitle("Inline Runner Distance Comparison", color="white", fontsize=14, y=0.97)

comp_labels = [f"{g}m Gap" for g in gap_list]
imgs = []
for ax, glab, gap_m in zip(axes, comp_labels, gap_list):
    fms = frames_map[gap_m]
    runners_list = [(2.0, 2.0), (2.0 + gap_m, 2.0)]
    _style_ax(ax, glab, fontsize=11)
    _draw_runners(ax, runners_list)
    im = ax.imshow(fms[0]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    imgs.append(im)

fig.subplots_adjust(left=0.04, right=0.88, top=0.93, bottom=0.04, hspace=0.08)
cbar_ax = fig.add_axes([0.90, 0.04, 0.025, 0.89])
cbar = fig.colorbar(imgs[-1], cax=cbar_ax)
cbar.set_label("Velocity magnitude (m/s)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

def update_comp(idx):
    for im, gap_m in zip(imgs, gap_list):
        fms = frames_map[gap_m]
        im.set_data(fms[idx]["velocity"].T)
    fig.suptitle(f"Inline Runner Distance Comparison - t = {frames_map[gap_list[0]][idx]['step'] * DT:.1f}s",
                 color="white", fontsize=14, y=0.97)
    return imgs

anim = FuncAnimation(fig, update_comp, frames=n_frames, interval=80, blit=True)
path = DRAFTING_DIR / "drafting_gap_comparison.mp4"
anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
print(f"  Saved {path}")
plt.close(fig)

print("\n── All outputs complete ──")
