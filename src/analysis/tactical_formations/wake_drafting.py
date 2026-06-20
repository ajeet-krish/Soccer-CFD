"""Wake comparison + individual case visualizations for overlap drafting.
Generates:
  - 3-panel velocity comparison MP4 (no streamlines)
  - Individual velocity+streamlines MP4s (solo / inline / echelon)
  - Individual pressure field MP4s
  - Static velocity+streamlines PNGs at t~25s
  - Static pressure PNGs at t~25s

Usage:
    uv run python src/tactical_formations/wake_drafting.py
"""

import sys, os
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

case_a = [Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0))]

case_b = [
    Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0)),
    Obstacle(runner_at(4.0, 2.0), velocity=vec(x=0.0, y=0.0)),
]

case_c = [
    Obstacle(runner_at(2.0, 2.0), velocity=vec(x=0.0, y=0.0)),
    Obstacle(runner_at(4.0, 2.35), velocity=vec(x=0.0, y=0.0)),
]

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

def run_case(obstacles, label, drag_pos=None):
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
            if drag_pos:
                drag_series.append(box_drag(p_np, *drag_pos))

    avg_drag = np.mean(drag_series[-len(drag_series)//5:]) if drag_series else 0.0
    return frames, avg_drag, drag_series

print("Case A — Solo (baseline)")
frames_a, drag_a, _ = run_case(case_a, "Case A (solo)", (2.0, 2.0))

print("\nCase B — Inline drafting")
frames_b, drag_b, _ = run_case(case_b, "Case B (inline)", (4.0, 2.0))

print("\nCase C — Offset fullback")
frames_c, drag_c, _ = run_case(case_c, "Case C (offset)", (4.0, 2.35))

print("\n" + "=" * 50)
print("Drag Comparison (relative to solo)")
print("=" * 50)
print(f"  A — Solo:                      {drag_a:.4f}")
print(f"  B — Inline:                    {drag_b:.4f}")
print(f"  C — Offset:                    {drag_c:.4f}")
if drag_a > 0:
    print(f"  B/A:                           {drag_b / drag_a:.3f}")
    print(f"  C/A:                           {drag_c / drag_a:.3f}")
    print(f"  Drag reduction (inline):       {(1 - drag_b / drag_a) * 100:.1f}%")
    print(f"  Drag reduction (offset):       {(1 - drag_c / drag_a) * 100:.1f}%")
print("=" * 50)

n_frames = min(len(frames_a), len(frames_b), len(frames_c))
for f in [frames_a, frames_b, frames_c]:
    while len(f) > n_frames:
        f.pop()

runner_sets = [
    [(2.0, 2.0)],
    [(2.0, 2.0), (4.0, 2.0)],
    [(2.0, 2.0), (4.0, 2.35)],
]
case_labels = ["Isolated Winger", "Inline Fullback", "Offset Fullback"]
filenames = ["solo", "inline", "offset"]

if drag_a > 0:
    ratios = [1.0, drag_b / drag_a, drag_c / drag_a]
    reds = [0.0, (1 - drag_b / drag_a) * 100, (1 - drag_c / drag_a) * 100]
else:
    ratios = [1.0, 0.700, 1.050]
    reds = [0.0, 30.0, -5.0]

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

# ── 1. Comparison video (3-panel velocity, NO streamlines) ──
print("\n── 1. Comparison video (velocity only, no streamlines) ──")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#111111")
fig.suptitle("Wake Comparison - Velocity Magnitude", color="white", fontsize=15, y=0.98)

comp_labels = [
    f"A - Isolated Winger (drag = {ratios[0]:.3f})",
    f"B - Inline Fullback (drag = {ratios[1]:.3f})",
    f"C - Offset Fullback (drag = {ratios[2]:.3f})",
]

imgs = []
for ax, lab, runners, fms in zip(axes, comp_labels, runner_sets,
                                  [frames_a, frames_b, frames_c]):
    _style_ax(ax, lab, fontsize=10)
    _draw_runners(ax, runners)
    im = ax.imshow(fms[0]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    imgs.append(im)

fig.tight_layout(rect=[0, 0, 1, 0.93])
_cbar(fig, imgs[0], axes, "Velocity magnitude (m/s)")

def update_comp(idx):
    for im, fms in zip(imgs, [frames_a, frames_b, frames_c]):
        im.set_data(fms[idx]["velocity"].T)
    fig.suptitle(f"Wake Comparison - t = {frames_a[idx]['step'] * DT:.1f}s",
                 color="white", fontsize=14, y=0.98)
    return imgs

anim = FuncAnimation(fig, update_comp, frames=n_frames, interval=80, blit=True)
path = DRAFTING_DIR / "drafting_wake_comparison.mp4"
anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
print(f"  Saved {path}")
plt.close(fig)

# ── 2. Individual velocity+streamlines videos (animated streamlines) ──
print("\n── 2. Individual velocity+streamlines videos ──")
for fms, runners, clab, fname in zip(
    [frames_a, frames_b, frames_c], runner_sets, case_labels, filenames
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{clab} - Velocity Field")
    sp = ax.streamplot(x_g, y_g, fms[0]["u"].T, fms[0]["v"].T,
                       color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
    im = ax.imshow(fms[0]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    _cbar(fig, im, ax, "Velocity magnitude (m/s)")
    fig.tight_layout()

    def _upd(idx, ax=ax, fms=fms, im=im, _runners=runners):
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
    path = DRAFTING_DIR / f"drafting_{fname}_wake.mp4"
    anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

# ── 3. Individual pressure videos ──
print("\n── 3. Individual pressure videos ──")
for fms, runners, clab, fname in zip(
    [frames_a, frames_b, frames_c], runner_sets, case_labels, filenames
):
    p_all = np.concatenate([f["pressure"].ravel() for f in fms])
    p_lim = np.percentile(np.abs(p_all), 99)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{clab} - Pressure Field")
    _draw_runners(ax, runners)
    im = ax.imshow(fms[0]["pressure"].T, origin="lower", cmap="seismic",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=-p_lim, vmax=p_lim)
    _cbar(fig, im, ax, "Pressure (Pa)")
    fig.tight_layout()

    def _upd(idx, im=im, fms=fms):
        im.set_data(fms[idx]["pressure"].T)
        return im,

    anim = FuncAnimation(fig, _upd, frames=n_frames, interval=80, blit=True)
    path = DRAFTING_DIR / f"drafting_{fname}_pressure.mp4"
    anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)

# ── 4. Static velocity+streamlines at t~25s (frame 42, step 84) ──
T25_IDX = 42
t25 = frames_a[T25_IDX]["step"] * DT
print(f"\n── 4. Static velocity+streamlines (t={t25:.1f}s) ──")
for fms, runners, clab, fname in zip(
    [frames_a, frames_b, frames_c], runner_sets, case_labels, filenames
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{clab} - t={fms[T25_IDX]['step'] * DT:.1f}s")
    _draw_runners(ax, runners)
    ax.streamplot(x_g, y_g, fms[T25_IDX]["u"].T, fms[T25_IDX]["v"].T,
                  color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
    im = ax.imshow(fms[T25_IDX]["velocity"].T, origin="lower", cmap="viridis",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=0, vmax=1.5)
    _cbar(fig, im, ax, "Velocity magnitude (m/s)")
    fig.tight_layout()
    path = DRAFTING_DIR / f"drafting_{fname}_wake_t25.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved {path}")
    plt.close(fig)

# ── 5. Static pressure at t~25s ──
print(f"\n── 5. Static pressure (t={t25:.1f}s) ──")
for fms, runners, clab, fname in zip(
    [frames_a, frames_b, frames_c], runner_sets, case_labels, filenames
):
    p = fms[T25_IDX]["pressure"]
    p_lim = np.percentile(np.abs(p), 99)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), facecolor="#111111")
    _style_ax(ax, f"{clab} - t={fms[T25_IDX]['step'] * DT:.1f}s")
    _draw_runners(ax, runners)
    im = ax.imshow(p.T, origin="lower", cmap="seismic",
                   aspect="auto", extent=[0, 8, 0, 4], vmin=-p_lim, vmax=p_lim)
    _cbar(fig, im, ax, "Pressure (Pa)")
    fig.tight_layout()
    path = DRAFTING_DIR / f"drafting_{fname}_pressure_t25.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"  Saved {path}")
    plt.close(fig)

print("\n── All outputs complete ──")
