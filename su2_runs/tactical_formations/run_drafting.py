"""ΦFlow parametric sweep of tandem bluff-body drafting for football overlaps.
Inline (directly behind) and staggered (shoulder-width offset) configurations
at gap distances from 0.25 to 4.0 body-lengths. Measures drag reduction vs solo runner.

Optimised for CPU execution: reduced grid (124×93) and steps (100) → ~6 min total.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phi import geom
from phi.jax.flow import Obstacle, StaggeredGrid, CenteredGrid, fluid, advect, vec, field, ZERO_GRADIENT
import numpy as np
import json
from tqdm import trange
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "tactical"
DRAFTING_DIR = IMAGES / "drafting"
DRAFTING_DIR.mkdir(parents=True, exist_ok=True)

# ── Domain (half-resolution) ──
BOX = geom.Box(x=8.0, y=6.0)
BOUNDARY = {
    "x-": vec(x=10.0, y=0.0),
    "x+": ZERO_GRADIENT,
    "y": 0,
}
NX, NY = 124, 93
CENTERED = CenteredGrid(0, x=NX, y=NY, bounds=BOX)

# ── Player shape ──
HW, HH = 0.15, 0.25
BODY_LENGTH = 0.3

# ── Sweep parameters (reduced for fast turnaround) ──
GAPS = [0.5, 0.75, 1.0, 1.5, 3.0]
LEAD_X, LEAD_Y = 2.0, 3.0
STAGGER_Y = 0.35

DT = 0.03
N_STEPS = 100
SAVE_EVERY = 2
INFLOW_VEL = 10.0

# ── Drag masks ──
x_g = np.linspace(0, 8.0, NX)
y_g = np.linspace(0, 6.0, NY)
XX, YY = np.meshgrid(x_g, y_g, indexing="ij")

def player_at(cx, cy):
    return geom.Box(x=(cx - HW, cx + HW), y=(cy - HH, cy + HH))

def face_drag_mask(cx, cy):
    front = (XX > cx - HW * 1.5) & (XX < cx - HW * 0.8) & (YY > cy - HH) & (YY < cy + HH)
    rear  = (XX > cx + HW * 0.8) & (XX < cx + HW * 1.5) & (YY > cy - HH) & (YY < cy + HH)
    return front, rear

def box_drag(p_np, cx, cy):
    front, rear = face_drag_mask(cx, cy)
    return float(p_np[front].mean() - p_np[rear].mean()) if front.sum() and rear.sum() else 0.0

def run_case(obstacles, label, drag_pos=None, save_frames=False):
    v = StaggeredGrid((INFLOW_VEL, 0.0), BOUNDARY, x=NX, y=NY, bounds=BOX)
    v, _ = fluid.make_incompressible(v, ())
    drag_series = []
    frames = []

    for i in trange(N_STEPS, desc=label):
        v = advect.semi_lagrangian(v, v, dt=DT)
        v, p = fluid.make_incompressible(v, obstacles)

        if drag_pos and i % SAVE_EVERY == 0:
            p_np = np.array(p.values.native("x,y"))
            drag_series.append(box_drag(p_np, *drag_pos))

        if save_frames and i == N_STEPS - 1:
            vc = v @ CENTERED
            p_np = np.array(p.values.native("x,y"))
            vel_mag = field.vec_length(v)
            vel_np = np.array(vel_mag.values.native("x,y"))
            frames.append({"pressure": p_np, "velocity": vel_np,
                           "u": np.array(vc.vector[0].values.native("x,y")),
                           "v": np.array(vc.vector[1].values.native("x,y"))})

    avg_drag = np.mean(drag_series[-len(drag_series) // 5:]) if drag_series else 0.0
    return avg_drag, drag_series, frames

# ── Baseline: solo runner ──
print("=== Baseline: Solo Runner ===")
drag_solo, _, solo_frames = run_case(
    [Obstacle(player_at(LEAD_X, LEAD_Y), velocity=vec(x=0.0, y=0.0))],
    "Solo runner", (LEAD_X, LEAD_Y), save_frames=True
)
print(f"  Baseline drag: {drag_solo:.4f}")

# ── Sweep ──
results = {"baseline_drag": drag_solo, "inline": {}, "staggered": {}}
opt_inline = {"gap": None, "drag": None, "frames": None}
opt_staggered = {"gap": None, "drag": None, "frames": None}

for label, y_offset, store in [("Inline", 0.0, "inline"), ("Staggered", STAGGER_Y, "staggered")]:
    print(f"\n=== {label} Sweep ===")
    for gap in GAPS:
        trail_x = LEAD_X + gap * BODY_LENGTH
        trail_y = LEAD_Y + y_offset
        obstacles = [
            Obstacle(player_at(LEAD_X, LEAD_Y), velocity=vec(x=0.0, y=0.0)),
            Obstacle(player_at(trail_x, trail_y), velocity=vec(x=0.0, y=0.0)),
        ]
        save = (gap == 1.0)
        drag_trail, _, frames = run_case(
            obstacles, f"{label} gap={gap:.2f}BL", (trail_x, trail_y), save_frames=save
        )
        reduction = (1 - drag_trail / drag_solo) * 100 if drag_solo != 0 else 0.0
        results[store][f"gap_{gap:.2f}"] = {
            "gap_body_lengths": gap,
            "trail_x": trail_x,
            "trail_y": trail_y,
            "drag": drag_trail,
            "reduction_pct": reduction,
        }
        print(f"  gap={gap:.2f} BL: drag={drag_trail:.4f}  reduction={reduction:.1f}%")

        if save:
            target = opt_inline if label == "Inline" else opt_staggered
            target.update({"gap": gap, "drag": drag_trail, "reduction": reduction, "frames": frames})

# ── Save results ──
results_path = WORKDIR / "results_drafting.json"
results_path.write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults saved to {results_path}")

# ── Save frames for viz ──
np.savez(DRAFTING_DIR / "frames_solo.npz",
         pressure=solo_frames[0]["pressure"], velocity=solo_frames[0]["velocity"],
         u=solo_frames[0]["u"], v=solo_frames[0]["v"])
if opt_inline["frames"]:
    np.savez(DRAFTING_DIR / "frames_inline.npz",
             pressure=opt_inline["frames"][0]["pressure"], velocity=opt_inline["frames"][0]["velocity"],
             u=opt_inline["frames"][0]["u"], v=opt_inline["frames"][0]["v"])
if opt_staggered["frames"]:
    np.savez(DRAFTING_DIR / "frames_staggered.npz",
             pressure=opt_staggered["frames"][0]["pressure"], velocity=opt_staggered["frames"][0]["velocity"],
             u=opt_staggered["frames"][0]["u"], v=opt_staggered["frames"][0]["v"])

print("\nDone. Flow field frames saved for visualization.")
