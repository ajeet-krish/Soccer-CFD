"""Tactical Formation Pressure Field Visualizations.

Generates static pressure field plots, lane analysis overlays,
and transition animations for soccer formations using the
Gaussian influence field method.

Colormap: turbo (blue = open space, yellow/orange/red = congested).
Lane analysis divides the pitch into 5 vertical lanes and scores
each by average defensive pressure relative to global maximum.

Transitions animate the same formation from attacking shape (spread)
to defending shape (compact), showing space closing as the team
gets into defensive structure.

Usage:
    uv run python src/tactical_formations/generate_formation_pressure.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Arc, Rectangle

matplotlib.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9,
    "pdf.fonttype": 42,
})

# ── Paths ──
SCRIPT_DIR = Path(__file__).parent
OUTPUT = SCRIPT_DIR.parent.parent / "docs" / "images" / "tactical"
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "formation_pressure").mkdir(exist_ok=True)
(OUTPUT / "formation_lanes").mkdir(exist_ok=True)
(OUTPUT / "formation_transition").mkdir(exist_ok=True)

# ── Pitch geometry (meters, left-to-right attack) ──
XLIM = (0, 105)
YLIM = (-34, 34)
PITCH_LENGTH = 105.0
HALF_W = 34.0

NX_PRESS = 800
NY_PRESS = 400

N_TRANSITION_FRAMES = 60
TRANSITION_FPS = 15

# Player sigma values by role (larger = wider defensive influence)
SIGMA = {
    "GK": 3.0,
    "CB": 4.0,
    "LB": 3.5,
    "RB": 3.5,
    "CDM": 3.0,
    "CM": 3.0,
    "CAM": 3.0,
    "LM": 3.0,
    "RM": 3.0,
    "LW": 2.5,
    "RW": 2.5,
    "ST": 2.5,
}

# ── Formation definitions ──
# Each formation has a defending (compact) and attacking (spread) shape.
# Attack direction: left to right.

FORMATIONS = {
    "433": {
        "label": "4-3-3",
        "defending": [
            (5.0, 0.0, "GK"),
            (18.0, -9.0, "CB"),
            (18.0, 9.0, "CB"),
            (18.0, -24.0, "LB"),
            (18.0, 24.0, "RB"),
            (30.0, 0.0, "CDM"),
            (30.0, -10.0, "CM"),
            (30.0, 10.0, "CM"),
            (42.0, -20.0, "LW"),
            (42.0, 20.0, "RW"),
            (50.0, 0.0, "ST"),
        ],
        "attacking": [
            (5.0, 0.0, "GK"),
            (35.0, -12.0, "CB"),
            (35.0, 12.0, "CB"),
            (38.0, -32.0, "LB"),
            (38.0, 32.0, "RB"),
            (50.0, 0.0, "CDM"),
            (52.0, -15.0, "CM"),
            (52.0, 15.0, "CM"),
            (72.0, -28.0, "LW"),
            (72.0, 28.0, "RW"),
            (80.0, 0.0, "ST"),
        ],
    },
    "442": {
        "label": "4-4-2",
        "defending": [
            (5.0, 0.0, "GK"),
            (18.0, -9.0, "CB"),
            (18.0, 9.0, "CB"),
            (18.0, -24.0, "LB"),
            (18.0, 24.0, "RB"),
            (30.0, -10.0, "CM"),
            (30.0, 10.0, "CM"),
            (30.0, -22.0, "LM"),
            (30.0, 22.0, "RM"),
            (48.0, -8.0, "ST"),
            (48.0, 8.0, "ST"),
        ],
        "attacking": [
            (5.0, 0.0, "GK"),
            (35.0, -12.0, "CB"),
            (35.0, 12.0, "CB"),
            (38.0, -32.0, "LB"),
            (38.0, 32.0, "RB"),
            (54.0, -12.0, "CM"),
            (54.0, 12.0, "CM"),
            (54.0, -28.0, "LM"),
            (54.0, 28.0, "RM"),
            (75.0, -9.0, "ST"),
            (75.0, 9.0, "ST"),
        ],
    },
    "4231": {
        "label": "4-2-3-1",
        "defending": [
            (5.0, 0.0, "GK"),
            (18.0, -9.0, "CB"),
            (18.0, 9.0, "CB"),
            (18.0, -24.0, "LB"),
            (18.0, 24.0, "RB"),
            (30.0, -7.0, "CDM"),
            (30.0, 7.0, "CDM"),
            (38.0, 0.0, "CAM"),
            (38.0, -20.0, "LW"),
            (38.0, 20.0, "RW"),
            (52.0, 0.0, "ST"),
        ],
        "attacking": [
            (5.0, 0.0, "GK"),
            (35.0, -12.0, "CB"),
            (35.0, 12.0, "CB"),
            (38.0, -32.0, "LB"),
            (38.0, 32.0, "RB"),
            (52.0, -9.0, "CDM"),
            (52.0, 9.0, "CDM"),
            (62.0, 0.0, "CAM"),
            (72.0, -26.0, "LW"),
            (72.0, 26.0, "RW"),
            (82.0, 0.0, "ST"),
        ],
    },
    "352": {
        "label": "3-5-2",
        "defending": [
            (5.0, 0.0, "GK"),
            (18.0, -14.0, "CB"),
            (18.0, 0.0, "CB"),
            (18.0, 14.0, "CB"),
            (28.0, -28.0, "LM"),
            (28.0, 28.0, "RM"),
            (30.0, -7.0, "CDM"),
            (30.0, 7.0, "CDM"),
            (40.0, 0.0, "CAM"),
            (50.0, -9.0, "ST"),
            (50.0, 9.0, "ST"),
        ],
        "attacking": [
            (5.0, 0.0, "GK"),
            (32.0, -18.0, "CB"),
            (32.0, 0.0, "CB"),
            (32.0, 18.0, "CB"),
            (58.0, -32.0, "LM"),
            (58.0, 32.0, "RM"),
            (50.0, -10.0, "CDM"),
            (50.0, 10.0, "CDM"),
            (65.0, 0.0, "CAM"),
            (80.0, -10.0, "ST"),
            (80.0, 10.0, "ST"),
        ],
    },
}

# Lanes: y-boundaries for 5 vertical lanes (attacking left to right)
LANE_BOUNDS = [-34, -20.15, -6, 6, 20.15, 34]
LANE_NAMES = ["Left Wing", "Left Half-Space", "Center", "Right Half-Space", "Right Wing"]

# Truncated twilight: avoids cyclic wrap (purple→pink→gold→teal, never wraps back to purple)
CMAP = "turbo"


# ── Utility functions ──

def draw_pitch(ax, color="white", alpha=0.4, linewidth=0.8):
    ax.plot([XLIM[0], XLIM[1], XLIM[1], XLIM[0], XLIM[0]],
            [YLIM[0], YLIM[0], YLIM[1], YLIM[1], YLIM[0]],
            color=color, alpha=alpha, linewidth=linewidth)
    ax.axvline(PITCH_LENGTH / 2, color=color, alpha=alpha, linewidth=linewidth)
    ax.add_patch(Arc((PITCH_LENGTH / 2, 0), 18.3, 18.3, angle=0,
                     theta1=0, theta2=360, color=color, alpha=alpha, linewidth=linewidth))
    for x0 in [0, PITCH_LENGTH - 16.5]:
        ax.add_patch(Rectangle((x0, -20.15), 16.5, 40.3,
                               fill=False, color=color, alpha=alpha, linewidth=linewidth))
    for x0 in [0, PITCH_LENGTH - 5.5]:
        ax.add_patch(Rectangle((x0, -9.16), 5.5, 18.32,
                               fill=False, color=color, alpha=alpha, linewidth=linewidth))


def draw_players(ax, formation, color="black", zorder=6):
    for x, y, role in formation:
        ax.add_patch(Circle((x, y), 1.2, fill=True, color=color, alpha=0.85, zorder=zorder))


def compute_field(formation, nx=800, ny=400):
    xs = np.linspace(XLIM[0], XLIM[1], nx)
    ys = np.linspace(YLIM[0], YLIM[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Phi = np.zeros_like(X)
    for x, y, role in formation:
        s = SIGMA.get(role, 3.0)
        Phi += np.exp(-((X - x)**2 + (Y - y)**2) / (2 * s**2))
    return X, Y, Phi


def compute_global_max():
    """Compute global maximum density and max lane avg across all defending formations."""
    gmax = 0.0
    all_lane_avgs = []
    xs = np.linspace(XLIM[0], XLIM[1], NX_PRESS)
    ys = np.linspace(YLIM[0], YLIM[1], NY_PRESS)
    X, Y = np.meshgrid(xs, ys)
    for name, phases in FORMATIONS.items():
        Phi = np.zeros_like(X)
        for x, y, role in phases["defending"]:
            s = SIGMA.get(role, 3.0)
            Phi += np.exp(-((X - x)**2 + (Y - y)**2) / (2 * s**2))
        gmax = max(gmax, Phi.max())
        for i in range(5):
            mask = (Y >= LANE_BOUNDS[i]) & (Y < LANE_BOUNDS[i + 1])
            all_lane_avgs.append(Phi[mask].mean())
    lane_max = max(all_lane_avgs)
    return gmax, lane_max


# ── Plot sizing ──
PITCH_ASPECT = (XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0])  # 105/68 ≈ 1.544


def make_figsize(height=5.0, colorbar=True):
    cbar_margin = 0.5 if colorbar else 0.1
    return (height * PITCH_ASPECT + cbar_margin, height)


def setup_figure(title="", height=5.0, colorbar=True):
    fig, ax = plt.subplots(figsize=make_figsize(height, colorbar))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Thin white spines with inward ticks
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_color("white")
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(axis="both", colors="white", width=0.5, length=3, direction="in")

    if title:
        fig.subplots_adjust(top=0.92)
        fig.suptitle(title, fontsize=13, color="white", y=0.98)
    return fig, ax


def add_colorbar(fig, ax, cntr, label="Defensive Pressure"):
    cbar = fig.colorbar(cntr, ax=ax, label=label, shrink=0.85, fraction=0.12, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", width=0.5, length=2.5, direction="in")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=8)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(label, color="white", fontsize=9, labelpad=8)
    return cbar


# ═══════════════════════════════════════════════════════════════
# 1. STATIC PRESSURE FIELD (defending shape, RdBu_r)
# ═══════════════════════════════════════════════════════════════

def plot_pressure(formation, name, vmax):
    print(f"  Pressure field: {name}")
    X, Y, Phi = compute_field(formation, NX_PRESS, NY_PRESS)
    fig, ax = setup_figure(title=f"{name} Formation Pressure Field", height=5.0, colorbar=True)
    cntr = ax.contourf(X, Y, Phi, levels=40, cmap=CMAP, vmin=0, vmax=vmax, extend="both")
    draw_pitch(ax)
    draw_players(ax, formation)
    add_colorbar(fig, ax, cntr)
    path = OUTPUT / "formation_pressure" / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor="#111111", bbox_inches="tight")
    print(f"    Saved: {path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 2. LANE ANALYSIS (RdBu_r + dashed lane dividers + pressure %)
# ═══════════════════════════════════════════════════════════════

def plot_lanes(formation, name, vmax, lane_max):
    print(f"  Lane analysis: {name}")
    X, Y, Phi = compute_field(formation, NX_PRESS, NY_PRESS)
    fig, ax = setup_figure(title=f"{name} Formation Pressure Field", height=5.0, colorbar=True)

    cntr = ax.contourf(X, Y, Phi, levels=40, cmap=CMAP, vmin=0, vmax=vmax, extend="both")

    # Draw dashed lane dividers across the full pitch
    for y in LANE_BOUNDS[1:-1]:
        ax.plot([XLIM[0], XLIM[1]], [y, y], color="white", linestyle="--",
                alpha=0.5, linewidth=0.9, zorder=4)

    # Compute per-lane average pressure as % of max lane avg across all formations
    lane_pcts = []
    for i in range(5):
        y_lo = LANE_BOUNDS[i]
        y_hi = LANE_BOUNDS[i + 1]
        mask = (Y >= y_lo) & (Y < y_hi)
        lane_avg = Phi[mask].mean()
        lane_pcts.append(lane_avg / lane_max * 100)

    # "Lane Pressure %" label above the annotations
    x_annot = 96.0
    ax.text(x_annot, YLIM[1] + 2.5, "Lane\nPressure %",
            ha="center", va="bottom", fontsize=8, color="white",
            alpha=0.8, zorder=5)

    # Annotate lane pressure % in the right penalty area
    for i in range(5):
        y_mid = (LANE_BOUNDS[i] + LANE_BOUNDS[i + 1]) / 2
        pct = lane_pcts[i]
        if pct > 60:
            txt_color = "#ffcccc"
        elif pct < 35:
            txt_color = "#ccccff"
        else:
            txt_color = "white"
        ax.text(x_annot, y_mid, f"{pct:.0f}%",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color=txt_color, alpha=0.9, zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#111111")])

    draw_pitch(ax)
    draw_players(ax, formation)
    add_colorbar(fig, ax, cntr)
    path = OUTPUT / "formation_lanes" / f"{name}.png"
    fig.savefig(path, dpi=200, facecolor="#111111", bbox_inches="tight")
    print(f"    Saved: {path.name}")
    plt.close()

    return lane_pcts


# ═══════════════════════════════════════════════════════════════
# 3. COMPARISON GRID (defending shapes)
# ═══════════════════════════════════════════════════════════════

def plot_comparison(vmax, lane_max):
    print("  Comparison grid (2x2)")
    items = [(name, phases["defending"]) for name, phases in FORMATIONS.items()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#111111")
    cntr = None
    for idx, (name, formation) in enumerate(items):
        ax = axes[idx // 2, idx % 2]
        X, Y, Phi = compute_field(formation, 400, 300)
        ax.set_facecolor("#111111")
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Spines
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_color("white")
            ax.spines[s].set_linewidth(0.4)
        ax.tick_params(axis="both", colors="white", width=0.4, length=2, direction="in")

        cntr = ax.contourf(X, Y, Phi, levels=40, cmap=CMAP, vmin=0, vmax=vmax, extend="both")
        draw_pitch(ax)
        draw_players(ax, formation)
        ax.set_title(name, fontsize=14, color="white", pad=8)

        # Lane dividers
        for y in LANE_BOUNDS[1:-1]:
            ax.plot([XLIM[0], XLIM[1]], [y, y], color="white", linestyle="--",
                    alpha=0.4, linewidth=0.7, zorder=4)

        # Lane % label
        ax.text(95.0, YLIM[1] + 2.0, "Lane\n%",
                ha="center", va="bottom", fontsize=6.5, color="white",
                alpha=0.8, zorder=5)

        # Lane annotations in right penalty area
        x_annot = 95.0
        for i in range(5):
            y_mid = (LANE_BOUNDS[i] + LANE_BOUNDS[i + 1]) / 2
            y_lo = LANE_BOUNDS[i]
            y_hi = LANE_BOUNDS[i + 1]
            mask = (Y >= y_lo) & (Y < y_hi)
            lane_avg = Phi[mask].mean()
            pct = lane_avg / lane_max * 100
            if pct > 60:
                txt_color = "#ffcccc"
            elif pct < 35:
                txt_color = "#ccccff"
            else:
                txt_color = "white"
            ax.text(x_annot, y_mid, f"{pct:.0f}%",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color=txt_color, alpha=0.85, zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="#111111")])

    fig.subplots_adjust(right=0.92, wspace=0.05, hspace=0.12, top=0.93)
    fig.suptitle("Defensive Pressure Field", fontsize=16, color="white", y=0.98)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.012, 0.8])
    cbar = fig.colorbar(cntr, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color="white", width=0.5, length=2.5, direction="in")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=8)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Defensive Pressure", color="white", fontsize=9, labelpad=8)
    path = OUTPUT / "formation_comparison.png"
    fig.savefig(path, dpi=200, facecolor="#111111", bbox_inches="tight")
    print(f"    Saved: {path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 5. TRANSITION ANIMATIONS (attacking → defending)
# ═══════════════════════════════════════════════════════════════

def animate_transition(name, vmax):
    out_name = f"attacking_to_defending_{name}"
    print(f"  Transition: {out_name}")
    att = FORMATIONS[name]["attacking"]
    deff = FORMATIONS[name]["defending"]
    n_players = len(att)
    xs = np.linspace(XLIM[0], XLIM[1], NX_PRESS)
    ys = np.linspace(YLIM[0], YLIM[1], NY_PRESS)
    Xg, Yg = np.meshgrid(xs, ys)
    fig, ax = plt.subplots(figsize=make_figsize(5.0, colorbar=True))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Spines
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_color("white")
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(axis="both", colors="white", width=0.5, length=3, direction="in")

    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"{name} Formation Pressure Field", fontsize=13, color="white", y=0.98)
    draw_pitch(ax)
    cbar = fig.colorbar(ax.contourf(Xg, Yg, np.zeros_like(Xg), levels=40, cmap=CMAP, vmin=0, vmax=vmax, extend="both"),
                        ax=ax, shrink=0.85, fraction=0.12, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", width=0.5, length=2.5, direction="in")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=8)
    cbar.outline.set_edgecolor("white")
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Defensive Pressure", color="white", fontsize=9, labelpad=8)
    out_path = OUTPUT / "formation_transition" / f"{out_name}.mp4"
    writer = FFMpegWriter(fps=TRANSITION_FPS, bitrate=3000)
    with writer.saving(fig, str(out_path), dpi=120):
        for t in range(N_TRANSITION_FRAMES):
            frac = t / (N_TRANSITION_FRAMES - 1)
            Phi = np.zeros_like(Xg)
            for i in range(n_players):
                x_a, y_a, r_a = att[i]
                x_b, y_b, r_b = deff[i]
                x = x_a + (x_b - x_a) * frac
                y = y_a + (y_b - y_a) * frac
                role = r_a if frac < 0.5 else r_b
                s = SIGMA.get(role, 3.0)
                Phi += np.exp(-((Xg - x)**2 + (Yg - y)**2) / (2 * s**2))
            for c in ax.collections[:]:
                if isinstance(c, matplotlib.contour.QuadContourSet):
                    c.remove()
            ax.contourf(Xg, Yg, Phi, levels=40, cmap=CMAP, vmin=0, vmax=vmax, extend="both")
            writer.grab_frame()
    print(f"    Saved: {out_path.name}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  GENERATING TACTICAL FORMATION VISUALIZATIONS")
    print("=" * 60)

    vmax, lane_max = compute_global_max()
    print(f"\n  Global max defensive pressure: {vmax:.4f}")
    print(f"  Global max lane average: {lane_max:.4f}")

    for name, phases in FORMATIONS.items():
        label = phases["label"]
        print(f"\n--- {label} (defending) ---")
        plot_pressure(phases["defending"], name, vmax)
        lane_pcts = plot_lanes(phases["defending"], name, vmax, lane_max)
        print(f"    Lane pressures: {[f'{p:.0f}%' for p in lane_pcts]}")

    print("\n--- Comparison ---")
    plot_comparison(vmax, lane_max)

    print("\n--- Transitions (attacking → defending) ---")
    for name in FORMATIONS:
        animate_transition(name, vmax)

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
