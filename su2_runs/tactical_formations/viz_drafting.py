"""Generate drafting visualizations: pressure field comparisons only.
The distance curve and bar chart are dropped due to coarse-grid artifacts
at this resolution. Pressure fields show the qualitative wake-shielding effect.

Usage:
    uv run python su2_runs/tactical_formations/viz_drafting.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "pdf.fonttype": 42,
})

WORKDIR = Path(__file__).parent
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "tactical"
DRAFTING_DIR = IMAGES / "drafting"
DRAFTING_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ──
results_path = WORKDIR / "results_drafting.json"
if not results_path.exists():
    print(f"ERROR: {results_path} not found. Run run_drafting.py first.")
    exit(1)
with open(results_path) as f:
    R = json.load(f)


def load_frame(key):
    path = DRAFTING_DIR / f"frames_{key}.npz"
    if path.exists():
        return np.load(path)
    return None


def plot_pressure_field(data, title, filename, markers):
    if data is None:
        print(f"  SKIP {filename}: no frame data")
        return
    p = data["pressure"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    vlim = max(abs(p.min()), abs(p.max()))
    im = ax.imshow(p.T, origin="lower", cmap="RdBu_r", aspect="auto",
                   extent=[0, 8, 0, 6], vmin=-vlim, vmax=vlim)
    plt.colorbar(im, ax=ax, label="Pressure", shrink=0.8)
    for cx, cy, style in markers:
        ax.plot([cx], [cy], style, color="white", ms=10, mec="white", mew=1.5)
    ax.set_title(title, color="#f8f8f2")
    ax.tick_params(colors="#6272a4")
    for spine in ax.spines.values():
        spine.set_color("#6272a4")
    fig.tight_layout()
    fig.savefig(DRAFTING_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Solo vs Inline pressure fields ──
solo = load_frame("solo")
inline_fr = load_frame("inline")

plot_pressure_field(solo, "Solo Runner — Pressure Field", "drafting_solo_pressure.png",
                    [(2.0, 3.0, "s")])
plot_pressure_field(inline_fr, "Inline Tandem at 1.0 BL — Pressure Field", "drafting_inline_pressure.png",
                    [(2.0, 3.0, "s"), (2.3, 3.0, "s")])

print(f"\nAll figures saved to {DRAFTING_DIR}")
