"""γ-Reθ transition model sweep on 2D cylinder — coarse unstructured mesh.
Runs Re=[40k, 100k, 140k] with KIND_TRANS_MODEL= LM (Langtry-Menter).

Uses the coarse mesh (cl=0.01, ~44k nodes) that converged for steady RANS SST.
γ-Reθ predictions are qualitative at y+≈13, but Re-to-Re comparisons are valid.

Expected physics:
  Re=40k:   laminar BL stays laminar → Cd≈1.0—1.2 (close to ΦFlow)
  Re=100k:  possible transition in shear layer → Cd between laminar and turbulent
  Re=140k:  more transition → Cd closer to fully-turbulent SST value (~0.68)
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, SU2Solver, MeshGenerator
from pathlib import Path
import csv
import numpy as np

WORKDIR = Path(__file__).parent

RE_VALUES = [40_000, 100_000, 140_000]

# ── Generate coarse mesh once ──
mesh_path = WORKDIR / "cylinder_coarse.su2"
if not mesh_path.exists():
    print("=== Generating coarse unstructured mesh ===")
    mesh_path = MeshGenerator.cylinder_2d(
        radius=0.3, farfield_radius=15.0,
        cl_cylinder=0.01, cl_farfield=0.5,
        name=str(WORKDIR / "cylinder_coarse"),
    )
    print(f"  Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")
else:
    print(f"  Mesh exists ({mesh_path.stat().st_size:,} bytes)")


def _clean_key(k: str) -> str:
    return k.strip().strip('"').strip()


def parse_history(hist_path: Path) -> dict:
    if not hist_path.exists():
        return {}
    with hist_path.open(newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [_clean_key(h) for h in reader.fieldnames]
        last_row = None
        for row in reader:
            last_row = row
    if last_row is None:
        return {}
    try:
        return {
            "cd": float(last_row.get("DRAG", last_row.get("CD", 0))),
            "cl": float(last_row.get("LIFT", last_row.get("CL", 0))),
            "iter": int(float(last_row.get("ITER", 0))),
        }
    except (ValueError, TypeError):
        return {}


def run_case(reynolds: int, mesh: Path) -> dict:
    label = f"Re={reynolds:,}"
    dir_name = f"output_transition_coarse_re{reynolds}"
    out_dir = WORKDIR / dir_name
    hist_path = out_dir / "history_transition.csv"

    if hist_path.exists():
        data = parse_history(hist_path)
        if data:
            print(f"  [SKIP] {label} — Cd={data['cd']:.4f}")
            return data | {"reynolds": reynolds, "converged": True}

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SU2Config(
        reynolds_number=reynolds,
        reynolds_length=0.6,
        solver="INC_RANS",
        turbulence_model="SST",
        kind_trans_model="LM",
        muscl_flow="NO",
        conv_numerical_method_flow="FDS",
        cfl_number=0.5,
        iterations=6000,
        conv_residual_minval=-8,
        conv_startiter=10,
        screen_output="WARNING",
    )
    cfg_path = out_dir / "cylinder_transition.cfg"
    cfg.write(cfg_path)

    mesh_local = out_dir / mesh.name
    if not mesh_local.exists():
        shutil.copy2(mesh, mesh_local)

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Output:  {dir_name}/")
    print(f"{'='*60}")

    solver = SU2Solver(workdir=out_dir)
    result = solver.run(cfg_path, mesh_local, timeout=7200)

    hist_src = out_dir / "history.csv"
    if hist_src.exists():
        hist_src.rename(hist_path)

    data = parse_history(hist_path)
    print(f"  Done. Cd={data.get('cd', result.cd):.4f}  Cl={data.get('cl', result.cl):.6e}")
    return data | {
        "reynolds": reynolds,
        "converged": result.converged,
        "iterations": result.iterations,
    }


results = {}
for re in RE_VALUES:
    d = run_case(re, mesh_path)
    results[f"output_transition_coarse_re{re}"] = d

print("\n" + "=" * 60)
print("  TRANSITION MODEL SWEEP — COARSE MESH — SUMMARY")
print("=" * 60)
for k, d in results.items():
    print(f"  Re={d['reynolds']:,}: Cd={d.get('cd', 0):.4f}  Cl={d.get('cl', 0):.4f}  converged={d.get('converged')}")

Path(WORKDIR / "results_transition_coarse.json").write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved to results_transition_coarse.json")
