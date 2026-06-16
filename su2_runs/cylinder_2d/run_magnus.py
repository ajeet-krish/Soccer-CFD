"""Run rotating cylinder (Magnus effect) — same Re, same mesh, spinning wall."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, SU2Solver
from pathlib import Path

WORKDIR = Path(__file__).parent

# Spin parameter S = ω·R/U∞ = 0.2 (typical soccer ball)
# ω_non_dim = ω_dim · D / V_ref = 2·S · U∞ · R / (U∞ · R) ... actually:
#   S = ω·R/U∞ → ω_dim = S·U∞/R = 0.2·1.0/0.3 = 0.667
#   ω_non_dim = ω_dim · D / U_ref = 0.667 · 0.6 / 1.0 = 0.4
SPIN_RATE = 0.4  # non-dimensional ωz

# Generate config with spin
config = SU2Config.from_re(Re=40_000, length=0.6, incompressible=True)
config.iterations = 3000
config.cfl_number = 0.5
config.conv_residual_minval = -8
config.screen_output = "WARNING"
config.rotation_rate = SPIN_RATE

cfg_path = WORKDIR / "cylinder_magnus.cfg"
config.write(cfg_path)
print(f"Config written: {cfg_path}")

# Use existing mesh (same geometry, just rotating wall)
mesh_path = WORKDIR / "cylinder.su2"
if not mesh_path.exists():
    # Generate mesh if missing
    from src.su2_runner import MeshGenerator
    mesh_path = MeshGenerator.cylinder_2d(name=str(WORKDIR / "cylinder"))

# Run SU2
print("Running SU2 (Magnus case)...")
solver = SU2Solver(workdir=WORKDIR)
result = solver.run(cfg_path, mesh_path, timeout=600)

print(f"\nResults:")
print(f"  Cd: {result.cd:.4f}")
print(f"  Cl: {result.cl:.4f}")
print(f"  Converged: {result.converged}")
print(f"  Iterations: {result.iterations}")

# Save results
Path(WORKDIR / "results_magnus.json").write_text(json.dumps({
    "case": "2D Cylinder — Magnus (spinning)",
    "reynolds": 40000,
    "spin_rate": SPIN_RATE,
    "spin_parameter": 0.2,
    "cd": result.cd,
    "cl": result.cl,
    "converged": result.converged,
    "iterations": result.iterations,
}, indent=2))
print("Results saved to results_magnus.json")
