"""Run 2D cylinder validation: generate mesh, config, run SU2, display results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.su2_runner import SU2Config, MeshGenerator, SU2Solver
from pathlib import Path

WORKDIR = Path(__file__).parent

# 1. Generate mesh
print("=== Generating 2D cylinder mesh ===")
mesh_path = MeshGenerator.cylinder_2d(
    radius=0.3,
    farfield_radius=15.0,
    cl_cylinder=0.01,
    cl_farfield=0.5,
    name=str(WORKDIR / "cylinder"),
)
print(f"  Mesh: {mesh_path} ({mesh_path.stat().st_size:,} bytes)")

# 2. Create config matching phiflow: Re = U*D/ν = 1.0*0.6/1.5e-5 = 40,000
print("\n=== Generating SU2 config ===")
config = SU2Config.from_re(Re=40_000, length=0.6, incompressible=True)
config.iterations = 3000
config.cfl_number = 0.5
config.conv_residual_minval = -8
config.screen_output = "WARNING"

cfg_path = WORKDIR / "cylinder.cfg"
config.write(cfg_path)
print(f"  Config: {cfg_path}")
print(cfg_path.read_text())

# 3. Run SU2
print("\n=== Running SU2_CFD ===")
solver = SU2Solver(workdir=WORKDIR)
results = solver.run(cfg_path, mesh_path, timeout=600)

print(f"\n=== Results ===")
print(f"  Converged: {results.converged}")
print(f"  Iterations: {results.iterations}")
print(f"  Cd (drag):  {results.cd:.6f}")
print(f"  Cl (lift):  {results.cl:.6f}")
print(f"  CMz:        {results.cmz:.6f}")
print(f"  History entries: {len(results.history)}")

if results.history:
    print("\n  Last 5 history entries:")
    for entry in results.history[-5:]:
        it = entry.get("ITER", "?")
        cd = entry.get("DRAG", "?")
        cl = entry.get("LIFT", "?")
        print(f"    Iter {it:>6s}  Cd={cd:>10s}  Cl={cl:>10s}")

print(f"\n=== Comparison ===")
print(f"  SU2 Cd:       {results.cd:.3f}")
print(f"  PhiFlow Cd:   ~1.2 (2D cylinder, sub-critical Re)")
print(f"  Difference:   {abs(results.cd - 1.2) / 1.2 * 100:.1f}%")
