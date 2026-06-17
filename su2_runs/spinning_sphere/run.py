"""Tier 1: Steady RANS SST baseline for 3D spinning sphere with Magnus effect.

Generates a refined tetrahedral mesh with wake refinement box, runs SU2
steady RANS SST with rotation (S=0.3), and saves structured results.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.su2_runner import SU2Config, SU2Solver

WORKDIR = Path(__file__).parent
TIER1_DIR = WORKDIR / "tier1"
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_spinning_sphere" / "tier1"

SPHERE_RADIUS = 0.11
SPHERE_DIAMETER = 0.22
SPHERE_AREA = 3.14159265 * SPHERE_DIAMETER**2 / 4

RE = 372000  # 25 m/s on size-5 ball
SPIN_PARAMETER_S = 0.3

# Non-dim rotation rate: S = ωR/U, so ω = S * U / R
ROTATION_RATE = SPIN_PARAMETER_S * 1.0 / SPHERE_RADIUS


def generate_mesh() -> Path:
    """Tier 1 mesh: refined tetrahedral with wake refinement box."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("spinning_sphere_tier1")

    sphere = gmsh.model.occ.addSphere(0, 0, 0, SPHERE_RADIUS)

    ff = 3.0
    box = gmsh.model.occ.addBox(-ff, -ff, -ff, 2 * ff, 2 * ff, 2 * ff)

    fluid_vols, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()

    inner_tags = []
    outer_tags = []
    for ent in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(ent[0], ent[1])
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        r = (dx + dy + dz) / 6
        if all(abs(s - 2 * SPHERE_RADIUS) < 0.1 * SPHERE_RADIUS for s in [dx, dy, dz]):
            inner_tags.append(ent[1])
        elif (abs(dx) < 0.01 and abs(dy - 2 * ff) < 0.1 * ff and abs(dz - 2 * ff) < 0.1 * ff):
            outer_tags.append(ent[1])
        elif (abs(dy) < 0.01 and abs(dx - 2 * ff) < 0.1 * ff and abs(dz - 2 * ff) < 0.1 * ff):
            outer_tags.append(ent[1])
        elif (abs(dz) < 0.01 and abs(dx - 2 * ff) < 0.1 * ff and abs(dy - 2 * ff) < 0.1 * ff):
            outer_tags.append(ent[1])

    # Size fields — set BEFORE mesh generation
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", inner_tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.003)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.15)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * SPHERE_RADIUS)

    # Wake refinement box
    gmsh.model.mesh.field.add("Box", 3)
    gmsh.model.mesh.field.setNumber(3, "VIn", 0.006)
    gmsh.model.mesh.field.setNumber(3, "VOut", 0.15)
    gmsh.model.mesh.field.setNumber(3, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(3, "XMax", 2.0)
    gmsh.model.mesh.field.setNumber(3, "YMin", -0.3)
    gmsh.model.mesh.field.setNumber(3, "YMax", 0.3)
    gmsh.model.mesh.field.setNumber(3, "ZMin", -0.3)
    gmsh.model.mesh.field.setNumber(3, "ZMax", 0.3)

    gmsh.model.mesh.field.add("Min", 10)
    gmsh.model.mesh.field.setNumbers(10, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(10)

    gmsh.model.mesh.generate(3)

    # Physical groups after mesh generation
    if inner_tags:
        wall = gmsh.model.addPhysicalGroup(2, inner_tags)
        gmsh.model.setPhysicalName(2, wall, "wall")
    if outer_tags:
        far = gmsh.model.addPhysicalGroup(2, outer_tags)
        gmsh.model.setPhysicalName(2, far, "farfield")
    vols = gmsh.model.getEntities(3)
    if vols:
        gv = gmsh.model.addPhysicalGroup(3, [v[1] for v in vols])
        gmsh.model.setPhysicalName(3, gv, "fluid")

    gmsh.model.mesh.createTopology()

    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    out = TIER1_DIR / "mesh.su2"
    gmsh.write(str(out))
    gmsh.finalize()

    return out


def print_mesh_info(mesh_path: Path) -> None:
    """Print mesh element/node counts from .su2 file."""
    text = mesh_path.read_text()
    n_elems = text.count("NELEM=")
    n_points = text.count("NPOIN=")
    n_lines = len(text.splitlines())
    # Count element type markers
    for marker in ("TRIANGLE", "TETRAHEDRON", "PRISM", "HEXAHEDRON"):
        cnt = text.count(marker)
        if cnt:
            print(f"  {cnt:>6} {marker}s")
    print(f"  File: {n_lines} lines, {mesh_path.stat().st_size / 1024:.0f} KB")


def run_tier1() -> dict:
    """Execute Tier 1 simulation."""
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # 1. Mesh
    print("=" * 60)
    print("Tier 1: Mesh Generation")
    print("=" * 60)
    t0 = time.time()
    mesh_path = generate_mesh()
    t1 = time.time()
    print(f"  Mesh: {mesh_path.name}")
    print_mesh_info(mesh_path)
    print(f"  Time: {t1 - t0:.1f}s")

    # 2. Config
    print("\n" + "=" * 60)
    print("Tier 1: Config")
    print("=" * 60)
    config = SU2Config.from_re(Re=RE, length=SPHERE_DIAMETER, incompressible=True)
    config.ref_area = SPHERE_AREA
    config.iterations = 500
    config.cfl_number = 0.5
    config.conv_residual_minval = -6
    config.screen_output = "WARNING"
    config.rotation_rate = ROTATION_RATE

    cfg_path = TIER1_DIR / "sphere_magnus.cfg"
    config.write(cfg_path)
    print(f"  Config: {cfg_path.name}")
    print(f"  Re = {RE:.0f}")
    print(f"  S = {SPIN_PARAMETER_S}  (ωz = {ROTATION_RATE:.3f} non-dim)")

    # 3. Solver
    print("\n" + "=" * 60)
    print("Tier 1: SU2 Solver")
    print("=" * 60)
    solver = SU2Solver(workdir=TIER1_DIR)
    t2 = time.time()
    result = solver.run(cfg_path, mesh_path, timeout=7200)
    t3 = time.time()
    print(f"  Cd = {result.cd:.4f}")
    print(f"  Cl = {result.cl:.4f}")
    print(f"  Converged = {result.converged}")
    print(f"  Iterations = {result.iterations}")
    print(f"  Wall time: {t3 - t2:.0f}s ({((t3 - t2) / 60):.1f} min)")

    # 4. Save results JSON
    output = {
        "tier": 1,
        "case": "3D spinning sphere — steady RANS SST",
        "mesh": {
            "cl_surface": 0.003,
            "cl_wake": 0.006,
            "cl_farfield": 0.15,
            "farfield_radius": 3.0,
        },
        "physics": {
            "reynolds_number": RE,
            "spin_parameter": SPIN_PARAMETER_S,
            "rotation_rate_non_dim": ROTATION_RATE,
            "solver": "INC_RANS",
            "turbulence_model": "SST",
            "ref_area": SPHERE_AREA,
        },
        "result": {
            "cd": result.cd,
            "cl": result.cl,
            "converged": result.converged,
            "iterations": result.iterations,
            "wall_time_s": round(t3 - t2, 1),
        },
    }
    results_path = TIER1_DIR / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {results_path}")

    return output


if __name__ == "__main__":
    result = run_tier1()
    print(f"\n{'=' * 60}")
    print(f"Tier 1 complete: Cd={result['result']['cd']:.4f}, Cl={result['result']['cl']:.4f}")
    print(f"{'=' * 60}")
