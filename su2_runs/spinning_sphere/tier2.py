"""Tier 2: Prism-layer mesh + γ-Reθ transition model for 3D spinning sphere.

Generates a tetrahedral mesh with prism boundary layers on the sphere surface,
runs SU2 steady RANS SST + γ-Reθ (Langtry-Menter) with rotation (S=0.3),
and saves structured results.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.su2_runner import SU2Config, SU2Solver

WORKDIR = Path(__file__).parent
TIER2_DIR = WORKDIR / "tier2"
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_spinning_sphere" / "tier2"

SPHERE_RADIUS = 0.11
SPHERE_DIAMETER = 0.22
SPHERE_AREA = 3.14159265 * SPHERE_DIAMETER**2 / 4

RE = 372000  # 25 m/s on size-5 ball
SPIN_PARAMETER_S = 0.3
ROTATION_RATE = SPIN_PARAMETER_S * 1.0 / SPHERE_RADIUS

# First-layer height for y+ < 1 at Re=3.72e5
# Cf ≈ 0.058/Re^0.2 (turbulent flat plate), u_τ = U*sqrt(Cf/2), ν = 1/Re
_CF = 0.058 / (RE**0.2)
_U_TAU = 1.0 * (_CF / 2) ** 0.5
_NU = 1.0 / RE
FIRST_LAYER_H = 0.8 * _NU / _U_TAU  # y+ ≈ 0.8 < 1
BL_GROWTH_RATE = 1.2
BL_LAYERS = 25
BL_TOTAL_H = FIRST_LAYER_H * (BL_GROWTH_RATE**BL_LAYERS - 1) / (BL_GROWTH_RATE - 1)

FARFIELD_SIZE = 3.0


def generate_mesh() -> Path:
    """Tier 2 mesh: tetrahedral with 25 prism boundary layers on sphere surface."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("spinning_sphere_tier2")

    sphere = gmsh.model.occ.addSphere(0, 0, 0, SPHERE_RADIUS)

    ff = FARFIELD_SIZE
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

    # Background mesh size field (gradation from sphere to farfield)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", inner_tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.006)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.15)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", SPHERE_RADIUS)

    # Boundary layer field — prism layers on sphere surface edges
    sphere_curves = []
    for tag in inner_tags:
        boundary = gmsh.model.getBoundary([(2, tag)])
        for bdim, btag in boundary:
            if bdim == 1:
                sphere_curves.append(btag)

    bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", sphere_curves)
    gmsh.model.mesh.field.setNumber(bl_field, "Size", FIRST_LAYER_H)
    gmsh.model.mesh.field.setNumber(bl_field, "Ratio", BL_GROWTH_RATE)
    gmsh.model.mesh.field.setNumber(bl_field, "Thickness", BL_TOTAL_H)

    # Combine fields (Min of background + BL)
    gmsh.model.mesh.field.add("Min", 10)
    gmsh.model.mesh.field.setNumbers(10, "FieldsList", [2, bl_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(10)

    gmsh.model.mesh.generate(3)

    # Physical groups
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

    TIER2_DIR.mkdir(parents=True, exist_ok=True)
    out = TIER2_DIR / "mesh.su2"
    gmsh.write(str(out))
    gmsh.finalize()

    return out


def print_mesh_info(mesh_path: Path) -> None:
    text = mesh_path.read_text()
    for marker in ("TRIANGLE", "TETRAHEDRON", "PRISM", "HEXAHEDRON"):
        cnt = text.count(marker)
        if cnt:
            print(f"  {cnt:>6} {marker}s")
    print(f"  File: {mesh_path.stat().st_size / 1024:.0f} KB")


def run_tier2() -> dict:
    """Execute Tier 2 simulation."""
    TIER2_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tier 2: Mesh Generation (prism boundary layers)")
    print("=" * 60)
    t0 = time.time()
    mesh_path = generate_mesh()
    t1 = time.time()
    print(f"  Mesh: {mesh_path.name}")
    print_mesh_info(mesh_path)
    print(f"  Time: {t1 - t0:.1f}s")
    print(f"  BL first layer: {FIRST_LAYER_H:.2e} ({FIRST_LAYER_H*SPHERE_DIAMETER*1000:.3f} mm)")
    print(f"  BL layers: {BL_LAYERS}, growth: {BL_GROWTH_RATE}, total: {BL_TOTAL_H:.4f}")

    print("\n" + "=" * 60)
    print("Tier 2: Config (γ-Reθ transition model)")
    print("=" * 60)
    config = SU2Config.from_re(Re=RE, length=SPHERE_DIAMETER, incompressible=True)
    config.ref_area = SPHERE_AREA
    config.iterations = 2000
    config.cfl_number = 0.5
    config.conv_residual_minval = -6
    config.screen_output = "WARNING"
    config.rotation_rate = ROTATION_RATE
    config.kind_trans_model = "LM"

    cfg_path = TIER2_DIR / "sphere_magnus_transition.cfg"
    config.write(cfg_path)
    print(f"  Config: {cfg_path.name}")
    print(f"  Re = {RE:.0f}, S = {SPIN_PARAMETER_S}  (ωz = {ROTATION_RATE:.3f})")
    print(f"  Transition model: γ-Reθ (LM)")

    print("\n" + "=" * 60)
    print("Tier 2: SU2 Solver")
    print("=" * 60)
    solver = SU2Solver(workdir=TIER2_DIR)
    t2 = time.time()
    result = solver.run(cfg_path, mesh_path, timeout=7200)
    t3 = time.time()
    print(f"  Cd = {result.cd:.4f}")
    print(f"  Cl = {result.cl:.4f}")
    print(f"  Converged = {result.converged}")
    print(f"  Iterations = {result.iterations}")
    print(f"  Wall time: {t3 - t2:.0f}s ({((t3 - t2) / 60):.1f} min)")

    output = {
        "tier": 2,
        "case": "3D spinning sphere — γ-Reθ transition model with prism layers",
        "mesh": {
            "cl_surface": 0.006,
            "cl_farfield": 0.15,
            "farfield_radius": FARFIELD_SIZE,
            "bl_first_layer": FIRST_LAYER_H,
            "bl_growth_rate": BL_GROWTH_RATE,
            "bl_layers": BL_LAYERS,
        },
        "physics": {
            "reynolds_number": RE,
            "spin_parameter": SPIN_PARAMETER_S,
            "rotation_rate_non_dim": ROTATION_RATE,
            "solver": "INC_RANS",
            "turbulence_model": "SST + γ-Reθ",
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
    results_path = TIER2_DIR / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {results_path}")

    return output


if __name__ == "__main__":
    result = run_tier2()
    print(f"\n{'=' * 60}")
    print(f"Tier 2 complete: Cd={result['result']['cd']:.4f}, Cl={result['result']['cl']:.4f}")
    print(f"{'=' * 60}")
