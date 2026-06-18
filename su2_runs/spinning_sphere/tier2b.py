"""Tier 2b: Prism-layer mesh (true PRISM elements via radial extrusion) + γ-Reθ.

Workflow:
  1. Generate 2D sphere surface mesh with gmsh (TRIANGLE elements only)
  2. Extract surface mesh, extrude radially to create PRISM boundary layers
  3. Write prism layer mesh
  4. Use gmsh to generate farfield tet volume (sharing nodes at BL outer surface)
  5. Combine prism + tet into hybrid .su2 mesh
  6. Run SU2 with γ-Reθ transition model
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.prism_extruder import (
    extrude_prism_layers,
    extract_gmsh_surface_mesh,
    write_su2_mesh,
    compute_yplus_first_layer,
)
from src.su2_runner import SU2Config, SU2Solver

WORKDIR = Path(__file__).parent
TIER2B_DIR = WORKDIR / "tier2b"
IMAGES = WORKDIR.parent.parent / "docs" / "images" / "su2_spinning_sphere" / "tier2b"

SPHERE_RADIUS = 0.11
SPHERE_DIAMETER = 0.22
SPHERE_AREA = 3.14159265 * SPHERE_DIAMETER**2 / 4

RE = 372000
SPIN_PARAMETER_S = 0.3
ROTATION_RATE = SPIN_PARAMETER_S * 1.0 / SPHERE_RADIUS
FARFIELD_SIZE = 3.0
CL_SURFACE = 0.006
BL_LAYERS = 15
BL_GROWTH_RATE = 1.25
FIRST_LAYER_H = compute_yplus_first_layer(RE, safety=0.8)


def generate_surface_mesh():
    """Generate 2D surface mesh on sphere with gmsh."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("sphere_surface")

    sphere = gmsh.model.occ.addSphere(0, 0, 0, SPHERE_RADIUS)
    box = gmsh.model.occ.addBox(
        -FARFIELD_SIZE, -FARFIELD_SIZE, -FARFIELD_SIZE,
        2 * FARFIELD_SIZE, 2 * FARFIELD_SIZE, 2 * FARFIELD_SIZE,
    )
    fluid_vols, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()

    inner_tag = None
    outer_tags = []
    for ent in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(ent[0], ent[1])
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if all(abs(s - 2 * SPHERE_RADIUS) < 0.1 * SPHERE_RADIUS for s in [dx, dy, dz]):
            inner_tag = ent[1]
        else:
            outer_tags.append(ent[1])

    # Mesh size via Distance+Threshold field for proper surface refinement
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", [inner_tag])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", CL_SURFACE)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", CL_SURFACE * 10)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", SPHERE_RADIUS)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)

    nodes, triangles, node_tags = extract_gmsh_surface_mesh(gmsh.model, inner_tag)

    # Detect outer surfaces for marker tracking
    farfield_polys = []
    for tag in outer_tags:
        _, _, ot_conn = gmsh.model.mesh.getElements(2, tag)
        if ot_conn and len(ot_conn[0]) >= 3:
            farfield_polys.append(np.array(ot_conn[0]).reshape(-1, 3))

    # Get vol info for nodes — getNodes returns (nodeTags, nodeCoords, parametricCoords)
    all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
    all_node_map = {tag: i for i, tag in enumerate(all_node_tags)}

    gmsh.finalize()

    # Remap farfield triangles to our 0-indexed numbering
    farfield_tris_list = []
    for poly in farfield_polys:
        for tri in poly:
            farfield_tris_list.append([all_node_map[t] for t in tri])
    farfield_tris = np.array(farfield_tris_list) if farfield_tris_list else None

    return nodes, triangles, node_tags, farfield_tris


def generate_farfield_tets(outer_nodes, outer_triangles, bl_total_h):
    """Generate tet mesh in farfield using gmsh with predefined outer BL surface.

    Uses addNodes / addElementsByType to pre-set the outer sphere surface mesh
    in gmsh before volume meshing, ensuring node-sharing at the BL interface.

    Returns:
        additional_nodes: (N, 3) new nodes from farfield interior.
        tet_remapped: (M, 4) tet connectivity in 0-indexed global numbering.
        farfield_tris: (P, 3) farfield box boundary triangles in global numbering.
    """
    import gmsh

    R_outer = SPHERE_RADIUS + bl_total_h

    gmsh.initialize()
    gmsh.model.add("farfield_tets")

    outer_sph = gmsh.model.occ.addSphere(0, 0, 0, R_outer)
    ff = FARFIELD_SIZE
    ff_box = gmsh.model.occ.addBox(-ff, -ff, -ff, 2 * ff, 2 * ff, 2 * ff)
    fluid_vols, _ = gmsh.model.occ.cut([(3, ff_box)], [(3, outer_sph)])
    gmsh.model.occ.synchronize()

    # Find outer sphere surface tag and farfield face tags
    outer_sf_tag = None
    farfield_face_tags = []
    for ent in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(ent[0], ent[1])
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if all(abs(s - 2 * R_outer) < 0.1 * R_outer for s in [dx, dy, dz]):
            outer_sf_tag = ent[1]
        else:
            farfield_face_tags.append(ent[1])

    if outer_sf_tag is None:
        gmsh.finalize()
        raise RuntimeError("Could not find outer sphere surface in gmsh model")

    # Pre-set nodes on outer sphere surface via addNodes
    n_outer = outer_nodes.shape[0]
    outer_node_tags = list(range(1, n_outer + 1))
    outer_coords_flat = outer_nodes.flatten().tolist()

    gmsh.model.mesh.addNodes(2, outer_sf_tag, outer_node_tags, outer_coords_flat)

    # Pre-set triangle elements on outer sphere surface via addElementsByType
    # Element type 2 = 3-node triangle
    # outer_triangles indices are in the extruded range (n_layers*N .. (n_layers+1)*N-1)
    # We need to remap them to 1..N for gmsh (matching addNodes)
    n_outer_tris = outer_triangles.shape[0]
    outer_elem_tags = list(range(1, n_outer_tris + 1))
    outer_base = n_outer * BL_LAYERS  # first index of outermost layer
    outer_conn_flat = []
    for tri in outer_triangles:
        outer_conn_flat.extend([int(t) - outer_base + 1 for t in tri])

    gmsh.model.mesh.addElementsByType(
        outer_sf_tag, 2, outer_elem_tags, outer_conn_flat,
    )

    # Set mesh size on farfield faces
    for tag in farfield_face_tags:
        gmsh.model.mesh.setSize(gmsh.model.getBoundary([(2, tag)]), CL_SURFACE * 10)

    # Generate 3D volume mesh — gmsh should preserve our pre-set surface mesh
    gmsh.model.mesh.generate(3)

    # Extract tetrahedral elements
    vol_tags = [v[1] for v in gmsh.model.getEntities(3)]
    all_tets = []
    for vt in vol_tags:
        etypes, etags, econn = gmsh.model.mesh.getElements(3, vt)
        for etype, conn in zip(etypes, econn):
            if etype == 4:
                conn_arr = np.array(conn).reshape(-1, 4)
                all_tets.append(conn_arr)

    # Extract farfield boundary triangles from box faces
    ff_tris_raw = []
    for tag in farfield_face_tags:
        _, _, ff_conn = gmsh.model.mesh.getElements(2, tag)
        if ff_conn and len(ff_conn[0]) >= 3:
            conn_arr = np.array(ff_conn[0]).reshape(-1, 3)
            ff_tris_raw.append(conn_arr)

    # Get all nodes — getNodes returns (nodeTags, nodeCoords, parametricCoords)
    all_ntags, all_ncoords, _ = gmsh.model.mesh.getNodes()
    gmsh.finalize()

    if not all_tets:
        raise RuntimeError("No tetrahedral elements generated in farfield")

    all_tets = np.concatenate(all_tets, axis=0)

    # Build node coordinate map from gmsh tags
    node_coord_map = {}
    for tag, x, y, z in zip(all_ntags, all_ncoords[0::3], all_ncoords[1::3], all_ncoords[2::3]):
        node_coord_map[int(tag)] = (x, y, z)

    # Remap tet connectivity to 0-indexed, merging with outer surface nodes
    tag_to_idx = {}
    for i in range(1, n_outer + 1):
        tag_to_idx[i] = i - 1

    next_idx = n_outer
    additional_nodes = []

    for tet in all_tets:
        for tag in tet:
            tag_int = int(tag)
            if tag_int not in tag_to_idx:
                tag_to_idx[tag_int] = next_idx
                next_idx += 1
                additional_nodes.append(node_coord_map[tag_int])

    additional_nodes = np.array(additional_nodes) if additional_nodes else np.empty((0, 3))

    tet_remapped = np.array([[tag_to_idx[int(t)] for t in tet] for tet in all_tets])

    # Remap farfield boundary triangles
    ff_tris_list = []
    for tris in ff_tris_raw:
        for tri in tris:
            ff_tris_list.append([tag_to_idx[int(t)] for t in tri])
    farfield_tris = np.array(ff_tris_list) if ff_tris_list else None

    return additional_nodes, tet_remapped, farfield_tris


def run_tier2b() -> dict:
    """Execute Tier 2b simulation."""
    TIER2B_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    # Step 1-2: Surface mesh + prism extrusion
    print("=" * 60)
    print("Tier 2b: Surface Mesh + Prism Extrusion")
    print("=" * 60)
    t0 = time.time()

    surf_nodes, surf_tris, _, farfield_tris = generate_surface_mesh()
    print(f"  Surface nodes: {surf_nodes.shape[0]}")
    print(f"  Surface triangles: {surf_tris.shape[0]}")

    bl_total_h = FIRST_LAYER_H * (BL_GROWTH_RATE**BL_LAYERS - 1) / (BL_GROWTH_RATE - 1)
    print(f"  BL first layer: {FIRST_LAYER_H:.2e} ({FIRST_LAYER_H*SPHERE_DIAMETER*1000:.4f} mm)")
    print(f"  BL layers: {BL_LAYERS}, growth: {BL_GROWTH_RATE}, total: {bl_total_h:.4f}")

    extrude_result = extrude_prism_layers(
        surf_nodes, surf_tris,
        first_layer_h=FIRST_LAYER_H,
        growth_rate=BL_GROWTH_RATE,
        n_layers=BL_LAYERS,
    )
    print(f"  Prism nodes: {extrude_result['nodes'].shape[0]}")
    print(f"  Prism elements: {extrude_result['prisms'].shape[0]}")

    # Step 3: Generate farfield tets
    print("\n" + "=" * 60)
    print("Tier 2b: Farfield Tet Volume")
    print("=" * 60)
    t1 = time.time()

    outer_nodes_extruded = extrude_result["nodes"][extrude_result["total_surface_nodes"] * extrude_result["n_layers"]:]
    additional_nodes, farfield_tets, farfield_tris = generate_farfield_tets(
        outer_nodes_extruded, extrude_result["outer_surface"], bl_total_h,
    )
    print(f"  Farfield tets: {farfield_tets.shape[0]}")
    print(f"  Additional nodes: {additional_nodes.shape[0]}")
    t2 = time.time()
    print(f"  Time: {t2 - t1:.1f}s")

    # Step 4: Combine prism + tet into single mesh
    all_nodes = np.concatenate([extrude_result["nodes"], additional_nodes], axis=0)

    # Remap tet connectivity to account for additional nodes
    n_prism_nodes = extrude_result["nodes"].shape[0]
    tet_remapped = farfield_tets + 0  # copy

    # The prism outer surface nodes are at indices:
    # n_prism_nodes - outer_nodes_extruded.shape[0] ... n_prism_nodes - 1
    # The farfield tets use indices 0..n_outer-1 for the shared surface
    # and n_outer..n_outer+n_additional-1 for interior nodes
    # We need to remap: shared surface nodes → prism outer indices
    #                    interior nodes → prism_total + unique_index

    n_outer = outer_nodes_extruded.shape[0]
    outer_start = n_prism_nodes - n_outer
    n_additional = additional_nodes.shape[0]

    # Build remap: for shared surface (gmsh tags 1..n_outer → prism indices outer_start..outer_start+n_outer-1)
    # and for additional nodes (gmsh tags n_outer+1.. → prism_total + 0..n_additional-1)
    tet_remapped = np.where(
        tet_remapped < n_outer,
        tet_remapped + outer_start,
        tet_remapped + n_prism_nodes - n_outer,
    )

    # Wall triangles: original surface mesh (0-indexed)
    wall_tris = surf_tris.copy()

    # Outer BL surface triangles: top of prism layer
    outer_bl_tris = extrude_result["outer_surface"].copy()

    # Farfield triangles: remap to match combined node numbering
    if farfield_tris is not None:
        ff_tris_remapped = np.where(
            farfield_tris < n_outer,
            farfield_tris + outer_start,
            farfield_tris + n_prism_nodes - n_outer,
        )
    else:
        ff_tris_remapped = None

    # Write hybrid mesh
    mesh_path = TIER2B_DIR / "mesh.su2"
    write_su2_mesh(
        mesh_path,
        all_nodes=all_nodes,
        prisms=extrude_result["prisms"],
        tets=tet_remapped,
        wall_triangles=wall_tris,
        farfield_triangles=ff_tris_remapped,
        outer_bl_triangles=outer_bl_tris,
        wall_name="wall",
        farfield_name="farfield",
        outer_bl_name="bl_outer",
    )
    print(f"  Written: {mesh_path.name} ({mesh_path.stat().st_size / 1024:.0f} KB)")

    # Print mesh info
    text = mesh_path.read_text()
    for marker in ("TRIANGLE", "TETRAHEDRON", "PRISM", "HEXAHEDRON"):
        cnt = text.count(marker)
        if cnt:
            print(f"  {cnt:>6} {marker}s")

    # Step 5: Config
    print("\n" + "=" * 60)
    print("Tier 2b: Config (γ-Reθ transition model)")
    print("=" * 60)
    config = SU2Config.from_re(Re=RE, length=SPHERE_DIAMETER, incompressible=True)
    config.ref_area = SPHERE_AREA
    config.iterations = 2000
    config.cfl_number = 0.5
    config.conv_residual_minval = -6
    config.screen_output = "WARNING"
    config.rotation_rate = ROTATION_RATE
    config.kind_trans_model = "LM"

    cfg_path = TIER2B_DIR / "sphere_magnus_transition.cfg"
    config.write(cfg_path)
    print(f"  Config: {cfg_path.name}")
    print(f"  Re = {RE:.0f}, S = {SPIN_PARAMETER_S}  (ωz = {ROTATION_RATE:.3f})")
    print(f"  Transition model: γ-Reθ (LM)")

    # Step 6: Solver
    print("\n" + "=" * 60)
    print("Tier 2b: SU2 Solver")
    print("=" * 60)
    solver = SU2Solver(workdir=TIER2B_DIR)
    t3 = time.time()
    result = solver.run(cfg_path, mesh_path, timeout=14400)
    t4 = time.time()
    print(f"  Cd = {result.cd:.4f}")
    print(f"  Cl = {result.cl:.4f}")
    print(f"  Converged = {result.converged}")
    print(f"  Iterations = {result.iterations}")
    print(f"  Wall time: {t4 - t3:.0f}s ({((t4 - t3) / 60):.1f} min)")

    output = {
        "tier": "2b",
        "case": "3D spinning sphere — true PRISM layers + γ-Reθ transition",
        "mesh": {
            "cl_surface": CL_SURFACE,
            "cl_farfield": CL_SURFACE * 10,
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
            "wall_time_s": round(t4 - t3, 1),
        },
    }
    results_path = TIER2B_DIR / "results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'=' * 60}")
    print(f"Wall time (total): {t4 - t0:.0f}s ({((t4 - t0) / 60):.1f} min)")
    print(f"{'=' * 60}")

    return output


if __name__ == "__main__":
    result = run_tier2b()
    print(f"\n{'=' * 60}")
    print(f"Tier 2b complete: Cd={result['result']['cd']:.4f}, Cl={result['result']['cl']:.4f}")
    print(f"{'=' * 60}")
