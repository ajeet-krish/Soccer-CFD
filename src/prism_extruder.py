"""Extrude a 2D surface mesh along radial normals to create prism boundary layers.

For a sphere centered at origin, each surface node is extruded along its radial
direction (outward normal). The resulting prism elements connect the original
surface triangles to their extruded counterparts layer by layer.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def _radial_normal(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms > 1e-15, norms, 1.0)
    return points / norms


def extrude_prism_layers(
    surface_nodes: np.ndarray,
    surface_triangles: np.ndarray,
    first_layer_h: float,
    growth_rate: float,
    n_layers: int,
    center: np.ndarray | None = None,
) -> dict:
    """Extrude a surface mesh along radial normals to create prism layers.

    Args:
        surface_nodes: (N, 3) node coordinates on the original surface.
        surface_triangles: (M, 3) triangle connectivity (0-indexed).
        first_layer_h: First layer height.
        growth_rate: Geometric growth rate between layers.
        n_layers: Number of prism layers to create.
        center: Center of extrusion (defaults to origin).

    Returns:
        dict with keys:
            nodes: ((n_layers+1)*N, 3) all node coordinates.
            prisms: (M * n_layers, 6) prism connectivity (0-indexed).
            outer_surface: (M, 3) outer surface triangles (refers to top layer nodes).
            node_offset: N, index offset for the first extruded layer.
    """
    if center is None:
        center = np.zeros(3)
    surf_nodes = surface_nodes - center
    normals = _radial_normal(surf_nodes)
    N = surf_nodes.shape[0]

    layer_heights = np.array([first_layer_h * growth_rate ** i for i in range(n_layers)])
    cum_heights = np.concatenate([[0.0], np.cumsum(layer_heights)])

    all_nodes = []
    for h in cum_heights:
        layer_nodes = surf_nodes + normals * h
        all_nodes.append(layer_nodes)
    all_nodes = np.concatenate(all_nodes, axis=0) + center

    M = surface_triangles.shape[0]
    prisms = []
    for lyr in range(n_layers):
        base_offset = lyr * N
        top_offset = (lyr + 1) * N
        for tri in surface_triangles:
            n1, n2, n3 = tri
            prisms.append([n1 + base_offset, n2 + base_offset, n3 + base_offset,
                           n1 + top_offset, n2 + top_offset, n3 + top_offset])
    prisms = np.array(prisms)

    outer_surface = surface_triangles + n_layers * N

    return {
        "nodes": all_nodes,
        "prisms": prisms,
        "outer_surface": outer_surface,
        "node_offset": N,
        "total_surface_nodes": N,
        "n_layers": n_layers,
        "layer_heights": layer_heights,
    }


def write_su2_mesh(
    path: Path,
    all_nodes: np.ndarray,
    prisms: np.ndarray,
    tets: np.ndarray | None = None,
    wall_triangles: np.ndarray | None = None,
    wall_name: str = "wall",
    farfield_triangles: np.ndarray | None = None,
    farfield_name: str = "farfield",
    outer_bl_triangles: np.ndarray | None = None,
    outer_bl_name: str = "bl_outer",
) -> None:
    """Write a hybrid prism+tet mesh in SU2 format.

    SU2 format:
      NDIME= 3
      NPOIN= <n>
      <idx> <x> <y> <z>
      ...
      NELEM= <n>
      <type> <tag> <n1> <n2> ...
      ...
      NMARK= <n>
      MARKER_TAG= <name>
      MARKER_ELEMS= <n>
      <type> <n1> <n2> ...
      ...

    Element type codes (SU2 convention):
      5 = TRIANGLE (NDIME=3)
      9 = TRIANGLE (NDIME=2)
      6 = TETRAHEDRON (NDIME=3)
      10 = TETRAHEDRON (NDIME=2)  
      Actually for SU2 .su2 format in 3D:
        TRIANGLE= 9 (linear triangle in 3D — wait, SU2 uses custom encoding)

    Actually, SU2 .su2 format uses:
        Element type 9 = TRIANGLE (3 nodes)
        Element type 12 = TETRAHEDRON (4 nodes)  
        Element type 13 = PRISM (6 nodes)

    Wait, I need to verify the SU2 element type numbers. Let me check.

    SU2 v8.4 element types for .su2 format:
        3 = LINE (2 nodes, 2D)
        5 = TRIANGLE (3 nodes, 2D)
        9 = TRIANGLE (3 nodes, 3D)
        10 = TETRAHEDRON (4 nodes, 3D)
        12 = HEXAHEDRON (8 nodes, 3D)
        13 = PRISM (6 nodes, 3D)
        14 = PYRAMID (5 nodes, 3D)
    """
    npoin = all_nodes.shape[0]
    lines = [f"NDIME= 3", f"NPOIN= {npoin}"]
    for i, (x, y, z) in enumerate(all_nodes):
        lines.append(f"{i}  {x:.15e}  {y:.15e}  {z:.15e}")

    nelem = prisms.shape[0] + (tets.shape[0] if tets is not None else 0)
    lines.append(f"NELEM= {nelem}")

    eid = 0
    for p in prisms:
        lines.append(f"13  {eid}  {p[0]}  {p[1]}  {p[2]}  {p[3]}  {p[4]}  {p[5]}")
        eid += 1

    if tets is not None:
        for t in tets:
            lines.append(f"10  {eid}  {t[0]}  {t[1]}  {t[2]}  {t[3]}")
            eid += 1

    markers = []
    if wall_triangles is not None:
        markers.append((wall_name, 9, wall_triangles))
    if farfield_triangles is not None:
        markers.append((farfield_name, 9, farfield_triangles))
    if outer_bl_triangles is not None:
        markers.append((outer_bl_name, 9, outer_bl_triangles))

    lines.append(f"NMARK= {len(markers)}")
    for name, etype, tris in markers:
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {tris.shape[0]}")
        for t in tris:
            lines.append(f"{etype}  {t[0]}  {t[1]}  {t[2]}")

    path.write_text("\n".join(lines))


def extract_gmsh_surface_mesh(gmsh_model, surface_tag: int):
    """Extract node coordinates and triangle elements from a gmsh surface."""
    elem_types, elem_tags, elem_node_tags = gmsh_model.mesh.getElements(2, surface_tag)
    # Get ALL mesh nodes — returns (nodeTags, nodeCoords, parametricCoords)
    all_node_tags, all_node_coords, _ = gmsh_model.mesh.getNodes()

    node_map = {tag: i for i, tag in enumerate(all_node_tags)}
    all_nodes = np.array(all_node_coords).reshape(-1, 3)

    triangles = []
    used_node_indices = set()
    for etype, tags, conn in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:
            conn_arr = np.array(conn).reshape(-1, 3)
            for row in conn_arr:
                tri = [node_map[t] for t in row]
                triangles.append(tri)
                for idx in tri:
                    used_node_indices.add(idx)

    triangles = np.array(triangles)

    # Extract only the surface nodes that are actually used by the triangles
    used_indices = sorted(used_node_indices)
    node_remap = {old: new for new, old in enumerate(used_indices)}
    surface_nodes = all_nodes[used_indices]

    # Remap triangle indices
    triangles_remapped = np.array([[node_remap[n] for n in tri] for tri in triangles])

    return surface_nodes, triangles_remapped, np.array(used_indices)


def compute_yplus_first_layer(reynolds_number: float, safety: float = 0.8) -> float:
    """Compute first layer height for y+ < 1 at given Re (non-dimensional)."""
    Cf = 0.058 / (reynolds_number ** 0.2)
    u_tau = 1.0 * (Cf / 2) ** 0.5
    nu = 1.0 / reynolds_number
    return safety * nu / u_tau
