"""SU2 simulation runner — config generation, mesh creation, solver invocation, result parsing, PyVista viz."""

from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

SU2_BIN = Path("/Users/ajeet/SU2_CFD/bin")
SU2_CFD = SU2_BIN / "SU2_CFD"
SU2_DEF = SU2_BIN / "SU2_DEF"
SU2_GEO = SU2_BIN / "SU2_GEO"


# ── Config generation ──

@dataclasses.dataclass
class SU2Config:
    """Parameters for an SU2 .cfg file."""

    reynolds_number: float = 40_000
    reynolds_length: float = 0.6  # cylinder diameter
    mach_number: float = 0.059
    angle_of_attack: float = 0.0
    solver: str = "INC_RANS"
    turbulence_model: str = "SST"
    inc_density_model: str = "CONSTANT"
    inc_energy_eq: str = "NO"
    inc_velocity_init: tuple = (1.0, 0.0, 0.0)
    inc_density_init: float = 1.2886
    cfl_number: float = 1.0
    conv_residual_minval: float = -8
    conv_startiter: float = 10
    conv_field: str = "DRAG"
    iterations: int = 2000
    screen_output: str = "WARNING"
    ref_area: Optional[float] = None  # 3D: set to frontal area (e.g. πD²/4 for sphere)
    rotation_rate: float = 0.0  # non-dim ωz (spin about z-axis, e.g. Magnus effect)
    wall_roughness: float = 0.0  # dimensional roughness height k_s (m) for WALL_ROUGHNESS
    muscl_flow: str = "NO"  # MUSCL reconstruction; "YES" for 2nd order (laminar), "NO" for robustness (RANS)
    conv_numerical_method_flow: str = "FDS"  # FDS (incompressible) or ROE (compressible laminar)
    # Unsteady (URANS) parameters
    time_domain: bool = False
    time_marching: str = "NO"
    time_step: float = 0.0
    max_time: float = 0.0
    time_iter: int = 1
    inner_iter: int = 15
    output_wrt_freq: int = 1

    @classmethod
    def from_re(cls, Re: float, length: float = 0.6, incompressible: bool = True) -> "SU2Config":
        """Create config from Reynolds number."""
        if incompressible:
            return cls(reynolds_number=Re, reynolds_length=length, solver="INC_RANS")
        return cls(reynolds_number=Re, reynolds_length=length, solver="RANS",
                    mach_number=0.059 * (Re / 200_000))

    def write(self, path: Path) -> None:
        """Generate a .cfg file at path."""
        is_inc = self.solver.startswith("INC_")
        lines = [
            f"% ------- CONFIG FILE (auto-generated) --------",
            f"SOLVER= {self.solver}",
            f"{('KIND_TURB_MODEL= ' + self.turbulence_model) if ('RANS' in self.solver or 'rans' in self.solver) else ('KIND_TURB_MODEL= NONE' if self.turbulence_model == 'NONE' else '% No turbulence model (laminar)')}",
            f"MATH_PROBLEM= DIRECT",
            f"RESTART_SOL= NO",
            f"SYSTEM_MEASUREMENTS= SI",
            f"",
        ]
        if is_inc:
            # Non-dimensional viscosity: mu_non_dim = 1/Re (since rho=V=L=1 in non-dim)
            mu_nd = 1.0 / self.reynolds_number
            lines += [
                f"% ---------------- INCOMPRESSIBLE FLOW CONDITION DEFINITION ------------",
                f"INC_DENSITY_MODEL= {self.inc_density_model}",
                f"INC_ENERGY_EQUATION= {self.inc_energy_eq}",
                f"INC_DENSITY_INIT= {self.inc_density_init}",
                f"INC_VELOCITY_INIT= ( {self.inc_velocity_init[0]}, {self.inc_velocity_init[1]}, {self.inc_velocity_init[2]} )",
                f"INC_NONDIM= INITIAL_VALUES",
                f"REYNOLDS_NUMBER= {int(self.reynolds_number)}",
                f"REYNOLDS_LENGTH= {self.reynolds_length}",
                f"VISCOSITY_MODEL= CONSTANT_VISCOSITY",
                f"MU_CONSTANT= {mu_nd:.6e}",
                f"",
            ]
        else:
            lines += [
                f"% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION -----------",
                f"MACH_NUMBER= {self.mach_number:.6f}",
                f"AOA= {self.angle_of_attack:.2f}",
                f"REYNOLDS_NUMBER= {int(self.reynolds_number)}",
                f"REYNOLDS_LENGTH= {self.reynolds_length}",
                f"FREESTREAM_TEMPERATURE= 288.15",
                f"",
            ]
        lines += [
            f"% ------------------------ BOUNDARY CONDITIONS -------------------------",
            f"MARKER_HEATFLUX= ( wall, 0.0 )",
            f"{f'WALL_ROUGHNESS= ( wall, {self.wall_roughness:.6e} )' if self.wall_roughness > 0 else '% No wall roughness'}",
            f"MARKER_MONITORING= ( wall )",
            f"MARKER_FAR= ( farfield )",
            f"{f'SURFACE_MOVEMENT= MOVING_WALL' if abs(self.rotation_rate) > 1e-10 else '% No moving wall'}",
            f"{f'MARKER_MOVING= ( wall )' if abs(self.rotation_rate) > 1e-10 else ''}",
            f"{f'SURFACE_MOTION_ORIGIN= 0.0 0.0 0.0' if abs(self.rotation_rate) > 1e-10 else ''}",
            f"{f'SURFACE_ROTATION_RATE= 0.0 0.0 {self.rotation_rate}' if abs(self.rotation_rate) > 1e-10 else ''}",
            f"",
            f"% ------------------------ NUMERICAL METHOD DEFINITION -------------------",
            f"CONV_NUM_METHOD_FLOW= {self.conv_numerical_method_flow}",
            f"MUSCL_FLOW= {self.muscl_flow}",
            f"SLOPE_LIMITER_FLOW= VENKATAKRISHNAN",
            f"TIME_DISCRE_FLOW= EULER_IMPLICIT",
            f"{f'CONV_NUM_METHOD_TURB= SCALAR_UPWIND' if 'RANS' in self.solver or 'rans' in self.solver else ''}",
            f"{f'MUSCL_TURB= NO' if 'RANS' in self.solver or 'rans' in self.solver else ''}",
            f"",
            f"% ------------------------- CONVERGENCE PARAMETERS -----------------------",
            f"{'' if self.time_domain else f'ITER= {self.iterations}'}",
            f"CFL_NUMBER= {self.cfl_number}",
            f"CFL_ADAPT= NO",
            f"CONV_FIELD= {self.conv_field}",
            f"CONV_RESIDUAL_MINVAL= {self.conv_residual_minval}",
            f"CONV_STARTITER= {self.conv_startiter}",
            f"CONV_CAUCHY_ELEMS= 100",
            f"CONV_CAUCHY_EPS= 1E-10",
            f"",
            f"% ------------------------- TIME DOMAIN (UNSTEADY) ------------------------",
            f"{f'TIME_DOMAIN= YES' if self.time_domain else '% No time domain' }",
            f"{f'TIME_MARCHING= {self.time_marching}' if self.time_domain else ''}",
            f"{f'TIME_STEP= {self.time_step}' if self.time_domain else ''}",
            f"{f'MAX_TIME= {self.max_time}' if self.time_domain else ''}",
            f"{f'TIME_ITER= {self.time_iter}' if self.time_domain else ''}",
            f"{f'INNER_ITER= {self.inner_iter}' if self.time_domain else ''}",
            f"{f'OUTPUT_WRT_FREQ= {self.output_wrt_freq}' if self.time_domain else ''}",
            f"",
            f"% ------------------------- LINEAR SOLVER --------------------------------",
            f"LINEAR_SOLVER= FGMRES",
            f"LINEAR_SOLVER_PREC= ILU",
            f"LINEAR_SOLVER_ERROR= 1E-6",
            f"LINEAR_SOLVER_ITER= 10",
            f"",
            f"% ------------------------- REFERENCE VALUE DEFINITION -------------------",
            f"{f'REF_AREA= {self.ref_area}' if self.ref_area else '%% REF_AREA not set (assuming 2D)'}",
            f"",
            f"% ------------------------- OUTPUT ---------------------------------------",
            f"SCREEN_OUTPUT= ({self.screen_output})",
            f"HISTORY_OUTPUT= ( TIME_ITER, INNER_ITER, RMS_RES, AERO_COEFF {f', CUR_TIME' if self.time_domain else ''})",
            f"TABULAR_FORMAT= CSV",
            f"OUTPUT_FILES= (RESTART, PARAVIEW)",
            f"",
            f"% ------------------------- MULTIGRID -------------------------------------",
            f"MGLEVEL= 0",
            f"",
        ]
        path.write_text("\n".join(lines))


# ── Mesh generation ──

class MeshGenerator:
    """gmsh-based mesh generation for SU2 cases."""

    @staticmethod
    def cylinder_2d(
        radius: float = 0.3,
        farfield_radius: float = 20.0,
        cl_cylinder: float = 0.01,
        cl_farfield: float = 1.0,
        cl_wake: Optional[float] = None,
        wake_length: float = 12.0,
        wake_width: float = 3.0,
        name: str = "cylinder",
    ) -> Path:
        """Create a 2D cylinder mesh with optional wake refinement.
        
        Args:
            cl_wake: Element size in wake region (None = no wake refinement)
            wake_length: Length of refined wake region downstream (x > 0)
            wake_width: Half-width of refined wake region (|y| < wake_width)
        """
        import gmsh

        gmsh.initialize()
        gmsh.model.add(name)

        # Inner circle (cylinder) and outer circle (farfield)
        inner_tag = gmsh.model.occ.addCircle(0, 0, 0, radius)
        outer_tag = gmsh.model.occ.addCircle(0, 0, 0, farfield_radius)

        # Create plane surface between circles
        loop_inner = gmsh.model.occ.addCurveLoop([inner_tag])
        loop_outer = gmsh.model.occ.addCurveLoop([outer_tag], -1)
        surface = gmsh.model.occ.addPlaneSurface([loop_outer, loop_inner])
        gmsh.model.occ.synchronize()

        # Mesh refinement — distance-based size field
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", [inner_tag])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", cl_cylinder)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", cl_farfield)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", farfield_radius * 0.3)

        if cl_wake is not None:
            # Box field for wake refinement
            gmsh.model.mesh.field.add("Box", 3)
            gmsh.model.mesh.field.setNumber(3, "VIn", cl_wake)
            gmsh.model.mesh.field.setNumber(3, "VOut", cl_farfield)
            gmsh.model.mesh.field.setNumber(3, "XMin", 0.0)
            gmsh.model.mesh.field.setNumber(3, "XMax", wake_length)
            gmsh.model.mesh.field.setNumber(3, "YMin", -wake_width)
            gmsh.model.mesh.field.setNumber(3, "YMax", wake_width)
            # Combine with distance field via Min
            gmsh.model.mesh.field.add("Min", 10)
            gmsh.model.mesh.field.setNumbers(10, "FieldsList", [2, 3])
            gmsh.model.mesh.field.setAsBackgroundMesh(10)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)

        # Physical groups (after mesh generation)
        wall_phys = gmsh.model.addPhysicalGroup(1, [inner_tag])
        gmsh.model.setPhysicalName(1, wall_phys, "wall")
        far_phys = gmsh.model.addPhysicalGroup(1, [outer_tag])
        gmsh.model.setPhysicalName(1, far_phys, "farfield")
        surf_phys = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, surf_phys, "fluid")

        gmsh.model.mesh.createTopology()

        out = Path(f"{name}.su2")
        gmsh.write(str(out))
        gmsh.finalize()
        return out

    @staticmethod
    def sphere_3d(
        radius: float = 0.11,
        farfield_radius: float = 3.0,
        cl_surface: float = 0.01,
        cl_farfield: float = 0.3,
        name: str = "sphere",
    ) -> Path:
        """Create a 3D sphere mesh. Returns path to .su2 mesh file.

        Uses a box farfield with a spherical hole for the ball surface.
        Mesh is graded from fine on the ball to coarse at the farfield.
        """
        import gmsh

        gmsh.initialize()
        gmsh.model.add(name)

        # Inner sphere (ball surface)
        sphere = gmsh.model.occ.addSphere(0, 0, 0, radius)

        # Outer farfield (box)
        box = gmsh.model.occ.addBox(
            -farfield_radius, -farfield_radius, -farfield_radius,
            2 * farfield_radius, 2 * farfield_radius, 2 * farfield_radius,
        )

        # Cut sphere out of box → fluid volume
        fluid_vols, _ = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
        gmsh.model.occ.synchronize()

        # Identify surfaces by bounding box:
        # sphere → all 3 dimensions ≈ 2*radius (cube-like)
        # box face → one dimension ≈ 0 (thin), the other two ≈ 2*farfield_radius
        inner_surf_tags = []
        outer_surf_tags = []
        for ent in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(ent[0], ent[1])
            dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
            # Sphere: all three spans approximately equal to 2*radius
            r_est = (dx + dy + dz) / 6  # average half-span
            if all(abs(s - 2 * radius) < 0.1 * radius for s in [dx, dy, dz]):
                inner_surf_tags.append(ent[1])
            # Box face: two spans ≈ 2*farfield_radius, one span ≈ 0
            elif (abs(dx) < 0.01 and abs(dy - 2 * farfield_radius) < 0.1 * farfield_radius
                  and abs(dz - 2 * farfield_radius) < 0.1 * farfield_radius):
                outer_surf_tags.append(ent[1])
            elif (abs(dy) < 0.01 and abs(dx - 2 * farfield_radius) < 0.1 * farfield_radius
                  and abs(dz - 2 * farfield_radius) < 0.1 * farfield_radius):
                outer_surf_tags.append(ent[1])
            elif (abs(dz) < 0.01 and abs(dx - 2 * farfield_radius) < 0.1 * farfield_radius
                  and abs(dy - 2 * farfield_radius) < 0.1 * farfield_radius):
                outer_surf_tags.append(ent[1])

        # Physical groups (after mesh generation)
        gmsh.model.mesh.generate(2)  # generate surface mesh first

        if inner_surf_tags:
            wall = gmsh.model.addPhysicalGroup(2, inner_surf_tags)
            gmsh.model.setPhysicalName(2, wall, "wall")
        if outer_surf_tags:
            far = gmsh.model.addPhysicalGroup(2, outer_surf_tags)
            gmsh.model.setPhysicalName(2, far, "farfield")

        # Volume mesh
        vols = gmsh.model.getEntities(3)
        if vols:
            vol_phys = gmsh.model.addPhysicalGroup(3, [v[1] for v in vols])
            gmsh.model.setPhysicalName(3, vol_phys, "fluid")

        # Mesh size field — refine near sphere
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", inner_surf_tags)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", cl_surface)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", cl_farfield)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", radius)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.createTopology()

        out = Path(f"{name}.su2")
        gmsh.write(str(out))
        gmsh.finalize()
        return out


# ── Solver interface ──

@dataclasses.dataclass
class SU2Results:
    """Parsed results from a SU2_CFD run."""
    cd: float = 0.0
    cl: float = 0.0
    cmz: float = 0.0
    converged: bool = False
    iterations: int = 0
    history: list[dict] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class SU2Solver:
    """Invoke SU2_CFD and collect results."""

    def __init__(self, su2_cfd: Path = SU2_CFD, workdir: Optional[Path] = None):
        self.su2_cfd = su2_cfd
        self.workdir = workdir or Path(tempfile.mkdtemp())

    def run(self, config: Path, mesh: Path, timeout: int = 600) -> SU2Results:
        """Run SU2_CFD with the given config and mesh files."""
        # Copy files to working directory
        cfg_local = self.workdir / config.name
        mesh_local = self.workdir / mesh.name
        cfg_local.write_text(config.read_text())
        mesh_local.write_text(mesh.read_text())

        # SU2 expects mesh reference in config via MESH_FILENAME
        # We inject this into the config
        cfg_text = cfg_local.read_text()
        if "MESH_FILENAME" not in cfg_text:
            cfg_text += f"\nMESH_FILENAME= {mesh_local.name}\n"
            cfg_local.write_text(cfg_text)

        cmd = [str(self.su2_cfd), str(cfg_local)]

        proc = subprocess.run(
            cmd,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return self._parse_results(proc, self.workdir)

    def _parse_results(self, proc: subprocess.CompletedProcess, workdir: Path) -> SU2Results:
        results = SU2Results()
        stdout = proc.stdout
        stderr = proc.stderr

        # Print SU2 output for debugging if there was an error
        if proc.returncode != 0:
            print(f"  [SU2 exited with code {proc.returncode}]")
            for line in (stderr or "").strip().split("\n")[-10:]:
                if line.strip():
                    print(f"  STDERR: {line.strip()}")
            return results

        # Check convergence from stdout
        results.converged = ("Convergence reached" in stdout or
                             "convergence" in stdout.lower())

        # Column name mapping (SU2 v8 uses different names than requested)
        col_map = {
            "ITER": "ITER", "Inner_Iter": "ITER", "INNER_ITER": "ITER",
            "DRAG": "DRAG", "CD": "DRAG", "cd": "DRAG",
            "LIFT": "LIFT", "CL": "LIFT", "cl": "LIFT",
            "MOMENT": "MOMENT", "CMZ": "MOMENT", "Cmz": "MOMENT",
        }

        # Parse history file
        hist_file = workdir / "history.csv"
        if hist_file.exists():
            lines = hist_file.read_text().strip().split("\n")
            if len(lines) > 1:
                header = [h.strip().strip('"') for h in lines[0].split(",")]
                for line in lines[1:]:
                    vals = line.split(",")
                    if len(vals) == len(header):
                        entry_orig = dict(zip(header, vals))
                        # Normalize column names
                        entry = {col_map.get(k, k): v for k, v in entry_orig.items()}
                        results.history.append(entry)
                        try:
                            results.iterations = int(float(entry.get("ITER", 0)))
                            results.cd = float(entry.get("DRAG", results.cd))
                            results.cl = float(entry.get("LIFT", results.cl))
                            results.cmz = float(entry.get("MOMENT", results.cmz))
                        except (ValueError, TypeError):
                            pass

        return results


# ── PyVista visualization helpers ──

def load_solution(mesh_path: Path, solution_dir: Optional[Path] = None):
    """Load SU2 mesh and solution into PyVista mesh objects."""
    import pyvista as pv
    reader = pv.get_reader(str(mesh_path))
    mesh = reader.read()
    if solution_dir:
        # Look for solution files
        for f in sorted(solution_dir.glob("*.vtu")):
            sol = pv.read(str(f))
            mesh.point_data.update(sol.point_data)
    return mesh


def plot_pressure_contours(mesh_path: Path, solution_path: Optional[Path] = None) -> None:
    """Interactive 3D plot of pressure on the sphere surface."""
    import pyvista as pv
    mesh = load_solution(mesh_path, solution_path)
    p = pv.Plotter()
    p.add_mesh(mesh, scalars="Pressure" if "Pressure" in mesh.point_data else None,
               cmap="inferno", show_edges=False, lighting=True)
    p.add_title("SU2 — Pressure Field")
    p.show()


def plot_cd_re(results_list: list[tuple[float, SU2Results]]) -> None:
    """Plot drag coefficient vs Reynolds number."""
    import matplotlib.pyplot as plt

    Re = [r[0] for r in results_list]
    cd = [r[1].cd for r in results_list]

    fig, ax = plt.subplots(facecolor="#111111")
    ax.set_facecolor("#111111")
    ax.plot(Re, cd, "o-", color="#f5a623", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Reynolds Number", color="white")
    ax.set_ylabel("Drag Coefficient $C_d$", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    plt.show()
