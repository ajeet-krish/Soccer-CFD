"""Build all simulations — orchestrator."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shot_aero import run_cylinder, animate_velocity_comparison, animate_pressure_comparison
from src.tactical import run_tactical, run_macro_stress, run_team_heatmaps, run_formation_fluid_arrays, run_lateral_shift, run_up_back_through, run_rapid_circulation
from phi.jax.flow import Obstacle, vec

from src.domain import CYLINDER
from src.utils import IMAGES


def main():
    (IMAGES / "phiflow_cylinder_2d").mkdir(parents=True, exist_ok=True)
    (IMAGES / "tactical").mkdir(parents=True, exist_ok=True)

    # ── Shot Aerodynamics ──
    print("=" * 60)
    print("SHOT AERODYNAMICS — Magnus vs Knuckleball")
    print("=" * 60)
    obstacle_spin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=10.0)
    obstacle_nospin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=0.0)

    print("\nMagnus (ω = 10 rad/s)...")
    frames_magnus = run_cylinder(obstacle_spin, "Magnus")

    print("\nKnuckleball (ω = 0 rad/s)...")
    frames_knuckle = run_cylinder(obstacle_nospin, "Knuckleball")

    print(f"\nCaptured {len(frames_magnus)} frames each")
    animate_velocity_comparison(frames_magnus, frames_knuckle)
    animate_pressure_comparison(frames_magnus, frames_knuckle)

    # ── Tactical Positioning ──
    print("\n" + "=" * 60)
    print("TACTICAL POSITIONING — Formation Aerodynamics")
    print("=" * 60)
    run_tactical()

    # ── Macro Stress Field ──
    print("\n" + "=" * 60)
    print("MACRO STRESS FIELD — 7v7 Tactical Continuum")
    print("=" * 60)
    run_macro_stress()

    # ── Team Heatmaps ──
    print("\n" + "=" * 60)
    print("TEAM HEATMAPS — 11v11 Defensive Influence & Passing Lanes")
    print("=" * 60)
    run_team_heatmaps()

    # ── Formations as Fluid Arrays ──
    print("\n" + "=" * 60)
    print("FORMATIONS AS STRUCTURAL FLUID ARRAYS")
    print("=" * 60)
    run_formation_fluid_arrays()

    # ── Lateral Shifting Mechanics ──
    print("\n" + "=" * 60)
    print("LATERAL SHIFTING MECHANICS")
    print("=" * 60)
    run_lateral_shift()

    # ── Up-Back-Through ──
    print("\n" + "=" * 60)
    print("UP-BACK-THROUGH — TRANSIENT CAVITATION")
    print("=" * 60)
    run_up_back_through()

    # ── Rapid Circulation ──
    print("\n" + "=" * 60)
    print("RAPID CIRCULATION — VISCOELASTIC RELAXATION")
    print("=" * 60)
    run_rapid_circulation()

    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 60)
    for sub in ["phiflow_cylinder_2d", "tactical"]:
        for f in sorted((IMAGES / sub).iterdir()):
            print(f"  {sub}/{f.name}")


if __name__ == "__main__":
    main()
