"""Shot aerodynamics domain — cylinder in crossflow."""

from phi.jax.flow import *


NX, NY = 256, 128
BOX = geom.Box(x=8.0, y=4.0)
BOUNDARY = {
    "x-": vec(x=1.0, y=0.0),
    "x+": ZERO_GRADIENT,
    "y": 0,
}

CENTER = (2.0, 2.0)
RADIUS = 0.3
CYLINDER = geom.Sphere(x=2.0, y=2.0, radius=RADIUS)

CENTERED = CenteredGrid(0, x=NX, y=NY, bounds=BOX)

DT = 0.3
N_STEPS = 300
SAVE_EVERY = 3
