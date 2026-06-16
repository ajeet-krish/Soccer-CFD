from phi.jax.flow import *
from phi import geom, math, field

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Ellipse
from pathlib import Path
from tqdm import trange


ASSETS = Path("assets")
IMAGES = Path("docs/images")


def setup_theme():
    plt.style.use("dark_background")
    matplotlib.rcParams.update({
        "figure.facecolor": "#111111",
        "axes.facecolor": "#111111",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
    })
    ASSETS.mkdir(exist_ok=True)
