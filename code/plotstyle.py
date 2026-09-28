"""Shared matplotlib style for all thesis figures (Times-like font, decimal comma)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

_GYRE = "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/tex-gyre"
for _f in ("regular", "bold", "italic", "bolditalic"):
    _p = os.path.join(_GYRE, f"texgyretermes-{_f}.otf")
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)

# Colors: first three slots of a CVD-validated categorical palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_TINT, ORANGE_TINT = "#dbe9fb", "#fbe3d8"
BLUE_TINT2 = "#9fc4ee"              # mid tint of BLUE for secondary lines
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

TEXTWIDTH_IN = 15.5 / 2.54          # A4, margins 3.5 cm / 2 cm

plt.rcParams.update({
    "font.family": "TeX Gyre Termes",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
})


def _fmt(v, _pos=None):
    s = f"{v:g}"
    return s.replace(".", ",").replace("-", "−")


COMMA = FuncFormatter(_fmt)


def comma_axes(ax, x=True, y=True):
    """Vietnamese decimal comma on the tick labels."""
    if x:
        ax.xaxis.set_major_formatter(COMMA)
    if y:
        ax.yaxis.set_major_formatter(COMMA)


def unit_square(ax, ticks=(0, 0.5, 1), labels=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    comma_axes(ax)
    if labels:
        ax.set_xlabel(r"$x(1)$")
        ax.set_ylabel(r"$x(2)$", labelpad=1)


def save(fig, name):
    out = os.path.join(os.path.dirname(__file__), "..", "de_an", "figures", name)
    fig.savefig(out)
    plt.close(fig)
    print("saved", os.path.normpath(out))
