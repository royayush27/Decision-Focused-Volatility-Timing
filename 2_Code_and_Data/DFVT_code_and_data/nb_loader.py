"""Run the code cells of the final notebook in this process, so the extra scripts reuse its data, forecasts and
backtests exactly (no copy of the analysis code).

paper_style=True redraws every notebook figure for the report: Times-like serif font, no figure-wide titles (the
report's captions carry them), slightly larger type so that it stays legible at page width, and output to
figures_paper/ instead of figures/."""
import io, json
from contextlib import redirect_stdout
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

NOTEBOOK = "vol_timing_final_revised.ipynb"
PAPER_DIR = Path("figures_paper")
PAPER_RC = {"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix", "font.size": 11.5, "axes.titlesize": 11.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.titleweight": "normal", "savefig.dpi": 200}
SHRINK = 1.0   # canvas scale (1 = the notebook's own figure sizes)


def _paper_style():
    plt.rcParams.update(PAPER_RC)
    Figure.suptitle = lambda self, *a, **k: None
    sub, fig = plt.subplots, plt.figure

    def subplots(*a, figsize=None, **k):
        return sub(*a, figsize=None if figsize is None else (figsize[0] * SHRINK, figsize[1] * SHRINK), **k)

    def figure(*a, figsize=None, **k):
        return fig(*a, figsize=None if figsize is None else (figsize[0] * SHRINK, figsize[1] * SHRINK), **k)
    plt.subplots, plt.figure = subplots, figure


def load_notebook(path=NOTEBOOK, paper_style=False):
    plt.show = lambda *a, **k: plt.close("all")
    ns = {"display": lambda *a, **k: None}
    first = True
    for cell in json.load(open(path, encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        with redirect_stdout(io.StringIO()):
            exec("".join(cell["source"]), ns)
        if first and paper_style:            # the first cell sets the notebook's own style and figure folder
            _paper_style()
            PAPER_DIR.mkdir(exist_ok=True)
            ns["FIGURES"] = PAPER_DIR
        first = False
    return ns
