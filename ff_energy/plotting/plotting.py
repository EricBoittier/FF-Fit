"""
Matplotlib Style Sheet based on Science Plots
"""

import matplotlib.pyplot as plt
import scienceplots

from matplotlib import rcParams
import warnings
from pathlib import Path
import patchworklib as pw

from ff_energy.latex_writer.report import REPORTS_PATH

warnings.filterwarnings("ignore", module="matplotlib\..*")


def set_style(no_latex=False):
    if True:
        plt.style.use(["science", "no-latex"])
    # else:
    #     plt.style.use(["science", "bright"])

    rcParams["figure.figsize"] = 8, 6
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 22
    rcParams["xtick.labelsize"] = 18
    rcParams["ytick.labelsize"] = 18
    rcParams["legend.fontsize"] = 18
    # rcParams['font.family'] = 'san-serif'
    # rcParams['font.sans-serif'] = ['Arial']
    # bold title and labels
    rcParams["axes.labelweight"] = "bold"
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.linewidth"] = 2
    rcParams["xtick.major.width"] = 2
    rcParams["ytick.major.width"] = 2
    rcParams["xtick.minor.width"] = 2
    rcParams["ytick.minor.width"] = 2
    rcParams["xtick.major.size"] = 8
    rcParams["ytick.major.size"] = 8
    rcParams["xtick.minor.size"] = 4
    rcParams["ytick.minor.size"] = 4
    # set colors
    rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set1.colors)
    rcParams['font.family'] = 'sans-serif'

    # import pylab as plt
    # params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    # plt.rcParams.update(params)
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False
    # mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    # mpl.rcParams['font.family'] = 'sans-serif'
    # mpl.rcParams['font.sans-serif'] = 'cm'




def save_fig(fig, filename, path=None) -> Path:
    """
    Save the figure to the path
    :param fig:
    :param path:
    :return: path to the figure
    """
    def save_(f, filename):
        if isinstance(f, list):
            fig = patchwork_grid(f)
            fig.savefig(filename)
        else:
            f.savefig(filename, bbox_inches="tight")

    if path is not None:
        if isinstance(path, str):
            path = Path(path)
        # check if path is absolute
        if not path.is_absolute():
            path = REPORTS_PATH / path
        #  make the dir if not exists
        path.mkdir(parents=True, exist_ok=True)
        save_(fig, path / filename)
    else:
        #  save locally or to the default path
        save_(fig, filename)

    if path is None:
        return Path(filename)

    return path / filename


def patchwork_grid(axes):
    n = len(axes)
    sqrt_n = int(n ** 0.5)
    if sqrt_n ** 2 == n:
        cols = sqrt_n
    else:
        cols = sqrt_n + 1

    row_axes = [pw.stack(axes[i : i + cols], operator="|", margin=0.2)
                for i in range(0, n, cols)]
    fig = pw.stack(row_axes, operator="/", margin=0.2)

    return fig



