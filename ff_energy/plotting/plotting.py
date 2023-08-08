"""
Matplotlib Style Sheet based on Science Plots
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
from pathlib import Path

from ff_energy.latex_writer.report import REPORTS_PATH

warnings.filterwarnings("ignore",
                        module="matplotlib\..*"
                        )


def set_style():
    plt.style.use('science')
    rcParams['figure.figsize'] = 8, 6
    rcParams['axes.labelsize'] = 18
    rcParams['axes.titlesize'] = 22
    rcParams['xtick.labelsize'] = 18
    rcParams['ytick.labelsize'] = 18
    rcParams['legend.fontsize'] = 18
    # rcParams['font.family'] = 'san-serif'
    # rcParams['font.sans-serif'] = ['Arial']
    # bold title and labels
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'
    rcParams['axes.linewidth'] = 2
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.minor.width'] = 2
    # set colors
    rcParams['axes.prop_cycle'] = plt.cycler(
        color=plt.cm.Dark2.colors)


def save_fig(fig, filename, path=None):
    """
    Save the figure to the path
    :param fig:
    :param path:
    :return:
    """
    if path is not None:
        if isinstance(path, str):
            path = Path(path)
        # check if path is absolute
        if not path.is_absolute():
            path = REPORTS_PATH / path
        #  make the dir if not exists
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / filename)
