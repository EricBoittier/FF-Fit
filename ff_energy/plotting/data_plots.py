import matplotlib.pyplot as plt
from pathlib import Path

from ff_energy.plotting.ffe_plots import plot_energy_MSE
from ff_energy.latex_writer.report import REPORTS_PATH
from ff_energy.plotting.plotting import set_style, save_fig

#  set the plot style
set_style()


class DataPlots:
    def __init__(self, data):
        self.data = data.data
        self.obj = data

    def energy_hist(self, path=None):
        fig = plt.figure()
        ax = fig.add_subplot(221)
        ax.hist(self.data["intE"], bins=100)
        ax.hist(self.data["P_intE"], bins=100)
        ax.set_xlabel("intE [kcal/mol]")
        ax.set_ylabel("count")
        ax.legend(["intE", "P_intE"])
        #  mse plot
        ax = fig.add_subplot(222)
        plot_energy_MSE(
            self.data,
            "intE",
            "P_intE",
            elec="M_ENERGY",
            ax=ax,
        )
        #  save file
        save_fig(fig, "energy_hist.pdf", path=path)

        return fig, ax
