import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import patchworklib as pw
from pathlib import Path

from ff_energy.plotting.ffe_plots import plot_energy_MSE
from ff_energy.latex_writer.report import REPORTS_PATH
from ff_energy.plotting.plotting import set_style, save_fig, patchwork_grid
from ff_energy.projections.plotting import plot_values_of_l


import numpy as np
from matplotlib import ticker

yfmt = ticker.ScalarFormatter(useMathText=True)
#  set the plot style
set_style()
plt.set_loglevel("notset")

labels = {
    "intE": "$\Delta H$ [kcal/mol]",
    "P_intE": "$\Delta H_{\mathrm{pairs}}$ [kcal/mol]",
    "C_ENERGY": "$\Delta H_{\mathrm{cluster}}$ [Hartree]",
    "M_ENERGY": "$\Delta H_{\mathrm{monomers}}$ [Hartree]",
}
char = "_"
emptystr = ""


class DataPlots:
    def __init__(self, data):
        self.data = data.data
        self.obj = data
        self.ase = None

    def energy_hist(self, path=None, key1="intE", key2="P_intE", label="") -> dict:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        #  hist plot
        ax.hist(self.data[key1], bins=100)
        ax.hist(self.data[key2], bins=100)
        ax.set_xlabel(labels[key1])
        ax.set_ylabel("Count")
        ax.legend([labels[key1], labels[key2]], ncol=1, fontsize=10)
        ax.xaxis.tick_top()
        #  mse plot
        ax = fig.add_subplot(222)
        plot_energy_MSE(
            self.data,
            "intE",
            "P_intE",
            elec="intE",
            ax=ax,
            xlabel=labels[key1],
            ylabel=labels[key2],
        )
        #  save file
        outpath = save_fig(fig, f"{key1}_{key2}.pdf", path=path)
        #  clear the figure
        plt.clf()
        out_dict = {
            "path": outpath,
            "caption": f"Analysis of {labels[key1].replace(char, emptystr)} and "
            f"{labels[key2].replace(char, emptystr)}",
            "label": f"{label}_{key1}_{key2}",
        }
        return out_dict

    def hist_kde(self, keys, path=None, label="") -> dict:

        n_keys = len(keys)
        axes = [pw.Brick(figsize=(3,2)) for _ in range(n_keys)]
        for i, key in enumerate(keys):
            ax = axes[i]
            snshist = sns.histplot(data=self.data, x=key, kde=True, ax=ax)
            #  set the x axis to scientific notation
            ax.xaxis.set_major_formatter(yfmt)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.get_xaxis().get_offset_text().set_visible(False)
            ax_max = max(ax.get_yticks())
            exponent_axis = np.floor(np.log10(ax_max)).astype(int)
            ax.annotate(r'$\times$10$^{%i}$' % (exponent_axis),
                         xy=(.01, .96), xycoords='axes fraction')

        #  save file
        key = key.replace(char, emptystr)
        outpath = save_fig(axes, f"{key}.pdf", path=path)
        #  clear the figure
        plt.clf()
        __ = " ".join([label, *key]  )
        out_dict = {
            "path": outpath,
            "caption": f"Distribution of {__}",
            "label": f"{label}singlekey",
        }
        return out_dict

    def structure_plot(self, function):
        self.ase = [s.get_ase() for s in self.obj.structures]
        return plot_values_of_l(function, self.ase)




