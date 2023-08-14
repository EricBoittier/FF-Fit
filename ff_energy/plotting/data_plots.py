import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ff_energy.plotting.ffe_plots import plot_energy_MSE
from ff_energy.latex_writer.report import REPORTS_PATH
from ff_energy.plotting.plotting import set_style, save_fig

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

    def energy_hist(self, path=None, key1="intE", key2="P_intE", label="") -> dict:
        fig = plt.figure()
        ax = fig.add_subplot(221)
        #  hist plot
        ax.hist(self.data[key1], bins=100)
        ax.hist(self.data[key2], bins=100)
        ax.set_xlabel(labels[key1])
        ax.set_ylabel("Count")
        ax.legend([labels[key1], labels[key2]],
                  ncol=1, fontsize=10)
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

    def hist_kde(self, key, path=None, label="") -> dict:
        _ = sns.histplot(data=self.data, x=key, kde=True)
        plt.xlabel(labels[key])
        fig = plt.gcf()

        #  save file
        key = key.replace(char, emptystr)
        outpath = save_fig(fig, f"{key}.pdf", path=path)
        #  clear the figure
        plt.clf()

        out_dict = {
            "path": outpath,
            "caption": f"Distribution of {key}",
            "label": f"{label}_{key}",
        }
        return out_dict
