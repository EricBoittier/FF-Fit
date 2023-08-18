import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import patchworklib as pw
from pathlib import Path

from ff_energy.plotting.ffe_plots import plot_energy_MSE
from ff_energy.latex_writer.report import REPORTS_PATH
from ff_energy.plotting.plotting import set_style, save_fig, patchwork_grid
from ff_energy.plotting.pov_ray import render_povray
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
latexuscore = r"\_"

rots = [
    '0x, 0y, 0z',
    '90x, 0y, 0z',
    '0x, 90y, 0z',
    '0x, 0y, 90z',
    '90x, 90y, 0z',
    '90x, 0y, 90z',
    '0x, 90y, 90z',
    '90x, 90y, 90z',
]



class DataPlots:
    def __init__(self, data):
        self.data = data.data
        self.obj = data
        self.ase = [
            self.obj.structure_key_pairs[s].get_ase() for s in self.data.index
        ]
        self.ase_dict = {
            s: self.obj.structure_key_pairs[s].get_ase() for s in self.data.index
        }

    def get_colors(self, key) -> list:
        """ returns the normalized colors for the given key

        :param key:
        :return:
        """
        values = self.data[key]
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        cmap = plt.cm.plasma
        colors = cmap(norm(values))
        return colors

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
        """
        Plot the histogram and kde of the given keys
        :param keys:
        :param path:
        :param label:
        :return:
        """
        n_keys = len(keys)
        axes = [pw.Brick(figsize=(3, 2)) for _ in range(n_keys)]
        for i, key in enumerate(keys):
            ax = axes[i]
            _ = sns.histplot(data=self.data, x=key, kde=True, ax=ax)
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
        __ = " ".join([label, *key])
        out_dict = {
            "path": outpath,
            "caption": f"Distribution of {__}",
            "label": f"{label}singlekey",
        }
        return out_dict

    def structure_plot(self, function, c=False):
        """
        Plot the projects of the structures
        :param function:
        :param c:
        :return:
        """
        if c:
            cs = [self.get_colors(_) for _ in ["intE", "P_intE", "M_ENERGY"]]
            return plot_values_of_l(function, self.ase, c=cs)
        else:
            return plot_values_of_l(function, self.ase)

    def mol_pics(self, keys, path, label):
        """
        Plot the molecules
        :return:
        """
        label = ("".join(label.split()[:2])).strip().strip("_").strip("/")
        filenames = []
        descriptions = []
        _labels = []
        values = []

        for ki, key in enumerate(keys):
            #  highest
            id = self.data[key].sort_values().index[-1]
            min_ = self.ase_dict[id]
            render_povray(min_,
                          path / f"{label}{key}min.pov",
                          rotation=rots[ki]
                          )
            filenames.append(path / f"{label}{key}min.png")
            _labels.append(f"{label}{key}min")
            v = self.data[key].sort_values()[-1]
            values.append(v)
            descriptions.append(f"Min: {v:.2f} \n {labels[key]} "
                                f"\n (id: {id.replace('_', latexuscore)})")

            #  middle
            id = self.data[key].sort_values().index[len(self.data)//2]
            mid_ = self.ase_dict[id]
            render_povray(mid_,
                            path / f"{label}{key}mid.pov",
                            rotation=rots[ki]
                            )
            filenames.append(path / f"{label}{key}mid.png")
            _labels.append(f"{label}{key}mid")
            v = self.data[key].sort_values()[len(self.data)//2]
            values.append(v)
            descriptions.append(f"Mid: {v:.2f} \n {labels[key]} "
                                f"\n (id: {id.replace('_', latexuscore)})")


            #  lowest
            id = self.data[key].sort_values().index[0]
            max_ = self.ase_dict[id]
            render_povray(max_,
                          path / f"{label}{key}max.pov",
                          rotation=rots[ki]
                          )
            filenames.append(path / f"{label}{key}max.png")
            _labels.append(f"{label}{key}max")
            v = self.data[key].sort_values()[0]
            values.append(v)
            descriptions.append(f"Max. {v:.2f} \n {labels[key]}"
                                f" \n (id: {id.replace('_', latexuscore)})")

        #  join all the pngs into one figure with patchworklib
        fig, axes = plt.subplots(len(keys), 3, figsize=(15, len(keys)*5))
        for i, f in enumerate(filenames):
            ax = axes[i // 3, i % 3]
            ax.imshow(plt.imread(f))
            ax.axis("off")
            ax.set_title(descriptions[i])

        outpath = save_fig(fig, f"{label}mols.png", path=path)
        plt.show()
        plt.clf()
        filenames = [outpath]
        descriptions = [f"{label}"]
        _labels = [f"{label}mols"]
        values = [0]

        return filenames, descriptions, _labels, values
