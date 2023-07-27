import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
from pathlib import Path
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap
from plotnine.data import mtcars

from ff_energy import KernelFit

from ff_energy import get_cclib_data
import numpy as np

from matplotlib import ticker
from ase import Atoms
import pandas as pd
import numpy as np
import pandas.api.types as pdtypes

RMSELABEL = 'RMSE [(kcal/mol)/$e$]'
ALPHA = "$\\alpha$"
LAMBDA = "$\lambda$"

from plotnine import (
    ggplot,
    aes,
    stage,
    geom_violin,
    geom_point,
    geom_line,
    geom_boxplot,
    scale_fill_manual,
    theme,
    theme_classic,
    scale_alpha,
    scale_color_cmap,
    scale_color_gradient,
    theme_minimal,
    theme_tufte,
    theme_xkcd,
    theme_void,
    geom_jitter,
    labs,
    ggtitle
)

import patchworklib as pw

import seaborn as sns

PAL = sns.color_palette("pastel")


def values_to_colors(values, cmap="viridis", lim=None):
    if lim is None:
        lim = [min(values), max(values)]
    norm = mpl.colors.Normalize(vmin=lim[0], vmax=lim[1])
    cmap = cm.get_cmap(cmap)
    return cmap(norm(values)), cm.ScalarMappable(norm=norm, cmap=cmap)


def orthographic_plot(x, y, z,
                      ax=None,
                      c="k",
                      plot=False,
                      s=1,
                      alpha=None,
                      ):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=0)  # Set the elevation and
    # azimuth angles for orthographic projection

    if plot:
        ax.plot(x, y, z, c=c, linewidth=0.1)
    else:
        ax.scatter(x, y, z, c=c, s=s, alpha=alpha)  # Plot the data points

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax


def prepare_standard(x: pd.DataFrame, csv_dir: str):
    csv_files = list(Path(csv_dir).glob("csvs/*.csv"))
    data_path = Path("/home/boittier/Documents/phd/ff_energy/cubes/dcm")
    if "pkl" in x.columns and "filename" not in x.columns:
        x["filename"] = x["pkl"]
    pkl_filenames = x["filename"]
    standard = x

    tmp = [Path(_).stem.strip("_esp") + ".out" for _ in pkl_filenames]
    filenames = [
        data_path / "nms" / _ if not (_.startswith("gaus")) else data_path / "scan" / _
        for _ in tmp]

    if "rmses" in x.columns and "rmse" not in x.columns:
        x["rmse"] = x["rmses"]
    standard["rmse_norm"] = standard["rmse"] / (standard["rmse"].max()) * 75
    data = [get_cclib_data(_) for _ in filenames]
    moments = [_.moments for _ in data]
    dipoles = [_[1] for _ in moments]
    m_dip = [np.linalg.norm(_) for _ in dipoles]
    pkl_path = Path("/home/boittier/Documents/phd/ff_energy/ff_energy/pydcm/tests")
    tmp = [Path(_).stem.strip("_esp") + ".out" for _ in pkl_filenames]
    picklenames = [
        pkl_path / "nms" / _ if not (_.startswith("gaus")) else pkl_path / "scan" / _
        for _ in tmp]
    pickles_data = [pd.read_pickle(pkl_path / pklfn) for pklfn in pkl_filenames]
    ase_data = [Atoms(numbers=_.atomnos, positions=_.atomcoords[0]) for _ in data]
    a1 = [_.get_angle(1, 0, 2) for _ in ase_data]
    a2 = [_.get_angle(1, 0, 2) for _ in ase_data]
    headers = []
    for i in range(len(pickles_data[0])):
        IDX = i // 4
        fstr = None
        if (i - IDX * 4) == 0:
            fstr = f"c{IDX + 1}a"
        if (i - IDX * 4) == 1:
            fstr = f"c{IDX + 1}b"
        if (i - IDX * 4) == 2:
            fstr = f"c{IDX + 1}c"
        if (i - IDX * 4) == 3:
            fstr = f"c{IDX + 1}q"
        headers.append(fstr)

    pkl_df = pd.DataFrame(pickles_data, columns=headers)

    #  join the dataframes
    standard = standard.join(pkl_df)
    standard["dip"] = m_dip
    standard["a102"] = a1
    return standard, headers


def get_change_graph(row, csv_dict, standard, pca_df):
    a, row = row
    k1 = row[0]
    k2 = row[1]
    a = row[5]
    n = row[2]
    rmse = row[3]
    rmse2 = row[4]
    l = row[6]
    return plot_change2(csv_dict[k1],
                        csv_dict[k2],
                        standard,
                        pca_df,
                        title=f"\n$\\alpha = $ {a} | $\lambda = $ {l:.1f} | RMSE = {rmse:.2f} | $n$ = {n}")


def get_brick(row, csv_dict, standard, pca_df):
    return pw.load_ggplot(get_change_graph(row), figsize=(4, 4))


def plot_hists(standard, headers, k, color_key="rmses"):
    #  plot a figure
    fig, ax = plt.subplots(len(headers) // 4, 4, figsize=(9, 9),
                           gridspec_kw={"hspace": 0.5,
                                        "wspace": 0.25})

    plt.suptitle(f"{k}")
    standard[color_key + "_norm"] = standard[color_key] / (
        standard[color_key].max())  # * 75

    for i, _ in enumerate(headers):
        ii = i // 4
        jj = i % 4
        if jj == 3:
            ax[ii][jj].set_xlim(-1.1, 1.1)
            ax[ii][jj].text(0.15, .65, f"{_} = {standard[_].mean():.2f}",
                            transform=ax[ii][jj].transAxes)
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax[ii][jj])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(s, cax=cax)

        else:
            ax[ii][jj].text(0.8, .65, _, transform=ax[ii][jj].transAxes)
            s = ax[ii][jj].scatter(standard[_], standard["rmse_norm"], alpha=0.5,
                                   c=standard[color_key], s=2)
            ax[ii][jj].hist(standard[_], color="gray", bins=20, density=True)

        ax[ii][jj].set_xticklabels(ax[ii][jj].get_xticks(), rotation=0, fontsize=10)
        ax[ii][jj].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    return plt.gcf()


def plot_angle(standard):
    fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    plt.subplots_adjust(wspace=0, )
    standard["cut_a102"] = pd.cut(standard["a102"], bins=10)
    standard["cut_a102"] = standard["cut_a102"].apply(
        lambda x: 0.5 * (x.left + x.right))
    sns.lineplot(data=standard, x="cut_a102", y="rmse", ax=axs[0])
    sns.scatterplot(data=standard, x="a102", y="rmse", ax=axs[0])
    axs[0].set_ylim(0, 2)
    axs[1].hist(standard["rmse"], orientation='horizontal')


def plot_change2(test, opt, standard, pca_df, title=""):
    """
    """
    pca_df["cosin_angle"] = np.arctan2(abs(pca_df[0]), abs(pca_df[1]))

    delta_rmse = test["rmse"] - standard["rmses"]

    _ = test[["rmse", "class", "pkl"]]
    _["when"] = "k-MDCM"
    _["ca"] = pca_df["cosin_angle"]
    _["drmse"] = abs(delta_rmse)

    b = standard[["rmses"]]
    b["rmse"] = standard["rmses"]
    b["class"] = _["class"]
    b["pkl"] = _["pkl"]
    b["when"] = "MDCM"
    b["ca"] = pca_df["cosin_angle"]
    b["drmse"] = abs(delta_rmse)

    c = opt[["rmse"]]
    c["class"] = _["class"]
    c["pkl"] = _["pkl"]
    c["when"] = "Opt."
    c["ca"] = pca_df["cosin_angle"]
    c["drmse"] = abs(delta_rmse)

    comb = _.append(b, ignore_index="true")
    comb = comb.append(c, ignore_index="true")
    comb = comb[comb["class"] == "test"]

    _ = (ggplot(comb, aes('when', 'rmse'))
         + geom_violin(comb, style='full')  # changed
         + geom_line(aes(group='pkl', color='ca', alpha="drmse"))  # new
         + theme_minimal()
         + ggtitle(title)
         + scale_alpha(range=(0.0, 0.5), name="$\Delta$RMSE", show_legend=False)
         + scale_color_cmap("PiYG", name="$cos^{-1}(\mathbf{PCA})$")
         + theme(figure_size=(4, 4))
         + labs(y=RMSELABEL,
                x="Optimization Process")
         + aes(ymin=0, ymax=1.5)
         )
    return _


def rmse_plot(ds_paired, standard, title=False, ax=None):
    if ax is None:
        ax = plt.gca()
    lp = sns.lineplot(data=ds_paired,
                      x=LAMBDA, y="rmse_kernel", markers=True, c="gray", ax=ax,
                      hue=ALPHA, palette="pastel")
    ax.axhline(standard.rmses.median(), c="k", linestyle="--")
    ax.axhline(ds_paired.rmse_opt.median(), c="gray", linestyle="--")
    # ax.set_ylim(0, 0.82)
    ax.set_xlabel(LAMBDA, fontsize=20)
    ax.set_ylabel(RMSELABEL, fontsize=20)
    # plt.tight_layout()
    # LABELS = [f"{x:.1e}" for x in ds_paired[ALPHA].unique()]
    # check axes and find which has a legend
    leg = lp.axes.get_legend()

    for t in leg.texts:
        t.set_text(f"{float(t.get_text()):.1e}")
    # plt.legend(title=ALPHA, loc='upper right', labels=LABELS)

    if title:
        ax.set_title(title)
    plt.savefig("median_RMSE_summary.pdf", bbox_inches="tight")
    return ax


def make_mol_plot(XYZs,
                  cXYZ,
                  colours,
                  atom_colors=None,
                  CMAP=None,
                  NORM=None,
                  ax=None,
                  save_name=None):
    colors = None

    if CMAP is not None and NORM is not None:
        colors = CMAP([NORM(_) for _ in colours])

    if atom_colors is None:
        atom_colors = ["red", "k", "gray", "gray", "gray", "gray"]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for i in range(180):
        orthographic_plot(XYZs[0, i, :],
                          XYZs[1, i, :],
                          XYZs[2, i, :],
                          ax=ax, c=atom_colors)

    if colors is None:
        colors = "k"

    for i in range(10):
        if CMAP is None:
            _C = PAL[i] if i < 7 else "gray"
        else:
            _C = colors

        if i < 7:
            orthographic_plot(cXYZ[0, :, i],
                              cXYZ[1, :, i],
                              cXYZ[2, :, i],
                              ax=ax, c=_C)
    plt.grid(b=None)
    ax.grid(False)

    bondsx = []
    bondsy = []
    bondsz = []
    bond_idxs = [(0, 2), (0, 1), (1, 3), (1, 4), (1, 5)]

    for a, b in bond_idxs:
        bondsx.append(XYZs[0, 80, a])
        bondsx.append(XYZs[0, 80, b])
        bondsy.append(XYZs[1, 80, a])
        bondsy.append(XYZs[1, 80, b])
        bondsz.append(XYZs[2, 80, a])
        bondsz.append(XYZs[2, 80, b])

    orthographic_plot(bondsx,
                      bondsy,
                      bondsz,
                      ax=ax,
                      c="k",
                      plot=True)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_zlabel("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    # tighten the figure
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
