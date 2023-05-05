import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ff_energy.simulations.utils
from ff_energy.simulations.utils import get_rdf_peaks

time_label = "$t$ (ps)"
energy_label = "$E$ (kcal/mol)"
temp_label = "$T$ (K)"
angle_label = "Angle ($^{\circ}$)"
volume_label = "$V$ ($\AA^3$)"
font_size = 14

color_dict = {"temp": "#EA6759", "tot": "#457b9d", "elec": "#F3C65F",
              "vdw": "#8BC28C", "user": "#6667AB", "volume": "#F88F58",
              "angle": "#219ebc", "rdf": "#e63946", "hbonds": "#457b9d",
              }


def running_average(x, n=100):
    """Calculate the running average of a 1D array"""
    return np.convolve(x, np.ones((n,)) / n, mode="valid")


def plot_timeseries(ax, x, y, label=None, n=100, **kwargs):
    """Plot a timeseries with a running average"""
    # plot the timeseries
    ax.plot(x, y, label=label, **kwargs)
    # plot the running average
    if len(x) > n * 2:
        ax.plot(x[n // 2 - 2:-(n // 2 + 1)], running_average(y, n=n),
                label=f"{label} (avg)", c="k", alpha=0.5)


def plot_energy_types(df, skip=None, filename=None):
    """Plot the energy types"""

    if skip:
        # skip n picoseconds after the first timepoint
        df = df[df["time"] > skip]

    time_period = (df["time"].iloc[-1] - df["time"].iloc[0]) // 1

    dyna_name = df["dyna"].iloc[0]
    # make a 2 x 3 grid of plots
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True,
                           gridspec_kw={"hspace": 0.3, "wspace": 0.5})

    title = df["title"].iloc[0]
    suptitle = f"{title}\n {dyna_name} [{time_period} ps]"
    fig.suptitle(suptitle, fontsize=font_size, fontweight="bold")

    plot_timeseries(ax[0, 0], df["time"], df["temp"],
                    label="temperature", color=color_dict["temp"])

    ax[0, 0].title.set_text("Temperature")
    ax[0, 0].set_xlabel(time_label)
    ax[0, 0].set_ylabel(temp_label)

    plot_timeseries(ax[0, 1], df["time"], df["tot"], color=color_dict["tot"])

    ax[0, 1].title.set_text("Total")
    ax[0, 1].set_xlabel(time_label)
    ax[0, 1].set_ylabel(energy_label)

    plot_timeseries(ax[1, 0], df["time"], df["elec"], color=color_dict["elec"])

    ax[1, 0].title.set_text("Elec")
    ax[1, 0].set_xlabel(time_label)
    ax[1, 0].set_ylabel(energy_label)

    plot_timeseries(ax[1, 1], df["time"], df["vdw"], color=color_dict["vdw"])

    ax[1, 1].title.set_text("Vdw")
    ax[1, 1].set_xlabel(time_label)
    ax[1, 1].set_ylabel(energy_label)

    plot_timeseries(ax[0, 2], df["time"], df["user"], color=color_dict["user"])

    ax[0, 2].title.set_text("User")
    ax[0, 2].set_xlabel(time_label)
    ax[0, 2].set_ylabel(energy_label)

    plot_timeseries(ax[1, 2], df["time"], df["volume"], color=color_dict["volume"])

    ax[1, 2].title.set_text("Volume")
    ax[1, 2].set_xlabel(time_label)
    ax[1, 2].set_ylabel(volume_label)

    return fig, ax


def plot_fluctuations(df, skip=None, filename=None):
    """Plot the fluctuations"""

    if skip:
        # skip n picoseconds after the first timepoint
        df = df[df["time"] > df["time"].iloc[0] + skip]

    dyna_name = df["dyna"].iloc[0]

    time_period = (df["time"].iloc[-1] - df["time"].iloc[0]) // 1

    # make a 2 x 3 grid of plots
    fig, ax = plt.subplots(2, 3, figsize=(12, 8),
                           gridspec_kw={"hspace": 0.5, "wspace": 0.5})

    title = df["title"].iloc[0]
    suptitle = f"{title}\n{dyna_name} [{time_period} ps]"
    fig.suptitle(suptitle, fontsize=font_size, fontweight="bold")

    # plot the energy types
    ax[0, 0].hist(df["temp"] - df["temp"].mean(), label="temperature",
                  color=color_dict["temp"])
    ax[0, 0].title.set_text("T [{:.0f} K]".format(df["temp"].mean()))
    # ax[0,0].set_xlabel(time_label)
    ax[0, 0].set_xlabel(temp_label)

    ax[0, 1].hist(df["tot"] - df["tot"].mean(), label="total", color=color_dict["tot"])
    ax[0, 1].title.set_text("Total [{:.0f} kcal/mol]".format(df["tot"].mean()))
    # ax[0,1].set_xlabel(time_label)
    ax[0, 1].set_xlabel(energy_label)

    ax[1, 0].hist(df["elec"] - df["elec"].mean(), label="elec",
                  color=color_dict["elec"])
    ax[1, 0].title.set_text("Elec [{:.0f} kcal/mol]".format(df["elec"].mean()))
    # ax[1,0].set_xlabel(time_label)
    ax[1, 0].set_xlabel(energy_label)

    ax[1, 1].hist(df["vdw"] - df["vdw"].mean(), label="vdw", color=color_dict["vdw"])
    ax[1, 1].title.set_text("Vdw [{:.0f} kcal/mol]".format(df["vdw"].mean()))
    # ax[1,1].set_xlabel(time_label)
    ax[1, 1].set_xlabel(energy_label)

    ax[0, 2].hist(df["user"] - df["user"].mean(), label="user",
                  color=color_dict["user"])
    ax[0, 2].title.set_text("User [{:.0f} kcal/mol]".format(df["user"].mean()))
    # ax[0,2].set_xlabel(time_label)
    ax[0, 2].set_xlabel(energy_label)

    ax[1, 2].hist(df["volume"] - df["volume"].mean(), label="volume",
                  color=color_dict["volume"])
    ax[1, 2].title.set_text("Volume [{:.0f}".format(df["volume"].mean()) + "$\AA^{3}$]")
    # ax[1,2].set_xlabel(time_label)
    ax[1, 2].set_xlabel(volume_label)

    return fig, ax


def make_plots(df, save=False, skips=None, psf="water.2000.psf", ind=False):
    dyna_keys = list(set(df["dyna"]))
    # sort the dyna keys by the first number in the string
    print("plotting")
    print(dyna_keys)
    dyna_keys.sort(key=lambda x: int(x.split(":")[0]))
    # set the title
    title = df["title"].iloc[0]

    if ind:
        dyna_keys = [dyna_keys[ind]]

    if skips is None:
        skips = [None] * len(dyna_keys)

    plot_filenames = []

    for i, dyna_key in enumerate(dyna_keys):

        if ind:
            i = ind

        df_ = df[df["dyna"] == dyna_key]
        # df_["dcd"].iloc[0]
        os.path.dirname(df_["path"].iloc[0])

        #  Plot energy components
        fig, ax = plot_energy_types(df_, skip=skips[i])
        if save:
            plot_filename = f"/home/boittier/Documents/simreports/" \
                            f"figures/figs/{title}_{i}_energy_types.pdf"
            fig.savefig(plot_filename, dpi=300)
            plot_filenames.append(trim_filename(plot_filename))

        # Plot fluctuations of energy components
        fig, ax = plot_fluctuations(df_, skip=skips[i])
        if save:
            plot_filename = f"/home/boittier/Documents/" \
                            f"simreports/figures/figs/{title}_{i}_fluctuations.pdf"
            fig.savefig(plot_filename, dpi=300)
            plot_filenames.append(trim_filename(plot_filename))


def plot_rdf(ax, u, title=False, step=100):
    """Plot the rdf"""
    # get the rdf
    irdf = ff_energy.simulations.utils.g_rdf(u, step=step)
    peaks = get_rdf_peaks(irdf)

    # plot the rdf
    p = peaks * (irdf.results.bins[1] - irdf.results.bins[0] + 0.005)
    ax.plot(irdf.results.bins, irdf.results.rdf, c="k")
    ax.plot(p, irdf.results.rdf[peaks], "x", c=color_dict["angle"])

    ax.set_xlabel("r ($\AA$)", fontsize=font_size)
    ax.set_ylabel("g(r)", fontsize=font_size)
    ax.set_title("RDF", fontsize=font_size, fontweight="bold")

    if title:
        odf = pd.DataFrame(irdf.results.rdf, index=irdf.results.bins)
        odf.to_csv(f"{title}_rdf.csv")

    return ax


def trim_filename(path, last=2):
    """Trim the filename to the last n directories"""
    splitted = path.split("/")
    return "/".join(splitted[-last:])
