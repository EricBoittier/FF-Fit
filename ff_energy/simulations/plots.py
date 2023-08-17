import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import ff_energy
from ff_energy import get_rdf_peaks

time_label = "$t$ (ps)"
energy_label = "$E$ (kcal/mol)"
temp_label = "$T$ (K)"
angle_label = "Angle ($^{\circ}$)"
volume_label = "$V$ ($\AA^3$)"
font_size = 14

color_dict = {
    "temp": "#EA6759",
    "tot": "#457b9d",
    "elec": "#F3C65F",
    "vdw": "#8BC28C",
    "user": "#6667AB",
    "volume": "#F88F58",
    "angle": "#219ebc",
    "rdf": "#e63946",
    "hbonds": "#457b9d",
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
        ax.plot(
            x[n // 2 - 2 : -(n // 2 + 1)],
            running_average(y, n=n),
            label=f"{label} (avg)",
            c="k",
            alpha=0.5,
        )


def plot_energy_types(df, skip=None, filename=None):
    """Plot the energy types"""

    if skip:
        # skip n picoseconds after the first timepoint
        df = df[df["time"] > skip]

    time_period = (df["time"].iloc[-1] - df["time"].iloc[0]) // 1

    dyna_name = df["dyna"].iloc[0]
    # make a 2 x 3 grid of plots
    fig, ax = plt.subplots(
        2, 3, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.3, "wspace": 0.5}
    )

    title = df["title"].iloc[0]
    suptitle = f"{title}\n {dyna_name} [{time_period} ps]"
    fig.suptitle(suptitle, fontsize=font_size, fontweight="bold")

    plot_timeseries(
        ax[0, 0], df["time"], df["temp"], label="temperature", color=color_dict["temp"]
    )

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
    fig, ax = plt.subplots(
        2, 3, figsize=(12, 8), gridspec_kw={"hspace": 0.5, "wspace": 0.5}
    )

    title = df["title"].iloc[0]
    suptitle = f"{title}\n{dyna_name} [{time_period} ps]"
    fig.suptitle(suptitle, fontsize=font_size, fontweight="bold")

    # plot the energy types
    ax[0, 0].hist(
        df["temp"] - df["temp"].mean(), label="temperature", color=color_dict["temp"]
    )
    ax[0, 0].title.set_text("T [{:.0f} K]".format(df["temp"].mean()))
    # ax[0,0].set_xlabel(time_label)
    ax[0, 0].set_xlabel(temp_label)

    ax[0, 1].hist(df["tot"] - df["tot"].mean(), label="total", color=color_dict["tot"])
    ax[0, 1].title.set_text("Total [{:.0f} kcal/mol]".format(df["tot"].mean()))
    # ax[0,1].set_xlabel(time_label)
    ax[0, 1].set_xlabel(energy_label)

    ax[1, 0].hist(
        df["elec"] - df["elec"].mean(), label="elec", color=color_dict["elec"]
    )
    ax[1, 0].title.set_text("Elec [{:.0f} kcal/mol]".format(df["elec"].mean()))
    # ax[1,0].set_xlabel(time_label)
    ax[1, 0].set_xlabel(energy_label)

    ax[1, 1].hist(df["vdw"] - df["vdw"].mean(), label="vdw", color=color_dict["vdw"])
    ax[1, 1].title.set_text("Vdw [{:.0f} kcal/mol]".format(df["vdw"].mean()))
    # ax[1,1].set_xlabel(time_label)
    ax[1, 1].set_xlabel(energy_label)

    ax[0, 2].hist(
        df["user"] - df["user"].mean(), label="user", color=color_dict["user"]
    )
    ax[0, 2].title.set_text("User [{:.0f} kcal/mol]".format(df["user"].mean()))
    # ax[0,2].set_xlabel(time_label)
    ax[0, 2].set_xlabel(energy_label)

    ax[1, 2].hist(
        df["volume"] - df["volume"].mean(), label="volume", color=color_dict["volume"]
    )
    ax[1, 2].title.set_text("Volume [{:.0f}".format(df["volume"].mean()) + "$\AA^{3}$]")
    # ax[1,2].set_xlabel(time_label)
    ax[1, 2].set_xlabel(volume_label)

    return fig, ax


def make_plots(df, save=False, skips=None, psf="water.2000.psf", ind=False):
    dyna_keys = list(set(df["dyna"]))
    if np.nan in dyna_keys:
        dyna_keys.remove(np.nan)
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
            plot_filename = (
                f"/home/boittier/Documents/simreports/"
                f"figures/figs/{title}_{i}_energy_types.pdf"
            )
            fig.savefig(plot_filename, dpi=300)
            plot_filenames.append(trim_filename(plot_filename))

        # Plot fluctuations of energy components
        fig, ax = plot_fluctuations(df_, skip=skips[i])
        if save:
            plot_filename = (
                f"/home/boittier/Documents/"
                f"simreports/figures/figs/{title}_{i}_fluctuations.pdf"
            )
            fig.savefig(plot_filename, dpi=300)
            plot_filenames.append(trim_filename(plot_filename))


water_rdf_1 = """2.4615076480275357 0.059468174017968334
2.512692313398513 0.3882072509136556
2.5574974407667312 1.423799138515099
2.5603163063958574 1.1114688268079047
2.589246769431628 1.9059735224445984
2.602945467313522 2.388157797130691
2.6548719394290066 2.6347046867350117
2.6890939572422585 2.8429051130266214
2.7816714389567228 2.5853201390640375
2.801551859709508 2.3825695196554055
2.8774634165640496 1.9715690201721972
2.9164824513251135 1.6482599686463002
2.992047831698885 1.2756158232323962
3.1762631732514377 0.8645559792096273
3.4116631801749673 0.7822352120825458
3.717534827826654 0.8916566522756137
4.0769154686487745 1.072281649184752
4.563787961960149 1.1268093902843077
5.197439283117961 0.9182430059987414
5.504695636692729 0.874239029914591
6.099575192004312 0.9615843013911327
6.658353485749892 1.0489493543808615
7.2905706472016565 0.9992878655252708
7.88648873195555 0.971564074793899
8.698618755841729 0.9875574282054682
9.330341379463821 0.992690730877456
10.449431034226963 0.9975569831214215
11.31596021937698 0.9861232684994214
"""

water_rdf_2 = """2.412776412776413 0.03828650785172538
2.515970515970516 0.438436064523021
2.5454545454545454 1.142826621087491
2.633906633906634 1.9951287255635082
2.751842751842752 2.482256169212691
2.86977886977887 2.065035786774917
2.9729729729729732 1.4999679521418652
3.194103194103194 0.9437666915927785
3.385749385749386 0.8483922657835701
3.8574938574938575 0.9882063882063881
4.211302211302211 1.0582843713278498
4.535626535626536 1.0500587544065805
4.88943488943489 1.0157889114410854
5.213759213759214 0.9988676423459031
5.523341523341523 0.9471423993163122
5.803439803439803 0.9388526866787736
6.083538083538084 0.9653455827368871
6.378378378378379 1.0179468005554961
6.658476658476658 1.0444396966136096
6.968058968058967 1.036192714453584
7.616707616707616 1.0110458284371329
8.132678132678132 1.0030979596196985
8.737100737100738 1.0039739344087169
9.533169533169533 0.9964320051276572
9.93120393120393 1.01440017092191
"""


def plot_rdf_exp(ax, shift=0, c="k", lw=1, linestyle="--", marker="o"):
    # exp. data
    ex = []
    ey = []
    for _ in water_rdf_1.split("\n"):
        if _:
            ex.append(float(_.split(" ")[0]))
            ey.append(float(_.split(" ")[1]) + shift)

    ax.plot(
        ex,
        ey,
        linestyle,
        fillstyle="none",
        marker=marker,
        c=c,
        label="Exp. 1",
        linewidth=lw,
    )
    # exp. data
    ex = []
    ey = []
    for _ in water_rdf_2.split("\n"):
        if _:
            ex.append(float(_.split(" ")[0]))
            ey.append(float(_.split(" ")[1]) + shift)

    ax.plot(
        ex,
        ey,
        linestyle,
        fillstyle="none",
        c=c,
        marker=marker,
        label="Exp. 2",
        linewidth=lw,
    )
    return ax


def plot_rdf_from_file(title, ax, label=None):
    df = pd.read_csv(title)
    ax.plot(df["r"], df["g(r)"], c="k", label=label)
    return ax


def plot_rdf(ax, u, title=False, step=100):
    """Plot the rdf"""
    # get the rdf
    irdf = ff_energy.simulations.utils.g_rdf(u, step=step)
    peaks = get_rdf_peaks(irdf)

    # plot the rdf
    p = peaks * (irdf.results.bins[1] - irdf.results.bins[0] + 0.005)
    ax.plot(irdf.results.bins, irdf.results.rdf, c="k")

    # ax.plot(p[:3], irdf.results.rdf[peaks][:3], "x", c=color_dict["angle"])

    ax.set_xlabel("$r$ ($\mathrm{\AA}$)", fontsize=font_size)
    ax.set_ylabel("$g(r)$", fontsize=font_size)
    ax.set_title("RDF", fontsize=font_size, fontweight="bold")
    ax.set_xlim(0, 10)

    print(p, irdf.results.rdf[peaks])

    if title:
        odf = pd.DataFrame(
            {"r": irdf.results.bins, "g(r)": irdf.results.rdf}, index=irdf.results.bins
        )
        odf.to_csv(f"{title}_rdf.csv")

    return ax


def trim_filename(path, last=2):
    """Trim the filename to the last n directories"""
    splitted = path.split("/")
    return "/".join(splitted[-last:])
