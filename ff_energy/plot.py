import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_energy_MSE(df, key1, key2, FONTSIZE=14,
                    xlabel="NBOND energy\n(kcal/mol)",
                    ylabel="CCSD(T) interaction energy\n(kcal/mol)",
                    elec="ele",
                    CMAP="viridis",
                    cbar_label = "ELEC (kcal/mol)"):
    """Plot the energy MSE"""

    fig, ax = plt.subplots()
    # calculate MSE
    ERROR = df[key1] - df[key2]
    MSE = np.mean(ERROR ** 2)
    df["SE"] = ERROR ** 2
    # add the MSE to the plot
    import scipy.stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        df[key1], df[key2])
    # print(slope, intercept, r_value, p_value, std_err)
    stats_str = f"MSE = {MSE:.2f} kcal/mol\nRMSE = {np.sqrt(MSE):.2f} kcal/mol" \
                f"\n$R$ = {r_value:.2f}, $R$$^2$ = {r_value**2:.2f}, $p$ = {p_value:.2e}"
    ax.text(0.00, 1.05, stats_str,
            transform=ax.transAxes, fontsize=FONTSIZE)
    # color points by MSE
    sc = ax.scatter(df[key1], df[key2],
                    c=df[elec], cmap=CMAP, alpha=0.5)


    #  make the aspect ratio square
    ax.set_aspect("equal")
    min = np.min([df[key1].min(), df[key2].min()])
    max = np.max([df[key1].max(), df[key2].max()])
    #  make the range of the plot the same
    ax.set_ylim(min, max)
    ax.set_xlim(min, max)
    #  plot the diagonal line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", alpha=0.5)
    #  plot line of best fit
    ax.plot([min,max], [slope*min + intercept,slope*max + intercept],
            ls="-", c="k", alpha=0.15)
    #  show the grid
    ax.grid(alpha=0.15)
    #  set the labels
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    plt.colorbar(sc, label=cbar_label)
    #  tight layout
    plt.tight_layout()

    return df
