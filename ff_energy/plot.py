import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_energy_MSE(
    df,
    key1,
    key2,
    FONTSIZE=14,
    xlabel="NBOND energy\n(kcal/mol)",
    ylabel="CCSD(T) interaction energy\n(kcal/mol)",
    elec="ele",
    CMAP="viridis",
    cbar_label="ELEC (kcal/mol)",
    ax=None,
    bootstrap=True,
    title=True,
    bounds = None,
):
    """Plot the energy MSE"""
    df = df.copy()
    if ax is None:
        fig, ax = plt.subplots()
    # calculate MSE
    ERROR = df[key1] - df[key2]
    MSE = np.mean(ERROR**2)
    df["SE"] = ERROR.copy() ** 2

    if bootstrap:
        from scipy.stats import bootstrap

        rng = np.random.default_rng()
        data = (df["SE"],)  # samples must be in a sequence
        res = bootstrap(
            data, np.mean, confidence_level=0.95, random_state=rng, n_resamples=100
        )
        seMin, seMax = res.confidence_interval
        rseMin, rseMax = np.sqrt(seMin), np.sqrt(seMax)

    # get spearman correlation
    spearman = df[key1].corr(df[key2], method="spearman")

    # add the MSE to the plot
    import scipy.stats

    try:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            df[key1], df[key2]
        )
    except:
        slope, intercept, r_value, p_value, std_err = 1, 0, 0, 0, 0
    if bootstrap:
        stats_str = (
            f"MSE = {MSE:.2e}$^{{{seMax:.1e}}}_{{{seMin:.1e}}}$ kcal/mol\n"
            f"RMSE = {np.sqrt(MSE):.2e}$^{{{rseMax:.1e}}}_{{{rseMin:.1e}}}$ kcal/mol"
            f"\n$R$ = {r_value:.2e}, $R_\mathrm{{S.}}$ = {spearman:.2f}, $n$ = {len(df)}"
        )
    else:
        stats_str = (
            f"MSE = {MSE:.2e} kcal/mol\n"
            f"RMSE = {np.sqrt(MSE):.2e} kcal/mol"
            f"\n$R$ = {r_value:.2e}, $R_\mathrm{{S.}}$ = {spearman:.2f}, $n$ = {len(df)}"
        )
    if title:
        ax.text(0.00, 1.05, stats_str, transform=ax.transAxes, fontsize=FONTSIZE)
    # color points by MSE
    sc = ax.scatter(df[key1], df[key2], c=df[elec], cmap=CMAP, alpha=0.5, s=1)

    #  make the aspect ratio square
    ax.set_aspect("equal")
    if bounds is not None:
        min, max = bounds
    else:
        min = np.min([df[key1].min(), df[key2].min()])
        max = np.max([df[key1].max(), df[key2].max()])
    #  make the range of the plot the same
    ax.set_ylim(min, max)
    ax.set_xlim(min, max)
    #  plot the diagonal line
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", alpha=0.5)
    #  plot line of best fit
    ax.plot(
        [min, max],
        [slope * min + intercept, slope * max + intercept],
        ls="-",
        c="k",
        alpha=0.15,
    )
    #  show the grid
    ax.grid(alpha=0.15)
    #  set the labels
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if ax is None:
        plt.colorbar(sc, label=cbar_label)
        #  tight layout
        plt.tight_layout()

    stats = {"MSE": MSE,
             "RMSE": np.sqrt(MSE),
             "R": r_value,
             "RS": spearman,
             "n": len(df)}

    return ax, stats
