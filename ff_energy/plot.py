import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
    bounds=None,
    s=1,
    cbar_bounds=(-120, 0),
):
    """Plot the energy MSE"""
    df = df.copy()
    _ax = ax
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
    except Exception:
        print("Error in linregress")
        (
            slope,
            intercept,
            r_value,
        ) = (
            np.nan,
            np.nan,
            0,
        )

    if bootstrap:
        stats_str = (
            f"MSE = {MSE:.2e}$^{{{seMax:.1e}}}_{{{seMin:.1e}}}$ kcal/mol\n"
            f"RMSE = {np.sqrt(MSE):.2e}$^{{{rseMax:.1e}}}_{{{rseMin:.1e}}}$ kcal/mol"
            f"\n$R$ = {r_value:.2e}, $R_\mathrm{{S.}}$ = "
            f"{spearman:.2f}, $n$ = {len(df)}"
        )
    else:
        stats_str = (
            f"MSE = {MSE:.2e} kcal/mol\n"
            f"RMSE = {np.sqrt(MSE):.2e} kcal/mol"
            f"\n$R$ = {r_value:.2e}, $R_\mathrm{{S.}}$ = "
            f"{spearman:.2f}, $n$ = {len(df)}"
        )
    if title:
        ax.text(0.00, 1.05, stats_str, transform=ax.transAxes, fontsize=FONTSIZE)
    # color points by MSE
    sc = ax.scatter(df[key1], df[key2], c=df[elec], cmap=CMAP, alpha=0.5, s=s)
    print(p_value)
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
    cbar = None
    if _ax is None:
        cbar = fig.colorbar(sc, label=cbar_label)
        # cbar.set_clim(*cbar_bounds)
        #  tight layout
        # fig.set_clim(*cbar_bounds)
        plt.tight_layout()

    stats = {
        "MSE": MSE,
        "RMSE": np.sqrt(MSE),
        "R": r_value,
        "RS": spearman,
        "n": len(df),
        "slope": slope,
        "intercept": intercept,
    }

    return ax, cbar, stats


def plot_ff_fit(ff_pairs, ff_fit, ecol="ECOL_PC", EB=("sd", 3), suptitle=None):
    """
    Bin by min_hbond distance
    """
    key = "min_hbond"
    key_min = ff_pairs.data[key].min()
    key_max = ff_pairs.data[key].max()
    key_range = key_max - key_min
    NBINS = 75
    bins = np.arange(key_min, key_max, key_range / NBINS)
    ff_pairs.data[f"{key}_bin"] = np.digitize(ff_pairs.data[key], bins=bins)
    ff_pairs.data[f"{key}_bin"] = (
        key_min + ff_pairs.data[f"{key}_bin"] * key_range / NBINS
    )
    ff_pairs.elec = ecol
    ff_pairs.func = ff_fit.func

    """
    Ignore ELEC calculations with > 1 kcal/mol error
    """
    ff_pairs.data["DUMMY"] = [0] * len(ff_pairs.data)
    MSE = (
        (ff_pairs.data.groupby("key")[ecol].sum() - ff_fit.data.sort_index()["ELEC"])
        ** 2
    ).mean()
    RMSE = np.sqrt(MSE)
    print(MSE, RMSE)
    _ = pd.DataFrame(
        {
            "true": ff_pairs.data.groupby("key")[ecol].sum(),
            "test": ff_fit.data.sort_index()["ELEC"],
        }
    )
    _["error"] = (_["true"] - _["test"]).abs()
    ECOLCUT = 1
    if ecol == "DUMMY" or ecol == "epol_mike":
        ECOLCUT = 1000000000

    list(_[_["error"] > ECOLCUT].index)
    # ff_fit.data = ff_fit.data.query("index not in @error_keys").copy()

    Nopts = len(ff_fit.opt_results)
    iis = [i for i in range(Nopts)]  # if ff_fit.opt_results[i]["fun"] < 10*10]

    out_stats = []

    for i in iis:
        print(i)
        print(ff_fit.opt_results[i]["fun"])
        """ Eval 2 body interactions
        """
        parms = ff_fit.opt_results[i]["x"]
        fit = ff_pairs.LJ_performace(ff_pairs.eval_func(parms))
        # fit = fit.query("key not in @error_keys").copy()
        """ Eval many body interactions
        """
        _ = ff_fit.LJ_performace(ff_fit.eval_func(parms))
        # _ = _.query("index not in @error_keys").copy()

        """Plot
        """
        fig, axs = plt.subplot_mosaic(
            [["a)", "b)"], ["c)", "d)"], ["e)", "f)"]],
            layout="constrained",
            figsize=(8, 8),
        )

        titles = {
            "a)": "Pairs(model): $E_{PP}(r)$, $E_{ELEC}(r)$",
            "b)": "Pairs: $E_{DFT}(r)$ $E_{model}(r)$",
            "c)": "$E_{pairs}$ Distribution",
            "d)": "DFT: Many-body Error",
            "e)": "Pairs: Two-body Error (Test)",
            "f)": "Clusters: Many-body Error (Train)",
        }
        # plot labels
        for label, ax in axs.items():
            ax.set_title(label + " " + titles[label])
        #  elec and LJ
        EvsR(fit, axs["a)"], ecol=ecol)
        #  intE fit vs data
        intEvsR(fit, axs["b)"])
        # distributions
        plot_dists(fit, axs["c)"])
        # pair correl
        QQQ, spp = cor_pairs(fit, axs["e)"])

        if spp["RMSE"] > 500 and ecol != "DUMMY":
            print("shuffling.. ooops")
            if len(parms) == 6:
                parms = [parms[1], parms[0], parms[3], parms[2], parms[5], parms[4]]
            else:
                parms = [parms[1], parms[0], parms[3], parms[2]]
            fit = ff_pairs.LJ_performace(ff_pairs.eval_func(parms))
            # fit = fit.query("key not in @error_keys").copy()
            QQQ, spp = cor_pairs(fit, axs["e)"])
            #  elec and LJ
            axs["a)"].cla()
            EvsR(fit, axs["a)"], ecol=ecol)
            #  intE fit vs data
            axs["b)"].cla()
            intEvsR(fit, axs["b)"])

            # pair/cluster correl
        QQQ, spc = cor_pairs_cluster(_, axs["d)"])
        # clusters correl
        QQQ, scc = cor_cluster(_, axs["f)"])

        out_stats.append([parms, spp, scc])

        #  plot statistics
        axes = [axs["e)"], axs["d)"], axs["f)"]]
        stats = [spp, spc, scc]
        for (
            ax_,
            s,
        ) in zip(axes, stats):
            ax_.text(
                1.0,
                0.6,
                "RMSE: {:.1f}\nR: {:.2f}\nRs: ${:.2f}$".format(
                    s["RMSE"], s["R"], s["RS"]
                ),
                transform=ax_.transAxes,
                ma="center",
                fontsize=15,
            )
        # title
        # plt.suptitle(ff_fit.name, fontsize=20)
        pt = "_".join(["{:.3f}".format(x) for x in parms])
        if suptitle is not None:
            # plt.suptitle(suptitle + f"_{pt}", fontsize=20)
            results = scc["RMSE"]
            save_path = (
                f"/home/boittier/Documents/phd/ff_energy/figs/ff/"
                f"{suptitle}_{ff_fit.name}_{ff_fit.intern}_{ff_fit.elec}/{results}_{pt}.png"
            )
            import os

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    return out_stats


def intEvsR(fit, ax, EB=("sd", 3)):
    sns.lineplot(
        x="min_hbond_bin",
        y="nb_intE",
        data=fit,
        errorbar=EB,
        color="purple",
        ax=ax,
        label="$E_{pairs(Model)}$",
    )
    sns.lineplot(
        x="min_hbond_bin",
        y="intE",
        data=fit,
        errorbar=EB,
        color="black",
        alpha=0.8,
        ax=ax,
        label="$E_{pairs(DFT)}$",
    )
    ax.axhline(0, c="k", linewidth=0.5)
    ax.set_xlim(1.0, 5)
    ax.set_xlabel("$r$ [$\mathrm{\\AA}$]")
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.5)
    return ax


def EvsR(fit, ax, ecol="ECOL_kmdcm", EB=("sd", 3)):
    sns.lineplot(
        x="min_hbond_bin",
        y=ecol,
        data=fit,
        errorbar=EB,
        color="orange",
        ax=ax,
        label="$E_{ELEC}$",
    )
    sns.lineplot(
        x="min_hbond_bin",
        y="LJ",
        data=fit,
        errorbar=EB,
        color="green",
        ax=ax,
        label="$E_{PP}$",
    )
    ax.axhline(0, c="k", linewidth=0.5)
    ax.grid(alpha=0.5)
    ax.set_xlim(1, 5)
    ax.set_xlabel("$r$ [$\mathrm{\\AA}$]")
    ax.set_ylim(-15, 15)
    return ax


def plot_dists(fit, ax):
    # fit = fit.dropna()
    fit = fit.query("-10 < nb_intE < 10")
    ax.hist(fit["intE"], color="k", alpha=0.3, label="$E_{pairs(DFT)}$")
    ax.hist(fit["nb_intE"], color="purple", alpha=0.3, label="$E_{pairs(Model)}$")
    ax.set_xlabel("$E$ [kcal/mol]")
    ax.legend()
    ax.set_xlim(-10, 10)
    return ax


# from ff_energy.plot import plot_energy_MSE
def cor_pairs(fit, ax):
    ax, _, stats = plot_energy_MSE(
        fit.groupby("key").sum(),
        "intE",
        "nb_intE",
        elec="ECOL_PC",
        bootstrap=True,
        xlabel="$E_{DFT}$ [kcal/mol]",
        ylabel="$E_{Model}$ [kcal/mol]",
        ax=ax,
        title=False,
        FONTSIZE=10,
        bounds=(-100, -20),
    )
    return ax, stats


def cor_pairs_cluster(fit, ax):
    ax, _, stats = plot_energy_MSE(
        fit,
        "intE",
        "P_intE",
        elec="ECOL",
        bootstrap=True,
        xlabel="$E_{Clusters}$ [kcal/mol]",
        ylabel="$E_{Pairs}$ [kcal/mol]",
        ax=ax,
        title=False,
        FONTSIZE=10,
        bounds=(-100, -20),
    )
    return ax, stats


def cor_cluster(fit, ax):
    ax, _, stats = plot_energy_MSE(
        fit,
        "intE",
        "nb_intE",
        elec="ECOL",
        bootstrap=True,
        xlabel="$E_{DFT}$ [kcal/mol]",
        ylabel="$E_{Model}$ [kcal/mol]",
        ax=ax,
        title=False,
        FONTSIZE=10,
        bounds=(-100, -20),
    )
    return ax, stats


#
# def plot_best_fits(ff, name=""):
#     """
#     Plot the best fits
#     :param ff:
#     :param name:
#     :return:
#     """
#     test_ = []
#     train_ = []
#     args_ = []
#     dfs_ = []
#
#     for res, data in zip(ff.opt_results, ff.opt_results_df):
#         if res["fun"] <= 30:
#             data = data.dropna()
#             test_keys = [_ for _ in list(ff.data_save.index) if _ not in data.index]
#             print("N test:", len(test_keys))
#             test_df = ff.data_save.query("index in @test_keys")
#             ff.data = test_df.copy()
#             test_len = len(ff.data)
#             test_res = ff.LJ_performace(ff.eval_func(res.x)).copy()
#
#             # fig, ax = plt.subplots(1,3,figsize=(15,15))
#             fig, ax = plt.subplot_mosaic([[0, 1], [2, 2]], figsize=(10, 7))
#
#             plot_LJintE(test_res, ax=ax[0], elec=ff.elec)
#
#             ff.data = ff.data_save.query("index not in @test_keys").copy()
#             train_len = len(ff.data)
#             train_res = ff.LJ_performace(ff.eval_func(res.x)).copy()
#             plot_LJintE(train_res, ax=ax[1], elec=ff.elec)
#             results = (
#                 "{:.1f}_{}_{:.1f}_{}".format(
#                     test_res["SE"].mean(), test_len, train_res["SE"].mean(), train_len
#                 )
#             )
#             print(results)
#             res = res.x
#             x = np.arange(0.1, 5, 0.05)
#             y = LJ(res[0] * 2, res[2], x)
#             ax[2].plot(x, y, c="grey")
#             y = LJ(res[1] * 2, res[3], x)
#             ax[2].plot(x, y, c="firebrick")
#             y = LJ(res[1] + res[1], np.sqrt(res[3] * res[2]), x)
#             ax[2].plot(x, y, c="k")
#             ax[2].axhline(0, c="k")
#             ax[2].set_ylim(-1, 1)
#             res_str = "_".join(["{:.2f}".format(x) for x in res])
#             save_path = (
#                 f"/home/boittier/Documents/phd/ff_energy/figs/ff/"
#                 f"{name}_{ff.name}_{ff.intern}_{ff.elec}/{results}_{res_str}.png"
#             )
#             #  make the directory
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             print("saving image to: ", save_path)
#             plt.suptitle(res_str, y=-0.051)
#             plt.savefig(save_path, bbox_inches="tight")
#
#             # append data
#             test_.append(test_res["SE"].mean())
#             train_.append(train_res["SE"].mean())
#             args_.append(res)
#             dfs_.append(test_res)
#
#     return {"test": test_, "train": train_, "args": args_, "dfs": dfs_}
