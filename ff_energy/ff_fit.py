from ff_energy.cli import load_config_maker, load_all_theory, charmm_jobs
# from ff_energy.structure import
from ff_energy.potential import FF, LJ, DE
import numpy as np
from ff_energy.data import Data, plot_ecol, plot_intE, plot_LJintE
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd
from ff_energy.utils import pickle_output, read_from_pickle
from ff_energy.data import Data

sig_bound = (0.25, 2.5)
ep_bound = (0.0001, 0.5)
LJ_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound))
DE_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound),
            (4, 16), (4, 16))

func_bounds = {"LJ": (LJ, LJ_bound),
               "DE": (DE, DE_bound)}


def load_ff(ff_name,
            structure,
            pickled_ff=False,
            pickled_dists=False,
            FUNC=LJ,
            BOUNDS=LJ_bound,
            pk="pickles/water_cluster/pbe0dz/pbe0_dz.kmdcm"
            ):
    ff = None
    ff_pkls = Path("pickles/ff")
    ff_pickles = ff_pkls.glob("*.pkl")
    ff_pickles = [_.name for _ in ff_pickles]
    struct_dist_pkls = Path("pickles/structures")
    struct_dist_pickles = struct_dist_pkls.glob("*.pkl")
    struct_dist_pickles = [_.name for _ in struct_dist_pickles]
    if pickled_ff:
        pickled_ff = (ff_pkls / f"{ff_name}.pkl").exists()
    if pickled_dists:
        pickled_dists = (struct_dist_pkls / f"{structure}.pkl").exists()

    if pickled_ff:
        print("Pickled FF exists!, loading: ", ff_pkls / f"{ff_name}.pkl")
        try:
            ff = next(read_from_pickle(ff_pkls / f"{ff_name}.pkl"))
        except StopIteration:
            print("Pickle read failed.")
            pickled_ff = False

    if not pickled_ff:
        if pickled_dists:
            print("loading pickled distances/structure")
            structures, dists = next(read_from_pickle(
                f"pickles/structures/{structure}.pkl"))
        else:
            print("No pickled distances/structure information, calculating:")
            CMS = load_config_maker("pbe0dz", structure, "mdcm")
            jobs = charmm_jobs(CMS)
            dists = {_.name.split(".")[0]: _.distances for _ in jobs[0].structures}
            structures = [_ for _ in jobs[0].structures]
            pickle_output((structures, dists),
                          name=f"structures/{structure}")
        s = structures[0]
        data_ = Data(pk)
        ff = FF(data_.data,
                dists,
                FUNC,
                BOUNDS,
                s)
    return ff


def fit_repeat(ff, n, k, bounds, clip=10, outname="ff/pbe0dz_mdcm"):
    ff.data_save["LJX"] = ff.data_save["intE"] - ff.data_save["ELEC"]
    ff.opt_results, ff.opt_results_df = [], []
    for i in range(k):
        d = ff.data_save.sort_values("LJX")[clip:-clip].sample(250)
        ff.data = d.copy()
        ff.fit_repeat(n, bounds=bounds)
    pickle_output(ff, outname)


def plot_best_fits(ff):
    test_ = []
    train_ = []
    args_ = []
    dfs_ = []

    for res, data in zip(ff.opt_results, ff.opt_results_df):
        if res["fun"] < 30:
            test_keys = [_ for _ in list(ff.data_save.index) if _ not in data.index]
            test_df = ff.data_save.query("index in @test_keys")
            ff.data = test_df.copy()
            test_len = len(ff.data)
            test_res = ff.LJ_performace(ff.eval_func(res.x))

            # fig, ax = plt.subplots(1,3,figsize=(15,15))
            fig, ax = plt.subplot_mosaic([[0, 1], [2, 2]],
                                         figsize=(10, 7))

            plot_LJintE(test_res, ax=ax[0])

            ff.data = ff.data_save.query("index not in @test_keys").copy()
            train_len = len(ff.data)
            train_res = ff.LJ_performace(ff.eval_func(res.x))
            plot_LJintE(train_res, ax=ax[1])
            print("{:.1f} ({}) {:.1f} ({})".format(test_res["SE"].mean(),
                                                   test_len,
                                                   train_res["SE"].mean(),
                                                   train_len))

            res = res.x
            x = np.arange(0.1, 5, 0.05)
            y = LJ(res[0] * 2, res[2], x)
            ax[2].plot(x, y, c="grey")
            y = LJ(res[1] * 2, res[3], x)
            ax[2].plot(x, y, c="firebrick")
            y = LJ(res[1] + res[1], np.sqrt(res[3] * res[2]), x)
            ax[2].plot(x, y, c="k")

            ax[2].axhline(0, c="k")
            ax[2].set_ylim(-1, 1)
            # plt.show()
            plt.savefig(f"plots/ff/{ff.name}_{res}.png")

            test_.append(test_res["SE"].mean())
            train_.append(train_res["SE"].mean())
            args_.append(res)
            dfs_.append(test_res)

    return {"test": test_, "train": train_, "args": args_, "dfs": dfs_}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    print("----")

    parser.add_argument('-f', '--ff', help='Forcefield name', required=True)
    parser.add_argument('-s', '--structure', help='Structure name', required=True)
    parser.add_argument('-n', '--n', help='Number of repeats', required=False,default=10)
    parser.add_argument('-k', '--k', help='Number of fits', required=False,default=10)
    parser.add_argument('-ft', '--fftype', help='Forcefield type', required=True)
    parser.add_argument('-c', '--clip', help='Clip', required=False,default=10)
    parser.add_argument('-o', '--outname', help='Outname', required=False,default=None)
    parser.add_argument('-p', '--pk', help='Pickle', required=False,default=None)

    args = parser.parse_args()
    print(args)
    print("----")
    # load the force field
    func, bounds = func_bounds[args.fftype]
    ff = load_ff(args.ff,
                 args.structure,
                 args.pk,
                 args.fftype)
    # do the fitting
    fit_repeat(ff,
               int(args.n),
               int(args.k),
               bounds,
               clip=int(args.clip),
               outname=args.outname)
