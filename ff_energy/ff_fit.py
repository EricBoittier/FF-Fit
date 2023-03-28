from ff_energy.cli import load_config_maker, load_all_theory, charmm_jobs
# from ff_energy.structure import
from ff_energy.potential import FF, LJ
import numpy as np
from ff_energy.data import Data, plot_ecol, plot_intE
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd
from ff_energy.utils import *
from ff_energy.data import Data

sig_bound = (0.5, 4.0)
ep_bound = (0.00001, 1.0)
LJ_bound = [(sig_bound), (sig_bound), (ep_bound), (ep_bound)]

FUNC = LJ
BOUNDS = LJ_bound
ff_name = "pbe0dz_pc"
structure = "water_cluster"
pk = "pickles/water_cluster/pbe0_dz.pc"


#########################################################
def init_job(ff_name, structure, pk):
    ff_pkls = Path("pickles/ff")
    ff_pickles = ff_pkls.glob("*.pkl")
    ff_pickles = [_.name for _ in ff_pickles]
    struct_dist_pkls = Path("pickles/structures")
    struct_dist_pickles = struct_dist_pkls.glob("*.pkl")
    struct_dist_pickles = [_.name for _ in struct_dist_pickles]

    pickled_ff = (ff_pkls / f"{ff_name}.pkl").exists()
    pickled_dists = (struct_dist_pkls / f"{structure}.pkl").exists()
    structures = None

    if pickled_ff:
        print("Pickled FF exists!, loading: ", ff_pkls / f"{ff_name}.pkl")
        ff = next(read_from_pickle(ff_pkls / f"{ff_name}.pkl"))
    else:
        if pickled_dists:
            print("loading pickled distances/structure")
            structures, dists = next(read_from_pickle(
                f"structures/{structure}.pkl"))
        else:
            print("No pickled distances/structure information, calculating:")
            theory, elec = ff_name.split("_")

            CMS = load_config_maker(theory, structure, elec)

            jobs = charmm_jobs(CMS)

            dists = {_.name.split(".")[0]: _.distances
                     for _ in jobs[0].structures}

            # print(dists.keys())

            structures = [_ for _ in jobs[0].structures]
            pickle_output((structures, dists),
                          name=f"structures/{structure}.pkl")

        s = structures[0]
        data_ = Data(pk)
        ff = FF(data_.data, dists, FUNC, BOUNDS, s)
        print(ff)

    return ff, structures, dists


def fit_repeat(ff_name, ff, n_rep = 10, n_sample = 100, bounds = None):
    for i in range(n_rep):
        d = ff.data_save.sample(n_sample)
        ff.data = d.copy()
        ff.fit_repeat(10, bounds=None)
    pickle_output(ff, f"ff/{ff_name}")

def test_train_mse(ff):
    test_ = []
    train_ = []
    args_ = []
    dfs_ = []

    for res, data in zip(ff.opt_results, ff.opt_results_df):
        test_keys = [_ for _ in list(ff.data_save.index) if _ not in data.index]
        test_df = ff.data_save.query("index in @test_keys")
        ff.data = test_df.copy()
        test_len = len(ff.data)
        test_res = ff.eval_func(res.x)
        plot_intE(test_res)

        ff.data = ff.data_save.query("index not in @test_keys").copy()
        train_len = len(ff.data)
        train_res = ff.eval_func(res.x)
        plot_intE(train_res)
        print("{:.1f} ({}) {:.1f} ({})".format(test_res["SE"].mean(),
                                               test_len,
                                               train_res["SE"].mean(),
                                               train_len))
        test_.append(test_res["SE"].mean())
        train_.append(train_res["SE"].mean())
        args_.append(res.x)
        dfs_.append(test_res)
    df = pd.DataFrame({"test": test_, "train": train_, "args": args_, "df": dfs_})
    return df


# d = ff.data_save.sample(200)
# ff.data = d.copy()
args = [1.7682, 0.2245, -0.1521, -0.0460]
ff, structures, dists = init_job(ff_name, structure, pk)
