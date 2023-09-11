from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ff_energy.ffe.potential import (
    LJ,
    DE,
)

from ff_energy.ffe.ff import FF
from ff_energy.plotting.fit_results import residuals_plot

from ff_energy.plotting.ffe_plots import plot_energy_MSE, plot_ff_fit
from ff_energy.ffe.ff_fit import (
    load_ff,
    fit_func,
    fit_repeat,
)
from ff_energy.utils.ffe_utils import pickle_output, read_from_pickle, str2int, PKL_PATH
from ff_energy.utils.json_utils import load_json

structure_data = {
    "dcm": "",
    "water_cluster": PKL_PATH / "20230823_water_clusters.pkl.pkl",
}

"""
Variables
"""
sig_bound = (0.001, 5.5)
ep_bound = (0.001, 5.5)
chg_bound = (100, 2000)

alpha_bound = (1, 8)
beta_bound = (6, 20)

CHGPEN_bound = [(chg_bound), (chg_bound), (chg_bound), (chg_bound), (0, 2000)]
LJ_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound))
DE_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound), alpha_bound, beta_bound)

NFIT_COUL = 1
NFIT_LJ = 4
NFIT_DE = 6

pkl_files = []
json_object = load_json("exp1.json")
#  make a product of all the values in the json object
experiments = list(it.product(*json_object.values()))
print(f"N experiments: {len(experiments)}")


def loop() -> list:
    """ " loop through the experiments and get info for the ff objects"""
    jobs = []
    for i, x in enumerate(experiments):
        print(f"Experiment {i}: {x}")
        jobs.append(x)
    return jobs


pc_charge_models = [
    "ELECnull", "ELECci", "ELECpol", "ELECp"
]


def make_ff_object(x):
    """ " make the ff object from the json list"""
    if x:
        structure = x[1]
        elec = x[2]
        fit = x[3]
        pair_dists = None
        #  which structure
        if structure == "dcm":
            #  load the pickle file
            pkl_file = PKL_PATH / "20230904_dcm.pkl.pkl"
            pkl_files.append(pkl_file)

        elif structure == "water_cluster":
            #  load the pickle files
            pkl_file = PKL_PATH / "20230823_water_clusters.pkl.pkl"
            pkl_files.append(pkl_file)
            pair_dists = next(read_from_pickle(PKL_PATH / "water_pc_pairs.pkl"))
            #  load the pair distances
            # if elec in pc_charge_models:
            #     pair_dists = next(read_from_pickle(PKL_PATH / "water_pc_pairs.pkl"))
            # elif elec == "ELECk":
            #     pair_dists = next(read_from_pickle(PKL_PATH / "water_kmdcm_pairs.pkl"))
            # elif elec == "ELECm":
            #     pair_dists = next(read_from_pickle(PKL_PATH / "water_mdcm_pairs.pkl"))
            # else:
            #     raise ValueError("Invalid elec type", elec)

        else:
            raise ValueError("Invalid structure")

        pkl_file = read_from_pickle(pkl_file)
        data = next(pkl_file)
        print(data.keys())
        print(data[elec])

        stuct_data = read_from_pickle(PKL_PATH / "structures" / f"{structure}.pkl")
        struct_data = next(stuct_data)
        structs, pdbs = struct_data
        # print(structs, pdbs)
        struct_data = structs[0]
        print(struct_data)
        print(pair_dists)
        #  set the 2body terms
        print("setting 2body")
        for _ in structs:
            # print("struct name", _.name)
            _.set_2body()

        dists = {
            str(Path(_.name).stem).split(".")[0]:
                _.distances for _ in structs
        }
        print(dists.keys())
        print(data.index)

        if fit == "lj":
            FUNC = LJ
            BOUNDS = 2 * len(set(struct_data.chm_typ)) * [sig_bound]
        elif fit == "de":
            FUNC = DE
            BOUNDS = 2 * len(set(struct_data.chm_typ)) * [sig_bound]
            BOUNDS.append(alpha_bound)
            BOUNDS.append(beta_bound)
        else:
            raise ValueError("Invalid fit type")

        #  make the ff object
        ff = FF(
            data,
            dists,
            FUNC,
            BOUNDS,
            struct_data,
            elec=elec,
        )

        # ff.set_targets()

        #  pickle the ff object
        pickle_output(ff, f"{elec}_{structure}_{fit}")

        outes = ff.out_es
        print(ff.p, ff.opt_parm)
        flateval = ff.eval_jax_flat(ff.p)
        resids = outes - flateval

        print(ff.p)
        print("out es", outes)
        print("es shape", ff.out_es.shape)
        print("flat eval", flateval)
        print("flat eval shape", flateval.shape)
        print("resids", resids)
        print("resids shape", resids.shape)

        #  make the dataframe
        df_test = pd.DataFrame(
            {
                "target": outes,
                "residuals": resids,
                "vals": flateval
            }
        )  # drop the nans

        print(df_test.describe())
        #  plot the results
        # residuals_plot(df_test, "flat_test")
        # plt.clf()

        # outes = ff.out_es
        # flateval = ff.eval_jax_flat([ff.p[2], ff.p[3], ff.p[0], ff.p[1]])
        # resids = outes - flateval
        #
        # print(ff.p)
        # print("out es", outes)
        # print("es shape", ff.out_es.shape)
        # print("flat eval", flateval)
        # print("flat eval shape", flateval.shape)
        # print("resids", resids)
        # print("resids shape", resids.shape)
        #
        # #  make the dataframe
        # df_test = pd.DataFrame(
        #     {
        #         "target": outes,
        #         "residuals": resids,
        #         "vals": flateval
        #     }
        # )  # drop the nans
        #
        # print(df_test.describe())
        return ff


def ff_fit(x, n=100):
    structure = x[1]
    elec = x[2]
    fit = x[3]
    ffpkl = f"{elec}_{structure}_{fit}"
    #  load_ff
    ff = read_from_pickle(PKL_PATH / (ffpkl + ".pkl"))
    ff = next(ff)
    print("FF:", ff)
    print("bounds:", ff.bounds)
    print("elec:", ff.elec)
    print(ff.data.keys())
    print("data:", ff.data[[ff.intE, ff.elec, "VDW", "P_intE"]].describe())
    print("data:", ff.data[[ff.intE, ff.elec, "VDW", "P_intE"]])
    print("targets:", ff.targets)
    print("nTargets:", ff.nTargets)

    ff.num_segments = ff.nTargets
    ff.set_targets()

    loss = "jax"
    LJFF = fit_repeat(
        ff, n, f"{ffpkl}_fitted", bounds=ff.bounds, loss=loss,
        quiet=False
    )

    print(LJFF.opt_results)
    pickle_output(LJFF, f"{ffpkl}_fitted")
    jaxeval = ff.eval_jax(LJFF.get_best_parm())

    print("eval_jax::", jaxeval)
    jaxloss = ff.get_loss_jax(LJFF.get_best_parm())

    print("jaxloss::", jaxloss)
    elec = ff.data[LJFF.elec]
    targets = ff.targets
    print("targets", targets)
    residuals = targets - jaxeval
    #  make the dataframe
    df_test = pd.DataFrame(
        {
            "target": targets,
            "residuals": residuals,
            "vals": jaxeval
        }
    ).dropna()  # drop the nans
    #  plot the results
    residuals_plot(df_test, ffpkl + "_targets")
    print(df_test.describe())
    plt.clf()

    #  make the dataframe
    df_test = pd.DataFrame(
        {
            "target": ff.data["intE"],
            "residuals": ff.data["intE"] - (jaxeval + elec),
            "vals": jaxeval + elec
        }
    ).dropna()  # drop the nans
    #  plot the results
    residuals_plot(df_test, ffpkl + "_elec_jax")
    print(df_test.describe())

    plt.clf()
    print(ff.data)
    print(LJFF.data)


if __name__ == "__main__":
    jobs = loop()
    import argparse

    #  argument for which experiment to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", "-x", type=int, help="Experiment number")
    parser.add_argument("--m", "-m", action='store_true', help="Make the ff object")
    parser.add_argument("--f", "-f", action='store_true', help="Fit the ff object")
    parser.add_argument("--n", "-n", type=int, help="Number of repeats", default=100)
    args = parser.parse_args()
    print(args.x)
    job = jobs[args.x]
    print(job)

    if args.m:
        print("Making ff object")
        make_ff_object(job)

    if args.f:
        print("Fitting ff object, repeats = {}".format(args.n))
        ff = ff_fit(job, n=args.n)
