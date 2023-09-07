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
CHGPEN_bound = [(chg_bound), (chg_bound), (chg_bound), (chg_bound), (0, 2000)]
LJ_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound))
DE_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound), (1, 8), (6, 20))

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

            #  load the pair distances
            if elec in pc_charge_models:
                pair_dists = next(read_from_pickle(PKL_PATH / "water_pc_pairs.pkl"))
            elif elec == "ELECk":
                pair_dists = next(read_from_pickle(PKL_PATH / "water_kmdcm_pairs.pkl"))
            elif elec == "ELECm":
                pair_dists = next(read_from_pickle(PKL_PATH / "water_mdcm_pairs.pkl"))
            else:
                raise ValueError("Invalid elec type", elec)

        else:
            raise ValueError("Invalid structure")

        if fit == "lj":
            FUNC = LJ
            BOUNDS = LJ_bound
        elif fit == "de":
            FUNC = DE
            BOUNDS = DE_bound
        else:
            raise ValueError("Invalid fit type")

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
        for _ in structs:
            print(_.name)
            _.set_2body()
        dists = {str(Path(_.name).stem).split(".")[0]: _.distances for _ in structs}
        print(dists.keys())
        #  make the ff object
        ff = FF(
            data,
            dists,
            FUNC,
            BOUNDS,
            struct_data,
            elec=elec,
        )
        #  pickle the ff object
        pickle_output(ff, f"{elec}_{structure}_{fit}")
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
    print("targets:", ff.targets)
    print("nTargets:", ff.nTargets)

    ff.num_segments = ff.nTargets
    ff.set_targets()
    loss = "jax"
    LJFF = fit_repeat(
        ff, n, f"{ffpkl}_fitted", bounds=ff.bounds, loss=loss,
        quiet="false"
    )
    print(LJFF.opt_results)
    pickle_output(LJFF, f"{ffpkl}_fitted")
    jaxeval = LJFF.eval_jax(LJFF.opt_parm)
    print(jaxeval)
    jaxloss = LJFF.get_loss_jax(LJFF.opt_parm)
    print(jaxloss)
    elec = LJFF.data[LJFF.elec]
    targets = LJFF["intE"]
    residuals = targets - jaxeval
    #  make the dataframe
    df_test = pd.DataFrame(
        {
            "target": targets + elec,
            "residuals": residuals,
            "vals": jaxeval + elec
        }
    ).dropna() # drop the nans
    #  plot the results
    residuals_plot(df_test, ffpkl)
    print(df_test.describe())

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
