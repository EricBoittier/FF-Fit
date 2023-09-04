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
sig_bound = (0.001, 2.5)
ep_bound = (0.001, 2.5)
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

#    vim.treesitter.language.add('python', { path = "/path/to/python.so" })


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
            pkl_file = PKL_PATH / "20230823_dcm.pkl.pkl"
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
        #print(structs, pdbs)
        struct_data = structs[0]
        print(struct_data)
        print(pair_dists)
        dists = { str(Path(_.name).stem).split(".")[0] : _.distances for _ in structs}
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
        #

        #  which ff_type


def ff_fit(x):
    if x:
        structure = x[1]
        elec = x[2]
        fit = x[3]
        pair_dists = None
        ffpkl = f"{elec}_{structure}_{fit}"
        #  load_ff
        ff = read_from_pickle(PKL_PATH / (ffpkl + ".pkl"))
        ff = next(ff)
        print(ff)

        loss = "jax"

        LJFF = fit_repeat(
            ff, 1000, f"{ffpkl}_fitted", bounds=ff.bounds, loss=loss, quiet="false"
        )
        #        print(LJFF.opt_parm)
        print(LJFF.opt_results)
        pickle_output(LJFF, f"{ffpkl}_fitted")


def run():
    for i in range(len(pkl_files)):
        f = pkl_files[i]
        exp = experiments[i]
        print(f"Running experiment {x}")
        print(f"Using pickle {f}")
        try:
            experiment(exp, f)
        except Exception as e:
            print(e)
            print("Failed to run experiment")


def experiment(exp, ffpkl):
    # Load data
    print(exp)
    _c = next(read_from_pickle(f"{ffpkl}"))
    print("Running Coulomb fit")
    ecol_fit(_c, ffpkl)
    elec_types = [exp[2]]
    if "ECOL" in elec_types:
        elec_types.append("fit_ECOL")
    for elec_type in elec_types:
        if exp[3] == "lj":
            print("Running LJ fit")
            rmse = lj_fit("fit_ECOL_" + ffpkl, elec_type=elec_type)
            print(f"LJ RMSE: {rmse} [kcal/mol]")
        elif exp[3] == "de":
            print("Running DE fit")
            rmse = de_fit("fit_ECOL_" + ffpkl, elec_type=elec_type)
            print(f"DE RMSE: {rmse} [kcal/mol]")
        else:
            raise ValueError("Invalid fit type")


def ecol_fit(_c, ffpkl):
    #  set the targets
    _c.intE = "ECOL"
    _c.set_targets()
    #  Coulomb fit
    fit_repeat(
        _c,
        NFIT_COUL,
        f"{ffpkl}_chgpen",
        bounds=CHGPEN_bound,
        loss="chgpen",
        quiet="true",
    )
    resx = _c.opt_parm
    loss = _c.get_loss_chgpen(resx)
    rmse = np.sqrt(loss)
    print(f"chgpen RMSE: {rmse} [kcal/mol]")
    _c.data["fit_ECOL"] = _c.eval_jax_chgpen(resx) + _c.data.ELEC

    #  plotting
    ax, cbar, stats = plot_energy_MSE(_c.data, "ECOL", "fit_ECOL", elec="ELEC")
    plt.show()
    plot_energy_MSE(_c.data, "ECOL", "ELEC", elec="ELEC")
    plt.show()

    print(f"ECOL RMSE: {stats['RMSE']} [kcal/mol]")
    print(f"ECOL rmse: {rmse} [kcal/mol]")
    print(f"Pickling to ff/fit_ECOL_{ffpkl}")
    pickle_output(_c, f"ff/fit_ECOL_{ffpkl[:-4]}")
    return rmse


def lj_fit(ffpkl, elec_type="fit_ECOL"):
    LJFF = next(read_from_pickle(PKL_PATH / f"ff/{ffpkl}"))
    LJFF.elec = elec_type
    LJFF.set_targets()
    #  reset the results dict
    LJFF.opt_results = []
    LJFF = fit_repeat(
        LJFF, NFIT_LJ, f"{ffpkl}_LJ", bounds=LJ_bound, loss="jax", quiet="true"
    )

    best = LJFF.get_best_parm()
    print(f"Best parm: {best}")
    res = LJFF.eval_jax(best)
    print("res", res)
    LJFF.data["fit_LJ"] = res + LJFF.data[elec_type]
    print(LJFF.data.columns)
    print(LJFF.data[["intE", "fit_LJ", "ECOL", "fit_ECOL"]].describe())

    _, _, stats = plot_energy_MSE(
        LJFF.data,
        "intE",
        "fit_LJ",
        elec="ELEC",
    )
    plt.show()
    return stats["RMSE"]


def de_fit(ffpkl, elec_type="fit_ECOL"):
    DEFF = next(read_from_pickle(PKL_PATH / f"ff/{ffpkl}"))
    DEFF.elec = elec_type
    DEFF.set_targets()
    #  reset the results dict
    DEFF.opt_results = []
    DEFF = fit_repeat(
        DEFF, NFIT_DE, "test", bounds=DE_bound, loss="jax_de", quiet="true"
    )

    DEFF.set_best_parm()
    best = DEFF.eval_jax_de(DEFF.opt_parm)
    DEFF.data["fit_DE"] = best
    ax, cbar, stats = plot_energy_MSE(
        DEFF.data,
        "intE",
        "fit_DE",
        elec="ELEC",
    )
    plt.show()

    return stats["RMSE"]


if __name__ == "__main__":
    jobs = loop()
    import argparse
    #  argument for which experiment to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", "-x", type=int, help="Experiment number")
    parser.add_argument("--m", "-m", action='store_true', help="Make the ff object")
    parser.add_argument("--f", "-f", action='store_true', help="Fit the ff object")
    args = parser.parse_args()
    print(args.x)
    job = jobs[args.x]    
    print(job)
    if args.m:
        print("Making ff object")
        make_ff_object(job)
    if args.f:
        print("Fitting ff object")
        ff = ff_fit(job)




