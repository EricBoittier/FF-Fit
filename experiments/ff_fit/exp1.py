from pathlib import Path
import pandas as pd
import jax; jax.config.update('jax_platform_name', 'cpu')
import itertools as it
import matplotlib.pyplot as plt

from ff_energy.ffe.constants import PDB_PATH

from ff_energy.ffe.potential import (
    LJ,
    DE,
    lj
)

from ff_energy.ffe.ff import FF

from ff_energy.plotting.fit_results import residuals_plot

from ff_energy.plotting.ffe_plots import plot_energy_MSE, plot_ff_fit

from ff_energy.ffe.ff_fit import (
    load_ff,
    fit_func,
    fit_repeat,
)
from ff_energy.utils.ffe_utils import pickle_output, read_from_pickle, str2int, \
    PKL_PATH, get_structures

from ff_energy.utils.json_utils import load_json

structure_data = {
    "dcm": "",
    "water_cluster": PKL_PATH / "20230823_water_clusters.pkl.pkl",
}

"""
Variables
"""
sig_bound = (0.25, 2.5)
ep_bound = (0.001, 1.0)
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
            pkl_file = PKL_PATH / "20230913_dcm.pkl.pkl"
            pkl_files.append(pkl_file)

        elif structure == "water_cluster":
            #  load the pickle files
            pkl_file = PKL_PATH / "20230823_water_clusters.pkl.pkl"
            pkl_files.append(pkl_file)
            pair_dists = next(read_from_pickle(PKL_PATH / "water_pc_pairs.pkl"))

        else:
            raise ValueError("Invalid structure")

        pkl_file = read_from_pickle(pkl_file)
        data = next(pkl_file)

        structs, pdbs  = get_structures(structure,
                                        pdbpath= PDB_PATH / structure)

        struct_data = structs[0]

        #  set the 2body terms
        # print("setting 2body")
        # for _ in structs:
        #     _.set_2body()

        dists = {
            str(Path(_.name).stem).split(".")[0]:
                _.distances for _ in structs
        }

        """
        FUNCTION TYPE
        """
        if fit == "lj":
            FUNC = LJ
            BOUNDS = []
            for i in range(len(set(struct_data.chm_typ))):
                BOUNDS.append(sig_bound)
            for i in range(len(set(struct_data.chm_typ))):
                BOUNDS.append(ep_bound)
        elif fit == "de":
            FUNC = DE
            BOUNDS = []
            for i in range(len(set(struct_data.chm_typ))):
                BOUNDS.append(sig_bound)
            for i in range(len(set(struct_data.chm_typ))):
                BOUNDS.append(ep_bound)
            BOUNDS.append(alpha_bound)
            BOUNDS.append(beta_bound)
        else:
            raise ValueError("Invalid fit type")

        print("bounds", BOUNDS)
        #  make the ff object
        ff = FF(
            data,
            dists,
            FUNC,
            BOUNDS,
            struct_data,
            elec=elec,
        )
        #  set the targets
        ff.num_segments = len(ff.data[elec].index)
        ff.set_targets()
        ##############################
        outes = ff.out_es
        print(ff.p, ff.opt_parm)
        flateval, sigma, episolon = ff.eval_jax_flat(ff.p)
        resids = outes - flateval
        # debugging
        print(ff.p)
        print("out es", outes)
        print("es shape", ff.out_es.shape)
        print("flat eval", flateval)
        print("flat eval shape", flateval.shape)
        print("dists", ff.out_dists)
        print("dists shape", ff.out_dists.shape)

        print(sigma)
        print(episolon)
        ff.debug_df["sigmas2"] = sigma
        ff.debug_df["epsilons2"] = episolon
        print(ff.debug_df["sigmas"], ff.debug_df["epsilons"])

        #  make the dataframe
        df_test = pd.DataFrame(
            {
                "target": outes,
                "residuals": resids,
                "vals": flateval
            }
        )  # drop the nans
        print("DF TEST")
        print(df_test.describe())
        print("DF DEBUG")
        ff.debug_df["jaxflat"] = flateval
        print(ff.debug_df)
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
    jaxeval, sigma, epsilon = ff.eval_jax(LJFF.get_best_parm())

    print("sigma::", sigma)
    print("epsilon::", epsilon)


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
    print("Running exp: ", args.x)
    job = jobs[args.x]
    print("Job: ", job)

    if args.m:
        print("Making ff object")
        make_ff_object(job)

    if args.f:
        print("Fitting ff object, repeats = {}".format(args.n))
        ff_fit(job, n=args.n)
