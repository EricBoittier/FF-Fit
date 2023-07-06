from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ff_energy.ffe.ffe_utils import read_from_pickle
import jax.numpy as jnp
from ff_energy.ffe.plot import plot_energy_MSE, plot_ff_fit
from ff_energy.ffe.structure import atom_key_pairs
from ff_energy.ffe.potential import LJ, akp_indx
from ff_energy.ffe.ff import FF
from ff_energy.ffe.ff_fit import load_ff, fit_func, fit_repeat
from ff_energy.ffe.data import pairs_data
from ff_energy.ffe.ffe_utils import pickle_output, read_from_pickle, str2int, PKL_PATH
from ff_energy.exp_help.helpers import load_json
import itertools as it

"""
Variables
"""
sig_bound = (0.001, 2.5)
ep_bound = (0.001, 2.5)
chg_bound = (100, 2000)
CHGPEN_bound = [(chg_bound), (chg_bound), (chg_bound), (chg_bound), (0, 2000)]
LJ_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound))
DE_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound),
            (1, 8), (6, 20))
NFIT_COUL = 1
NFIT_LJ = 4
NFIT_DE = 6
pkl_files = []
json_object = load_json("exp1.json")
experiments = list(it.product(*json_object.values()))
print(f"N experiments: {len(experiments)}")
for i, x in enumerate(experiments):
    print(f"Experiment {i}: {x}")
    pkl_file = f"{x[2]}_{x[0]}_{x[1]}.ff.pkl_{x[3].upper()}.pkl"
    print(pkl_file)
    if x[3] != "de":
        pkl_files.append(pkl_file)




print("Pickles path")
print(PKL_PATH)

def run():
    experiment(pkl_files[0])



def experiment(ffpkl):
    # Load data
    _c = next(read_from_pickle(f"{ffpkl}"))
    print("Running Coulomb fit")
    ecol_fit(_c, ffpkl)
    print("Running LJ fit")
    rmse = lj_fit(ffpkl)
    print(f"LJ RMSE: {rmse} [kcal/mol]")
    print("Running DE fit")
    rmse = de_fit(ffpkl)
    print(f"DE RMSE: {rmse} [kcal/mol]")


def get_rmse_res(ff):
    best = list(pd.DataFrame(
        ff.opt_results)
                .sort_values("fun").fun)[0]
    return np.sqrt(best)


def ecol_fit(_c, ffpkl):
    _c.intE = "ECOL"
    _c.set_targets()
    fit_repeat(_c,
               NFIT_COUL,
               f"{ffpkl}_chgpen",
               bounds=CHGPEN_bound,
               loss="chgpen",
               quiet='true'
               )
    resx = _c.opt_parm
    loss = _c.get_loss_chgpen(resx)
    rmse = np.sqrt(loss)
    print(f"chgpen RMSE: {rmse} [kcal/mol]")
    _c.data["fit_ECOL"] = _c.eval_jax_chgpen(resx) + _c.data.ELEC

    #  plotting
    plot_energy_MSE(_c.data, "ECOL", "fit_ECOL", elec="ELEC")
    plot_energy_MSE(_c.data, "ECOL", "ELEC", elec="ELEC")
    pickle_output(_c, f"ff/fit_ECOL_{ffpkl}.pkl")
    return rmse


def lj_fit(ffpkl, type="fit_ECOL"):
    LJFF = next(read_from_pickle(f"{ffpkl}"))
    LJFF.elec = type
    fit_repeat(LJFF,
               NFIT_LJ,
               f"{ffpkl}_LJ",
               bounds=LJ_bound,
               loss="jax",
               quiet='true'
               )
    return get_rmse_res(LJFF)


def de_fit(ffpkl, type="fit_ECOL"):
    DEFF = next(read_from_pickle(f"{ffpkl}"))
    DEFF.elec = type
    fit_repeat(DEFF,
               NFIT_DE,
               "test",
               bounds=DE_bound,
               loss="jax_de",
               quiet='true'
               )
    return get_rmse_res(DEFF)


run()

