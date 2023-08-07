from pathlib import Path
import pandas as pd
import jax.numpy as jnp
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ff_energy.ffe.plot import (
    plot_energy_MSE, plot_ff_fit
)
from ff_energy.ffe.ff_fit import (
    load_ff, fit_func, fit_repeat,
)
from ff_energy.utils.ffe_utils import (
    pickle_output, read_from_pickle, str2int, PKL_PATH
)
from ff_energy.utils.json_utils import (
    load_json
)

"""
Variables
"""
sig_bound = (0.001, 2.5)
ep_bound = (0.001, 2.5)
chg_bound = (100, 2000)
CHGPEN_bound = [(chg_bound), (chg_bound), (chg_bound),
                (chg_bound), (0, 2000)]
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
    pkl_file = f"{x[2]}_{x[0]}_{x[1]}.ff.pkl_{x[3].upper()}" \
               f".pkl"
    print(pkl_file)
    if x[3] != "de":
        pkl_files.append(pkl_file)

print("Pickles path")
print(PKL_PATH)


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
            rmse = lj_fit("fit_ECOL_" + ffpkl,
                          elec_type=elec_type)
            print(f"LJ RMSE: {rmse} [kcal/mol]")
        elif exp[3] == "de":
            print("Running DE fit")
            rmse = de_fit("fit_ECOL_" + ffpkl,
                          elec_type=elec_type)
            print(f"DE RMSE: {rmse} [kcal/mol]")
        else:
            raise ValueError("Invalid fit type")


def ecol_fit(_c, ffpkl):
    #  set the targets
    _c.intE = "ECOL"
    _c.set_targets()
    #  Coulomb fit
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
    _c.data["fit_ECOL"] = _c.eval_jax_chgpen(resx) \
                          + _c.data.ELEC

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
    LJFF = fit_repeat(LJFF,
                      NFIT_LJ,
                      f"{ffpkl}_LJ",
                      bounds=LJ_bound,
                      loss="jax",
                      quiet='true'
                      )

    best = LJFF.get_best_parm()
    print(f"Best parm: {best}")
    res = LJFF.eval_jax(best)
    print("res", res)
    LJFF.data["fit_LJ"] = res + LJFF.data[elec_type]
    print(LJFF.data.columns)
    print(LJFF.data[
              ["intE", "fit_LJ", "ECOL", "fit_ECOL"]
          ].describe()
          )

    _, _, stats = plot_energy_MSE(LJFF.data,
                                  "intE", "fit_LJ",
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
    DEFF = fit_repeat(DEFF,
                      NFIT_DE,
                      "test",
                      bounds=DE_bound,
                      loss="jax_de",
                      quiet='true'
                      )

    DEFF.set_best_parm()
    best = DEFF.eval_jax_de(DEFF.opt_parm)
    DEFF.data["fit_DE"] = best
    ax, cbar, stats = plot_energy_MSE(DEFF.data,
                                      "intE", "fit_DE",
                                      elec="ELEC",
                                      )
    plt.show()

    return stats["RMSE"]


run()
