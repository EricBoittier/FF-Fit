import sys

import numpy as np

from scipy.optimize import minimize

from ff_energy.utils.ffe_utils import charmm_jobs
from ff_energy.ffe.cli import load_config_maker
from ff_energy.ffe.potential import LJ, DE
from ff_energy.ffe.ff import FF
from ff_energy.ffe.data import Data
from pathlib import Path
from ff_energy.utils.ffe_utils import pickle_output, read_from_pickle, PKL_PATH

sig_bound = (0.001, 2.5)
ep_bound = (0.001, 2.5)
chg_bound = (100,2000)

CHGPEN_bound = [(chg_bound),(chg_bound),(chg_bound),(chg_bound),(0,2000)]
LJ_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound))
DE_bound = ((sig_bound), (sig_bound), (ep_bound), (ep_bound),
            (1, 8), (6, 20))

func_bounds = {"LJ": (LJ, LJ_bound), "DE": (DE, DE_bound)}


def load_ff(
    ff_name,
    structure,
    FUNC=LJ,
    BOUNDS=LJ_bound,
    pickled_ff=False,
    pickled_dists=False,
    pk=None,
    intern=False,
    elec=False,
):
    """
    Load the force field, if the force field is not pickled, it will be created
    :param ff_name:
    :param structure:
    :param FUNC:
    :param BOUNDS:
    :param pickled_ff:
    :param pickled_dists:
    :param pk:
    :param intern:
    :param elec:
    :return:
    """
    ff = None
    data_ = None
    ff_pkls = Path("pickles/ff")
    ff_pickles = ff_pkls.glob("*.pkl")
    ff_pickles = [_.name for _ in ff_pickles]
    struct_dist_pkls = PKL_PATH / "structures"
    struct_dist_pickles = struct_dist_pkls.glob("*.pkl")
    struct_dist_pickles = [_.name for _ in struct_dist_pickles]
    if len(struct_dist_pickles) > 0:
        pickled_dists = True
        print("Pickled structures/dists found: ", struct_dist_pickles)

    #  check if the pickles exists
    if pickled_ff:
        pickled_ff = (ff_pkls / f"{ff_name}.pkl").exists()
    if pickled_dists:
        pickled_dists = (struct_dist_pkls / f"{structure}.pkl").exists()

    if pickled_ff:
        print("Pickled FF exists!, loading: ", ff_pkls / f"{ff_name}.pkl")
        try:
            ff = next(read_from_pickle(ff_pkls / f"{ff_name}.pkl"))
            #  return the ff and the data
            return ff, ff.data
        except StopIteration:
            print("Pickle read failed...")
            pickled_ff = False

    if not pickled_ff:
        if pickled_dists:  # load the data
            print(f"Loading pickled distances/structure: {structure}")
            structures, dists = next(read_from_pickle(f"structures/{structure}.pkl"))
        else:  # compute the data
            print("No pickled distances/structure information, calculating:")
            CMS = load_config_maker("pbe0dz", structure, "pc")
            jobs = charmm_jobs(CMS)
            dists = {_.name.split(".")[0]: _.distances for _ in jobs[0].structures}
            structures = [_ for _ in jobs[0].structures]
            pickle_output((structures, dists), name=f"structures/{structure}")

        s = structures[0]
        data_pickle = (PKL_PATH / pk).exists()
        if data_pickle and pk is not None:
            #  read the data directly from the pickle
            print(f"loading data from pickle: {pk}")
            data_ = next(read_from_pickle(pk))
        elif pk is None:
            # error
            print("Data pickle requested but no pickle provided")
            sys.exit(1)
        else:
            #  load all the small pickles
            data_ = Data(pk)

        #  create the FF object
        ff = FF(
            data_.data,
            dists,
            FUNC,
            BOUNDS,
            s,
            intern=intern,
            elec=elec,
        )
    return ff, data_


def fit_repeat(
    ff,
    N,
    outname,
    bounds=None,
    maxfev=10000,
    method="Nelder-Mead",
    quiet=False,
    loss=None,
):
    if bounds is None:
        bounds = ff.bounds
    for i in range(N):
        fit_func(
            ff,
            None,
            bounds=bounds,
            maxfev=maxfev,
            method=method,
            quiet=quiet,
            loss=loss,
        )
    ff.get_best_loss()
    # ff.eval_best_parm()
    pickle_output(ff, outname)
    return ff


def fit_func(
    ff,
    x0,
    bounds=None,
    maxfev=10000,
    method="Nelder-Mead",
    loss="jax",
    quiet=False,
):
    if bounds is None:
        bounds = ff.bounds

    # set which func we're using
    whichLoss = {
        "standard": (ff.get_loss, ff.eval_func),
        "jax": (ff.get_loss_jax, ff.eval_jax),
        "jax_de": (ff.get_loss_jax_de, ff.eval_jax_de),
        "lj_ecol": (ff.get_loss_lj_coulomb, ff.eval_lj_coulomb),
        "ecol": (ff.get_loss_coulomb, ff.eval_coulomb_nb),
        "chgpen": (ff.get_loss_chgpen, ff.eval_jax_chgpen),
    }
    func, eval = whichLoss[loss]

    # make a uniform random guess if no x0 value is provided
    if x0 is None and bounds is not None:
        x0 = [np.random.uniform(low=a, high=b) for a, b in bounds]

    if not quiet:
        print(
            f"Optimizing LJ parameters...\n"
            f"function: {func.__name__}\n"
            f"bounds: {bounds}\n"
            f"maxfev: {maxfev}\n"
            f"initial guess: {x0}"
        )
    res = minimize(
        func,
        x0,
        method=method,
        tol=1e-6,
        bounds=bounds,
        options={
            "maxfev": maxfev,
            # "pgtol": 1e-8
        },
    )

    if not quiet:
        print("final_loss_fn: ", res.fun)
        print(res)

    ff.opt_parm = res.x
    ff.opt_results.append(res)
    ff.opt_results_df.append(eval(ff.opt_parm))

    return res


if __name__ == "__main__":
    """
    Run the program from the command line
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="FFfit",
        description="Fit a force field to a set of data",
        epilog="...",
    )
    print("----")

    parser.add_argument("-f", "--ff", help="Forcefield name", required=True)
    parser.add_argument("-s", "--structure", help="Structure name", required=True)
    parser.add_argument(
        "-n", "--n", help="Number of repeats", required=False, default=10
    )
    parser.add_argument("-k", "--k", help="Number of fits", required=False, default=10)
    parser.add_argument("-ft", "--fftype", help="Forcefield type", required=True)
    parser.add_argument(
        "-c", "--clip", help="Clip", required=False, default=0, type=int
    )
    parser.add_argument(
        "-o", "--outname", help="Outname", required=False, default=False
    )
    parser.add_argument("-p", "--pk", help="Pickle", required=False, default=None)
    parser.add_argument(
        "-sa", "--sample", help="Sample", required=False, default=500, type=int
    )
    parser.add_argument(
        "-dp", "--dp", help="Data Pickle", required=False, default=False
    )
    parser.add_argument(
        "-i",
        "--intern",
        help="Internal energy: [Exact] or harmonic",
        required=False,
        default="Exact",
    )
    parser.add_argument(
        "-e", "--elec", help="[ELEC] or ECOL", required=False, default="ELEC"
    )
    args = parser.parse_args()
    print(args)
    print("----")
    # load the force field
    func, bounds = func_bounds[args.fftype]
    ff, data = load_ff(
        args.ff,
        args.structure,
        FUNC=func,
        BOUNDS=bounds,
        pk=args.pk,
        intern=args.intern,
        elec=args.elec,
    )
    outname = f"{args.ff}_{ff.name}_{ff.intern}_{ff.elec}"

    # do the fitting
    fit_repeat(
        ff,
        int(args.n),
        outname,
        bounds=bounds,
    )
