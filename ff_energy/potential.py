import numpy as np
import pandas as pd
import itertools
# import numba
# from numba import vectorize, float64

from ff_energy.structure import atom_key_pairs, valid_atom_key_pairs


# @numba.njit (parallel=True, nopython=True)
# @numba.njit (nopython=True, parallel=True)
def LJ(sig, ep, r):
    """
    Lennard-Jones potential for a pair of atoms
    """
    a = 6
    b = 2
    c = 2
    r6 = (sig / r) ** a
    return ep * (r6 ** b - c * r6)


# vfunc_LJ = np.vectorize(LJ)
# vfunc_LJ = numba.vectorize([(np.float,)], nopython=True)(LJ)
# vfunc_LJ = vectorize([(float64,float64,float64)])(LJ)
vfunc_LJ = LJ


def freeLJ(sig, ep, r, a, b, c):
    """
    Lennard-Jones potential for a pair of atoms
    """
    return ep * ((sig / r) ** a - c * (sig / r) ** b)


#  double exp. pot.
#  https://chemrxiv.org/engage/chemrxiv/article-details/6401c0a163e8d44e594addea
def DE(x, a, b, c, e):
    """
    Double exponential potential
    """
    return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
                - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c)))


# def DEplus(x, a, b, c, e, f, g):
#     """
#     Double exponential potential
#     """
#     return e * (
#             (( b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
#                 - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))
#             + (((a * b) * np.exp(f)) / (a - b)) * np.exp(-f * (x / c))
#                 )

# def DEplus(x, a, b, c, e, f, g):
#     """
#     Double exponential potential
#     """
#     return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
#                 - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))) - f*(x/(g))**(-2)


def DEplus(x, a, b, c, e, f, g):
    """
    Double exponential potential
    """
    return e * (((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
                - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))) - f * (c / x) ** g


epsilons = {"OG311": -0.192, "CG331": -0.078, "HGP1": -0.046, "HGA3": -0.024, "OT": -0.1521, "HT": -0.0460}
rminhalfs = {"OG311": 1.765, "CG331": 2.050, "HGP1": 0.225, "HGA3": 1.340, "OT": 1.7682, "HT": 0.2245, }

akp_indx = {akp: i for
            i, akp in enumerate(atom_key_pairs)}
for i, akp in enumerate(atom_key_pairs):
    print(i, akp)


def Ecoloumb(q1, q2, r):
    """Calculate the coulombic energy between two charges,
    with atomic units and angstroms for distance"""
    coloumns_constant = 3.32063711e+2
    return coloumns_constant * q1 * q2 / r


def LJ_akp(r, akp, epsilons=None, rminhalfs=None):
    r = np.array(r)
    a, b = akp
    aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
    sig = (asig + bsig)
    ep = (aep * bep) ** 0.5
    # print(sig, ep)
    return LJ(sig, ep, r)


def combination_rules(atom_key_pairs, epsilons=None, rminhalfs=None):
    """Calculate the combination rules for the LJ potential"""
    sigs = []
    eps = []
    for a, b in atom_key_pairs:
        aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
        sig = (asig + bsig)
        ep = (aep * bep) ** 0.5
        sigs.append(sig)
        eps.append(ep)
    return sigs, eps


epsilons = {"OG311": -0.192, "CG331": -0.078, "HGP1": -0.046, "HGA3": -0.024, "OT": -0.1521, "HT": -0.0460}
rminhalfs = {"OG311": 1.765, "CG331": 2.050, "HGP1": 0.225, "HGA3": 1.340, "OT": 1.7682, "HT": 0.2245, }


# epsilons = {"OG311": -0.20, "CG331": -0.08, "HGP1":-0.05 ,"HGA3": -0.03,"OT": -0.1521,"HT": -0.0460}
# rminhalfs = {"OG311": 1.79, "CG331": 2.08, "HGP1": 0.23,"HGA3": 1.36,"OT":1.7682 ,"HT": 0.2245,}

class DistPrep:
    def __init__(self, dists):
        self.dists = dists


class FF:
    def __init__(self, data, dists, func,
                 bounds,
                 structure,
                 nobj=4, elec="ELEC"):
        self.data = data
        self.data_save = data.copy()

        self.structure = structure
        self.atom_types = list(set([structure.atom_types[(a, b)]
                                    for a, b in
                                    zip(structure.restypes,
                                        structure.atomnames)]))
        # print(self.atom_types)
        self.atom_type_pairs = valid_atom_key_pairs(self.atom_types)
        print(self.atom_type_pairs)
        self.df = data.copy()
        self.dists = dists
        self.do_cache = False

        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.opt_parm = None
        self.opt_results = []
        self.opt_results_df = []
        self.mse_df = None
        self.all_dists = None
        self.all_sig = None
        self.all_ep = None
        self.n_dists = []
        self.bounds = bounds

        for i, akp in enumerate(self.atom_type_pairs):
            print(i, akp, atom_key_pairs[akp_indx[akp]])

    def __repr__(self):
        return f"FF: {self.func.__name__}"

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists

    def LJ_(self, epsilons=None, rminhalfs=None, DISTS=None, data=None):
        if data is None:
            data = self.data
        if DISTS is None:
            DISTS = self.dists
        Es = []
        Keys = []
        sig, ep = combination_rules(self.atom_type_pairs, epsilons, rminhalfs)
        for k in self.data.index:
            dists = DISTS[k]
            E = 0
            for i, akp in enumerate(self.atom_type_pairs):
                # print(akp_indx[akp])
                if len(dists[akp_indx[akp]]) > 0:
                    ddists = np.array(dists[akp_indx[akp]])
                    e = np.sum(vfunc_LJ(sig[i], ep[i], ddists))
                    E += e
            Es.append(E)
            Keys.append(k)
        return pd.DataFrame(Es, index=Keys)

    def LJ_performace(self, res, data=None):
        if data is None:
            data = self.data.copy()
        data["LJ"] = res
        # print(res)
        data["VDW_ERROR"] = data["VDW"] - data["LJ"]
        data["VDW_SE"] = data["VDW_ERROR"] ** 2
        data["nb_intE"] = data[self.elec] + data["LJ"]
        data["SE"] = (data["intE"] - data["nb_intE"]) ** 2
        return data

    def eval_func(self, x):
        s = {}
        e = {}
        for i, atp in enumerate(self.atom_types):
            s[atp] = x[i]
            e[atp] = x[i + len(self.atom_types)]
        return self.LJ_(epsilons, rminhalfs)

    def get_loss(self, x):
        res = self.eval_func(x)
        tmp = self.LJ_performace(res)
        return tmp["SE"].mean()

    def get_best_loss(self):
        results = pd.DataFrame(self.opt_results)
        results["data"] = [list(_.index) for _ in self.opt_results_df]
        best = results[results["fun"] == results["fun"].min()]
        return best

    def get_best_df(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        return tmp

    def set_best_parm(self):
        best = self.get_best_loss()
        self.opt_parm = best["x"].values[0]
        print("Set optimized parameters to FF object, "
              "use FF.opt_parm to get the optimized parameters")

    def eval_best_parm(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")
        self.df = tmp

    def fit_repeat(self, N, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False):
        if bounds is None:
            bounds = self.bounds
        for i in range(N):
            self.fit_func(None, bounds=bounds, maxfev=maxfev, method=method, quiet=quiet)
        self.get_best_loss()
        self.eval_best_parm()

    def fit_func(self, x0, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False):
        from scipy.optimize import minimize

        if bounds is None:
            bounds = self.bounds

        if x0 is None and bounds is not None:
            x0 = [np.random.uniform(low=a, high=b) for a, b in bounds]

        if not quiet:
            print(f"Optimizing LJ parameters...\n"
                  f"function: {self.func.__name__}\n"
                  f"bounds: {bounds}\n"
                  f"maxfev: {maxfev}\n"
                  f"initial guess: {x0}")

        res = minimize(self.get_loss, x0, method=method,
                       tol=1e-6,
                       bounds=bounds,
                       options={"maxfev": maxfev})

        if not quiet:
            print("final_loss_fn: ", res.fun)
            print(res)

        self.opt_parm = res.x
        self.opt_results.append(res)
        # self.opt_results["data"] = self.data.copy() # save the data
        self.opt_results_df.append(self.eval_func(self.opt_parm))
        # tmp = self.eval_func(self.opt_parm)

        if not quiet:
            print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")

        return res
