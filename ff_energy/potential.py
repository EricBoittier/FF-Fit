import numpy as np
import pandas as pd
from scipy.optimize import minimize
import re

from ff_energy.structure import atom_key_pairs, valid_atom_key_pairs

import jax
from jax import grad
from jax import jit
import jax.numpy as jnp

H2KCALMOL = 627.503


def LJ(sig, ep, r):
    """
    Lennard-Jones potential for a pair of atoms
    """
    a = 6
    b = 2
    c = 2
    r6 = (sig / r) ** a
    return ep * (r6 ** b - c * r6)


def freeLJ(sig, ep, r, a, b, c):
    """
    Lennard-Jones potential for a pair of atoms
    """
    return ep * ((sig / r) ** a - c * (sig / r) ** b)


#  double exp. pot.
#  https://chemrxiv.org/engage/chemrxiv/article-details/6401c0a163e8d44e594addea
def DE(c, e, x, a, b):
    """
    Double exponential potential
    """
    return e * (
            ((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
            - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))
    )


def DEplus(x, a, b, c, e, f, g):
    """
    Double exponential potential
    """
    return (
            e
            * (
                    ((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
                    - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))
            )
            - f * (c / x) ** g
    )


epsilons = {
    "OG311": -0.192,
    "CG331": -0.078,
    "HGP1": -0.046,
    "HGA3": -0.024,
    "OT": -0.1521,
    "HT": -0.0460,
}
rminhalfs = {
    "OG311": 1.765,
    "CG331": 2.050,
    "HGP1": 0.225,
    "HGA3": 1.340,
    "OT": 1.7682,
    "HT": 0.2245,
}

akp_indx = {akp: i for i, akp in enumerate(atom_key_pairs)}


def Ecoloumb(q1, q2, r):
    """Calculate the coulombic energy between two charges,
    with atomic units and angstroms for distance"""
    coloumns_constant = 3.32063711e2
    return coloumns_constant * q1 * q2 / r


def LJ_akp(r, akp, epsilons=None, rminhalfs=None):
    r = np.array(r)
    a, b = akp
    aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
    sig = asig + bsig
    ep = (aep * bep) ** 0.5
    # print(sig, ep)
    return LJ(sig, ep, r)


def combination_rules(atom_key_pairs, epsilons=None, rminhalfs=None):
    """Calculate the combination rules for the LJ potential"""
    sigs = []
    eps = []
    for a, b in atom_key_pairs:
        aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
        sig = asig + bsig
        ep = (aep * bep) ** 0.5
        sigs.append(sig)
        eps.append(ep)
    return sigs, eps


@jit
def lj(sig, ep, r):
    """Lennard-Jones potential for a pair of atoms"""
    a = 6
    b = 2
    c = 2
    r6 = (sig / r) ** a
    return ep * (r6 ** b - c * r6)


@jit
def LJRUN(dists, indexs, groups, parms):
    LJE = LJflat(dists, indexs, parms)
    OUT = jax.ops.segment_sum(LJE, groups, num_segments=500)
    return OUT


@jit
def LJflat(dists, indexs, parms):
    parms = jnp.array([2 * parms[0], parms[0] + parms[1], 2 * parms[1],
                       parms[2], jnp.sqrt((parms[2] * parms[3])), parms[3]])
    sigma = jnp.take(parms, indexs, unique_indices=False)
    eps = jnp.take(parms, indexs + 3, unique_indices=False)
    LJE = lj(sigma, eps, dists)
    return LJE


@jit
def LJRUN_LOSS(dists, indexs, groups, parms, target):
    ERROR = LJRUN(dists, indexs, groups, parms) - target
    return jnp.mean(ERROR ** 2)

@jit
def LJRUN_LOSS_GRAD(dists, indexs, groups, parms, target):
    return grad(LJRUN_LOSS(dists, indexs, groups, parms, target))

class DistPrep:
    def __init__(self, dists):
        self.dists = dists


class FF:
    def __init__(
            self,
            data,
            dists,
            func,
            bounds,
            structure,
            nobj=4,
            elec="ELEC",
            intern="Exact",
            pairs=False,
    ):
        self.data = data
        #  make a dummy zero column for the energy
        self.data["DUMMY"] = len(self.data) * [0]
        self.data_save = data.copy()

        self.structure = structure
        self.atom_types = list(
            set(
                [
                    structure.atom_types[(a, b)]
                    for a, b in zip(structure.restypes, structure.atomnames)
                ]
            )
        )
        # print(self.atom_types)
        self.atom_type_pairs = valid_atom_key_pairs(self.atom_types)
        self.df = data.copy()
        self.dists = dists
        self.name = f"{func.__name__}_{structure.system_name}_{elec}"
        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.intern = intern
        self.opt_parm = None
        self.opt_results = []
        self.opt_results_df = []
        self.mse_df = None
        self.all_dists = None
        self.all_sig = None
        self.all_ep = None
        self.n_dists = []
        self.bounds = bounds
        self.out_dists = None

        self.out_groups_dict = None
        self.out_es = None
        self.out_groups = None
        self.out_akps = None
        self.targets = None
        self.p = None

        self.sort_data()

        if len(self.opt_results) > 0:
            self.p = self.get_best_parm()
        else:
            self.p = jnp.array([1.764e00, 2.500e-01, 1.687e-01, 6.871e-03])

        # Internal energies
        if self.intern == "Exact":
            if pairs:
                self.data["intE"] = self.data["p_int_ENERGY"] * H2KCALMOL
            else:
                self.data["intE"] = self.data["C_ENERGY_kcalmol"] - self.data["m_E_tot"]
        elif self.intern == "harmonic":
            if pairs:
                #  TODO: add harmonic
                self.data["intE"] = self.data["p_int_ENERGY"] * H2KCALMOL
            else:
                self.data["intE"] = (
                        self.data["C_ENERGY_kcalmol"] - self.data["p_m_E_tot"]
                )
        else:
            #  if the internal energy is not supported, raise an error
            raise ValueError(f"intern = {self.intern} not implemented")

        self.jax_init()

    def jax_init(self, p=None):
        if p is None:
            p = self.p
        #  Jax arrays
        self.set_targets()
        (
            out_dists,
            out_groups,
            out_akps,
            out_ks,
            out_es,
            out_sig,
            out_ep,
        ) = self.eval_dist(p)
        self.out_dists = jnp.array(out_dists)
        #  turn groups (str) into group_keys (int)
        group_key_dict = {
            g: int(re.sub("[^0-9]", "", g)) for g in list(set(out_groups))
        }
        self.out_groups_dict = group_key_dict
        out_groups = jnp.array([group_key_dict[g] for g in out_groups])
        self.out_groups = jnp.array(out_groups)
        self.out_es = jnp.array(out_es)
        self.out_akps = jnp.array(out_akps)

    def set_targets(self):
        self.targets = jnp.array(
            jnp.array(
                self.data["intE"].to_numpy()
                - jnp.array(self.data[self.elec].to_numpy())
            )
        )

    def __repr__(self):
        return f"FF: {self.func.__name__}"

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists

    def sort_data(self):
        self.data["k"] = [int(re.sub("[^0-9]", "", i)) for i in self.data.index]
        self.data = self.data.sort_values(by="k")

    def LJ_(self, epsilons=None, rminhalfs=None, DISTS=None, data=None, args=None):
        """pairwise interactions"""
        if DISTS is None:
            DISTS = self.dists
        # outputs
        Es = []
        #  calculate combination rules
        sig, ep = combination_rules(self.atom_type_pairs, epsilons, rminhalfs)
        for k in self.data.index:
            # get the distance
            dists = DISTS[k]
            # calculate the energy
            E = 0
            #  loop over atom pairs
            for i, akp in enumerate(self.atom_type_pairs):
                #  if there are distances for this atom pair
                if len(dists[akp_indx[akp]]) > 0:
                    ddists = np.array(dists[akp_indx[akp]])
                    e = np.sum(self.func(sig[i], ep[i], ddists, *args))
                    E += e
            Es.append(E)
        return pd.DataFrame({"LJ": Es}, index=self.data.index)

    def LJ_dists(self, epsilons=None, rminhalfs=None, DISTS=None, data=None, args=None):
        """pairwise interactions"""
        if DISTS is None:
            DISTS = self.dists
        # outputs
        #  calculate combination rules
        sig, ep = combination_rules(self.atom_type_pairs, epsilons, rminhalfs)
        out_dists = []
        out_akps = []
        out_groups = []
        out_ks = []
        out_es = []
        out_sig = []
        out_ep = []
        for ik, k in enumerate(self.data.index):
            # get the distance
            dists = DISTS[k]
            #  loop over atom pairs
            for i, akp in enumerate(self.atom_type_pairs):
                #  if there are distances for this atom pair
                if len(dists[akp_indx[akp]]) > 0:
                    ddists = np.array(dists[akp_indx[akp]]).flatten()
                    for d in ddists:
                        out_dists.append(d)
                        out_akps.append(i)
                        out_groups.append(k)
                        out_ks.append(ik)
                        out_ep.append(ep[i])
                        out_sig.append(sig[i])
                        out_es.append(self.func(sig[i], ep[i], d, *args))

        out = [
            _
            for _ in [out_dists, out_groups, out_akps, out_ks, out_es, out_sig, out_ep]
        ]
        return out

    def LJ_performace(self, res, data=None):
        if data is None:
            data = self.data.copy()
        data["LJ"] = res["LJ"].copy()
        data["nb_intE"] = data[self.elec] + data["LJ"]
        data["SE"] = (data["intE"] - data["nb_intE"]) ** 2
        return data.copy()

    def eval_func(self, x):
        s = {}
        e = {}
        for i, atp in enumerate(self.atom_types):
            s[atp] = x[i]
            e[atp] = x[i + len(self.atom_types)]
        return self.LJ_(e, s, args=x[len(self.atom_types) * 2:])

    def eval_dist(self, x):
        s = {}
        e = {}
        for i, atp in enumerate(self.atom_types):
            s[atp] = x[i]
            e[atp] = x[i + len(self.atom_types)]
        return self.LJ_dists(e, s, args=x[len(self.atom_types) * 2:])

    def get_loss(self, x):
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        res = self.eval_func(x)
        tmp = self.LJ_performace(res)
        loss = tmp["SE"].mean()
        return loss

    def eval_jax(self, x):
        LJE = LJRUN(
            self.out_dists, self.out_akps, self.out_groups, x,
        )
        return LJE

    def eval_jax_flat(self, x):
        return LJflat(self.out_dists, self.out_akps, x)

    def get_loss_jax(self, x):
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return LJRUN_LOSS(
            self.out_dists, self.out_akps, self.out_groups, x,
            self.targets,
        )


    def get_loss_grad(self, x):
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return grad(LJRUN_LOSS,3)(
            self.out_dists, self.out_akps, self.out_groups, x,
            self.targets,
        )

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
        print(
            "Set optimized parameters to FF object, "
            "use FF.opt_parm to get the optimized parameters"
        )

    def get_random_parm(self):
        return [np.random.uniform(low=a, high=b) for a, b in self.bounds]

    def get_best_parm(self):
        best = self.get_best_loss()
        return best["x"].values[0]

    def eval_best_parm(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        print('Set optimized parameters to FF object, self.df["LJ"] is updated.')
        self.df = tmp
        return tmp

    def fit_repeat(
            self, N, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False
    ):
        if bounds is None:
            bounds = self.bounds
        for i in range(N):
            self.fit_func(
                None, bounds=bounds, maxfev=maxfev, method=method, quiet=quiet
            )
        self.get_best_loss()
        self.eval_best_parm()

    def fit_func(
            self, x0, bounds=None, maxfev=10000, method="Nelder-Mead", loss="standard", quiet=False
    ):
        if bounds is None:
            bounds = self.bounds

        # set which func we're using
        whichLoss = {"standard": self.get_loss, "jax": self.get_loss_jax}
        func = whichLoss[loss]

        # make a uniform random guess if no x0 value is provided
        if x0 is None and bounds is not None:
            x0 = [np.random.uniform(low=a, high=b) for a, b in bounds]

        if not quiet:
            print(
                f"Optimizing LJ parameters...\n"
                f"function: {self.func.__name__}\n"
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
            options={"maxfev": maxfev},
        )

        if not quiet:
            print("final_loss_fn: ", res.fun)
            print(res)

        self.opt_parm = res.x
        self.opt_results.append(res)
        self.opt_results_df.append(self.eval_func(self.opt_parm))

        if not quiet:
            print('Set optimized parameters to FF object, self.df["LJ"] is updated.')

        return res
