import numpy as np
import pandas as pd
import itertools
import numba
from numba import vectorize, float64

from ff_energy.structure import atom_key_pairs, valid_atom_key_pairs

# @numba.njit (parallel=True, nopython=True)
@numba.njit (nopython=True)
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
# vfunc_LJ = vectorize([(float64,float64,float64)], nopython=True)(LJ)
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
                - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))) - f * (c/x)**g


epsilons = {"OG311": -0.192, "CG331": -0.078, "HGP1":-0.046 ,"HGA3": -0.024,"OT": -0.1521,"HT": -0.0460}
rminhalfs = {"OG311": 1.765, "CG331": 2.050, "HGP1": 0.225,"HGA3": 1.340,"OT":1.7682 ,"HT": 0.2245,}

akp_indx = {akp:i for 
            i, akp in enumerate(atom_key_pairs)}

def Ecoloumb(q1,q2,r):
    """Calculate the coulombic energy between two charges,
    with atomic units and angstroms for distance"""
    coloumns_constant = 3.32063711e+2
    return coloumns_constant * q1*q2/r

def LJ_akp(r, akp,epsilons=None,rminhalfs=None):
    r = np.array(r)
    a,b = akp
    aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
    sig = (asig + bsig)
    ep = (aep * bep)**0.5
    # print(sig, ep)
    return LJ(sig, ep, r)

def combination_rules(atom_key_pairs,epsilons=None,rminhalfs=None):
    """Calculate the combination rules for the LJ potential"""
    sigs = []
    eps = []
    for a,b in atom_key_pairs:
        aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
        sig = (asig + bsig)
        ep = (aep * bep)**0.5
        sigs.append(sig)
        eps.append(ep)
    return sigs, eps

epsilons = {"OG311": -0.192, "CG331": -0.078, "HGP1":-0.046 ,"HGA3": -0.024,"OT": -0.1521,"HT": -0.0460}
rminhalfs = {"OG311": 1.765, "CG331": 2.050, "HGP1": 0.225,"HGA3": 1.340,"OT":1.7682 ,"HT": 0.2245,}
# epsilons = {"OG311": -0.20, "CG331": -0.08, "HGP1":-0.05 ,"HGA3": -0.03,"OT": -0.1521,"HT": -0.0460}
# rminhalfs = {"OG311": 1.79, "CG331": 2.08, "HGP1": 0.23,"HGA3": 1.36,"OT":1.7682 ,"HT": 0.2245,}

class DistPrep:
    def __init__(self, dists):
        self.dists = dists




class FF:
    def __init__(self, data, dists, func, 
                 structure,
                 nobj=4, elec="ELEC"):
        self.data = data

        self.structure = structure
        self.atom_types = list(set([structure.atom_types[(a,b)] 
                                    for a,b in 
                                    zip(structure.restypes,
                                        structure.atomnames)]))
        # print(self.atom_types)
        self.atom_type_pairs = valid_atom_key_pairs(self.atom_types)

        self.df = data.copy()
        self.dists = dists

        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.opt_parm = None
        self.opt_results = []
        self.mse_df = None
        self.all_dists = None
        self.all_sig = None
        self.all_ep = None
        self.n_dists = []
        self.caches(data,epsilons,rminhalfs)

        # self.LJ_performace(self.data)

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists
        

    def LJ_(self,epsilons=None,rminhalfs=None):
        Es = []
        sig, ep = combination_rules(self.atom_type_pairs,epsilons,rminhalfs)
        for dists in self.dists:
            E = 0
            for i,akp in enumerate(self.atom_type_pairs):
                dists = np.array(dists[akp_indx[akp]])
                e = np.sum(LJ(sig[i],ep[i],dists))
                E += e
            Es.append(E)
        return Es

    # @numba.njit(nopython=True)
    def cache_LJ(self,epsilons,rminhalfs):
        rminhalfs, epsilons = combination_rules(self.atom_type_pairs,
                                    epsilons,
                                    rminhalfs)
        # print((self.all_sig[0]))
        for di, n_dists in enumerate(self.n_dists):
            s = 0
            # print((self.all_sig[di]))
            # self.all_sig[di] = 0.0 * (self.all_sig[di])
            # self.all_ep[di] = 0.0 * (self.all_ep[di])
            for i,akp in enumerate(self.atom_type_pairs):
                # print("i:", i, akp, n_dists[i], s)
                for j in range(n_dists[i]):
                    self.all_sig[di][s] = rminhalfs[i]
                    self.all_ep[di][s] = epsilons[i]
                    s += 1




    def caches(self, data,epsilons=None,rminhalfs=None):
        self.all_dists = []
        self.all_sig = []
        self.all_ep = []
        self.n_dists = []
        keys = data.index
        sig, ep = combination_rules(self.atom_type_pairs,
                                    epsilons,
                                    rminhalfs)

        for di, key in enumerate(keys):
            dists = self.dists[key]
            self.all_dists.append([])
            self.all_sig.append([])
            self.all_ep.append([])
            self.n_dists.append([])
            # print("di:", di, len(dists))
            for i, akp in enumerate(self.atom_type_pairs):
                # print(akp)
                _dists_ = dists[akp_indx[akp]]
                n_dists = len(list(itertools.chain.from_iterable(_dists_)))
                self.n_dists[di].append(n_dists)
                # print(len(_dists_), n_dists)
                self.all_dists[di].extend(itertools.chain.from_iterable(_dists_))
                # print(i, di, n_dists,len([sig[i]]*n_dists))
                self.all_sig[di].extend([sig[i]]*n_dists)
                self.all_ep[di].extend([ep[i]]*n_dists)

            self.all_ep[di] = np.array(self.all_ep[di])
            self.all_sig[di] = np.array(self.all_sig[di])
            self.all_dists[di] = np.array(self.all_dists[di])

        # print(self.all_sig[0])

    def LJ_performace(self, data):

        data["LJ"] = [np.sum(vfunc_LJ(self.all_sig[i],
                                        self.all_ep[i],
                                        self.all_dists[i])) for i in range(len(self.all_sig))]
        data["VDW_ERROR"] = data["VDW"] - data["LJ"]
        data["nb_intE"] = data[self.elec] + data["LJ"]
        data["SE"] = (data["intE"] - data["nb_intE"])**2

        # print(data.head())
        # print("LJ", )
        # print("LJ", sum(LJ(all_sig[1], all_ep[1], all_dists[1])))
        # print(self.df["LJ"].head())
        # print("LJ MSE:", data["SE"].mean())
        # print("LJ RMSE:", data["SE"].mean()**0.5)
        return data


    """
    The most optimized pair pot. would:
    input step:
         - precompute all distances

     - precompute all combination rules
     - precompute all terms needed for the LJ MSE terms
     
     all_dists *for* all_pair_combos... []
     all_sigma *for* all_pair_combos... []
     all_eps *for* all_pair_combos... []
     
     LJ ( sig, ep, r_min)
     
    """


    def eval_func(self, x):
        s = {}
        e = {}
        for i,atp in enumerate(self.atom_types):
            s[atp] =  x[i]
            e[atp] = x[i+len(self.atom_types)]

        # self.caches(self.data, epsilons=e,rminhalfs=s)
        self.cache_LJ(epsilons=e,rminhalfs=s)
        return self.LJ_performace(self.data)

    def get_loss(self, x):
        tmp = self.eval_func(x)
        return tmp["SE"].mean()

    def get_best_loss(self):
        results = pd.DataFrame(self.opt_results)
        best = results[results["fun"] == results["fun"].min()]
        return best
    
    def get_best_df(self):
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        #  get squared error
        tmp["LJ_SE"] = (tmp["ETOT"] - (tmp[self.elec] + tmp["LJ"])) ** 2
        loss = tmp["LJ_SE"].mean()
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
        for i in range(N):
            self.fit_func(None, bounds=bounds, maxfev=maxfev, method=method, quiet=quiet)
        self.get_best_loss()
        self.eval_best_parm()

    def fit_func(self, x0, bounds=None, maxfev=10000, method="Nelder-Mead", quiet=False):
        from scipy.optimize import minimize

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
        # tmp = self.eval_func(self.opt_parm)

        if not quiet:
            print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")

        # self.df["LJ"] = tmp
        # self.df["ETOT_LJ"] = self.df["LJ"] + self.df[self.elec]

        return res

