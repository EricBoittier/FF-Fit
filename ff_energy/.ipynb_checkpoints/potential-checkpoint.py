import numpy as np
import pandas as pd
import itertools

from ff_energy.structure import atom_key_pairs

def LJ(sig, ep, r, a=6, b=2, c=2):
    """
    Lennard-Jones potential for a pair of atoms
    """
    r6 = (sig / r) ** a
    return ep * (r6 ** b - c * r6)

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
    # print(aep, asig, bep, bsig)
    sig = (asig + bsig)
    ep = (aep * bep)**0.5
    # print(sig, ep)
    return LJ(sig, ep, r)

epsilons = {"OG311": -0.192, "CG331": -0.078, "HGP1":-0.046 ,"HGA3": -0.024,"OT": -0.1521,"HT": -0.0460}
rminhalfs = {"OG311": 1.765, "CG331": 2.050, "HGP1": 0.225,"HGA3": 1.340,"OT":1.7682 ,"HT": 0.2245,}
# epsilons = {"OG311": -0.20, "CG331": -0.08, "HGP1":-0.05 ,"HGA3": -0.03,"OT": -0.1521,"HT": -0.0460}
# rminhalfs = {"OG311": 1.79, "CG331": 2.08, "HGP1": 0.23,"HGA3": 1.36,"OT":1.7682 ,"HT": 0.2245,}


class FF:
    def __init__(self, data, dists, func, 
                 structure,
                 nobj=4, elec="ELEC"):
        self.data = data
        self.data[""] = 
        self.structure = structure
        self.atom_types = list(set([structure.atom_types[(a,b)] 
                                    for a,b in 
                                    zip(structure.restypes,
                                        structure.atomnames)]))
        self.atom_type_pairs = list(
            itertools.combinations_with_replacement(
                self.atom_types,2)
        )
        

        self.df = data.copy()
        self.dists = dists

        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.opt_parm = None
        self.opt_results = []
        self.mse_df = None

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists
        
    def loop_akp():
    # E = 0
    # for i, akp in enumerate(atom_key_pairs):
    #     e = np.sum(LJ_akp(dists[i],akp))
    #     if len(dists[i]) != 0 :
    #         print(akp,e)
    #     E += e
    # print(E)
        pass
    
    def LJ_(self,epsilons=None,rminhalfs=None):
        Es = []
        for dists in self.dists:
        
            E = 0
            for i,akp in enumerate(self.atom_type_pairs):
                e = np.sum(LJ_akp(dists[akp_indx[akp]],akp,
                                  epsilons=epsilons,rminhalfs=rminhalfs))
                # if len(dists[akp_indx[akp]]) != 0 :
                #     print(akp,e)
                E += e
            Es.append(E)
        return Es


    def eval_func(self, x):
        s = {}
        e = {}
        for i,atp in enumerate(self.atom_types):
            s[atp] =  x[i]
            e[atp] = x[i+len(self.atom_types)]
        return self.LJ_(epsilons=e,rminhalfs=s)

    def get_loss(self, x):
        tmp = self.data.copy()
        
        tmp["LJ"] = self.eval_func(x)
        #  get squared error
        tmp["LJ_SE"] = (tmp["intE"] - (tmp[self.elec] + tmp["LJ"])) ** 2
        loss = tmp["LJ_SE"].mean()
        self.mse_df = tmp
        return loss

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
        self.df["LJ"] = tmp["LJ"]
        self.df["ETOT_LJ"] = tmp["LJ"] + self.df[self.elec]

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
        tmp = self.eval_func(self.opt_parm)

        if not quiet:
            print("Set optimized parameters to FF object, self.df[\"LJ\"] is updated.")

        self.df["LJ"] = tmp["LJ"]
        self.df["ETOT_LJ"] = tmp["LJ"] + self.df[self.elec]

        return res

