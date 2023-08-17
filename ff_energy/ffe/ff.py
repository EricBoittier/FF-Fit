import typing

import pandas as pd
import re
import numpy as np
from ff_energy.utils.ffe_utils import str2int

from jax import grad
import jax.numpy as jnp

from ff_energy.ffe.structure import valid_atom_key_pairs
from ff_energy.ffe.potential import (
    LJflat,
    LJRUN_LOSS,
    LJRUN,
    combination_rules,
    akp_indx,
    DERUN,
    DERUN_LOSS,
    CHGPENRUN,
    CHGPEN_LOSS,
)
from ff_energy.ffe.potential import ecol, ecol_seg


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
        intE="intE",
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

        self.atom_type_pairs = valid_atom_key_pairs(self.atom_types)
        self.df = data.copy()
        self.dists = dists
        self.name = f"{func.__name__}_{structure.system_name}_{elec}"
        self.func = func
        self.nobj = nobj
        self.elec = elec
        self.intE = intE
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

        #  for jax
        self.out_groups_dict = None
        self.out_es = None
        self.out_groups = None
        self.out_akps = None
        self.targets = None
        self.nTargets = None
        self.p = None
        #  for jax ecol
        self.dcm_ecols = None
        self.dcm_c_idx1 = None
        self.dcm_c_idx2 = None
        self.dcm_c1s = None
        self.dcm_c2s = None
        self.dcm_pairs_idx = None
        self.dcm_dists = None
        self.dcm_dists_labels = None
        self.cluster_labels = None
        self.coloumb_init = False

        self.sort_data()

        if len(self.opt_results) > 0:
            self.p = self.get_best_parm()
        else:
            self.p = jnp.array([1.764e00, 2.500e-01, 1.687e-01, 6.871e-03])

        #  initialize the interaction energies
        self.set_intE()
        #  initialize the jax arrays
        self.jax_init()

    def __repr__(self):
        return (
            f"FF: {self.func.__name__}"
            f" {self.structure.system_name}"
            f" {self.elec}"
            f" {self.intern}"
            f" {self.intE}"
            f" (jax_coloumb: {self.coloumb_init})"
        )

    def set_intE(self, pairs=False):
        # Internal energies
        #  ab initio reference energies
        if self.intern == "Exact":
            if pairs:
                self.data["intE"] = self.data["P_intE"]
            else:
                self.data["intE"] = (
                    self.data["C_ENERGY"] - self.data["M_ENERGY"]
                ) * 627.509
        #  harmonic fit
        elif self.intern == "harmonic":
            if pairs:
                #  TODO: add harmonic
                self.data["intE"] = self.data["P_intE"]
            else:
                self.data["intE"] = (
                    self.data["C_ENERGY_kcalmol"] - self.data["p_m_E_tot"]
                )
        #  error
        else:
            #  if the internal energy is not supported, raise an error
            raise ValueError(f"intern = {self.intern} not implemented")

    def init_jax_col(self, col_dict):
        """Initialize jax arrays from a dictionary"""
        for k, v in col_dict.items():
            setattr(self, k, v)
        #  set the attribute to identify that the coulomb
        #  arrays have been initialized
        self.coloumb_init = True

    def jax_init(self, p=None):
        """Initialize the jax arrays"""
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
        group_key_dict = {g: str2int(g) for g in list(set(out_groups))}
        self.out_groups_dict = group_key_dict
        out_groups = jnp.array([group_key_dict[g] for g in out_groups])
        self.out_groups = jnp.array(out_groups)
        self.out_es = jnp.array(out_es)
        self.out_akps = jnp.array(out_akps)

    def set_targets(self):
        """Set the targets for the objective function: intE - elec = targets"""
        self.targets = jnp.array(
            self.data[self.intE].to_numpy() - jnp.array(self.data[self.elec].to_numpy())
        )

        self.nTargets = int(len(self.targets))
        assert self.nTargets == len(self.targets)
        assert isinstance(self.nTargets, typing.Hashable) is True

    def eval_coulomb(self, scale=1.0):
        """Evaluate the coulomb energy for each element in the dist. array"""
        outE = ecol(scale * self.dcm_c1s, scale * self.dcm_c2s, self.dcm_dists)
        return outE

    def get_coulomb(self, scale=1.0):
        """Get the coulomb energy for each segment"""
        outE = self.eval_coulomb(scale=scale)
        return ecol_seg(outE, self.cluster_labels, num_segments=self.nTargets)

    def set_dists(self, dists):
        """Overwrite distances"""
        self.dists = dists

    def sort_data(self):
        """Sort the data by some integer in the index"""
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

    def LJ_dists(self, epsilons=None, rminhalfs=None, DISTS=None, args=None):
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
        """evaluate the performance of the LJ potential"""
        if data is None:
            data = self.data.copy()
        data["LJ"] = res["LJ"].copy()
        data["nb_intE"] = data[self.elec] + data["LJ"]
        data["SE"] = (data["intE"] - data["nb_intE"]) ** 2
        return data.copy()

    def eval_func(self, x, func=None):
        """convert the input x into the parameters for the LJ potential
        and run some function [default func is self.LJ_]
        """
        if func is None:
            func = self.LJ_
        s = {}
        e = {}
        for i, atp in enumerate(self.atom_types):
            s[atp] = x[i]
            e[atp] = x[i + len(self.atom_types)]
        return func(e, s, args=x[len(self.atom_types) * 2 :])

    def eval_dist(self, x):
        """wrapper to evaluate the LJ potential and get the distances"""
        return self.eval_func(x, func=self.LJ_dists)

    def get_loss(self, x) -> float:
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        res = self.eval_func(x)
        tmp = self.LJ_performace(res)
        loss = tmp["SE"].mean()
        return loss

    def eval_jax_chgpen(self, x):
        return CHGPENRUN(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
        )

    def eval_jax(self, x):
        """evaluate the LJ potential"""
        LJE = LJRUN(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
        )
        return LJE

    def eval_jax_de(self, x):
        """evaluate the LJ potential"""
        LJE = DERUN(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
        )
        return LJE

    def eval_jax_flat(self, x):
        """evaluate the LJ potential over all distances"""
        return LJflat(self.out_dists, self.out_akps, x)

    def get_loss_jax(self, x) -> float:
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return LJRUN_LOSS(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
            self.targets,
            self.nTargets,
        )

    def get_loss_jax_de(self, x) -> float:
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return DERUN_LOSS(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
            self.targets,
            self.nTargets,
        )

    def get_loss_chgpen(self, x) -> float:
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return CHGPEN_LOSS(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
            self.targets,
            self.nTargets,
        )

    def get_loss_coulomb(self, x) -> float:
        """
        :param x:
        :return:
        """
        # the charge scale will be the final parameter
        scale = x[0]
        # set the elec column
        self.data[self.elec] = self.get_coulomb(scale=scale)
        # update the targets
        self.set_targets()
        # get the loss as usual
        parms = self.p
        return self.get_loss_jax(parms)

    def get_loss_lj_coulomb(self, x) -> float:
        """
        :param x:
        :return:
        """
        # the charge scale will be the final parameter
        scale = x[-1]
        # set the elec column
        self.data[self.elec] = self.get_coulomb(scale=scale)
        # update the targets
        self.set_targets()
        # get the loss as usual
        parms = x[:-1]
        return self.get_loss_jax(parms)

    def eval_lj_coulomb(self, x) -> float:
        """
        :param x:
        :return:
        """
        # the charge scale will be the final parameter
        scale = x[-1]
        # set the elec column
        self.data[self.elec] = self.get_coulomb(scale=scale)
        # update the targets
        self.set_targets()
        # get the loss as usual
        parms = x[:-1]
        return self.eval_jax(parms)

    def eval_coulomb_nb(self, x) -> float:
        """
        :param x: scale of the charges
        :return:
        """
        # the charge scale will be the final parameter
        scale = x[0]
        # set the elec column
        self.data[self.elec] = self.get_coulomb(scale=scale)
        # update the targets
        self.set_targets()
        # get the loss as usual
        parms = self.p
        return self.eval_jax(parms)

    def get_loss_grad(self, x) -> float:
        """
        get the mean squared error of the LJ potential
        :param x:
        :return:
        """
        return grad(LJRUN_LOSS, 3)(
            self.out_dists,
            self.out_akps,
            self.out_groups,
            x,
            self.targets,
            self.nTargets,
        )

    def get_best_loss(self) -> pd.DataFrame:
        """get the best loss"""
        results = pd.DataFrame(self.opt_results)
        # results["data"] = [list(_.index) for _ in self.opt_results_df]
        best = results[results["fun"] == results["fun"].min()]
        return best

    def get_best_df(self) -> pd.DataFrame:
        """get the best dataframe"""
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm)
        return tmp

    def set_best_parm(self) -> None:
        """set the optimized parameters to the FF object"""
        best = self.get_best_loss()
        self.opt_parm = best["x"].values[0]
        print(
            "Set optimized parameters to FF object, "
            "use FF.opt_parm to get the optimized parameters"
        )

    def get_random_parm(self) -> list:
        """get a random set of parameters"""
        return [np.random.uniform(low=a, high=b) for a, b in self.bounds]

    def get_best_parm(self) -> list:
        """get the best parameters"""
        best = self.get_best_loss()
        return best["x"].values[0]

    def eval_best_parm(self, func=None) -> pd.DataFrame:
        """evaluate the best parameters"""
        self.set_best_parm()
        tmp = self.eval_func(self.opt_parm, func=func)
        self.df = tmp
        return tmp
