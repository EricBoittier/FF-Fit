from functools import partial

import numpy as np

from ff_energy.ffe.structure import atom_key_pairs

import jax
from jax import grad
from jax import jit
import jax.numpy as jnp

"""
SimTK_COULOMB_CONSTANT_IN_KCAL_ANGSTROM   3.32063711e+2L
Coulomb's constant kappa = 1/(4*pi*e0) in kcal-Angstroms/e^2.
"""
coloumns_constant = 3.32063711e2

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


def LJ(sig, ep, r):
    """
    Lennard-Jones potential for a pair of atoms
    """
    a = 6
    b = 2
    c = 2
    r6 = (sig / r) ** a
    return ep * (r6**b - c * r6)


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


def Ecoloumb(q1, q2, r):
    """Calculate the coulombic energy between two charges,
    with atomic units and angstroms for distance"""
    return coloumns_constant * q1 * q2 / r


@jit
def ecol(q1, q2, r):
    """Calculate the coulombic energy between two charges,
    with atomic units and angstroms for distance"""
    return coloumns_constant * ((q1 * q2) / r)


@jit
def lj(sig, ep, r):
    """Lennard-Jones potential for a pair of atoms"""
    a = 6
    b = 2
    c = 2
    r6 = (sig / r) ** a
    return ep * (r6**b - c * r6)


@partial(jit, static_argnames=["num_segments"])
def LJRUN(dists, indexs, groups, parms, num_segments=500):
    LJE = LJflat(dists, indexs, parms)
    OUT = jax.ops.segment_sum(LJE, groups, num_segments=num_segments)
    return OUT


@jit
def LJflat(dists, indexs, parms):
    parms = jnp.array(
        [
            2 * parms[0],
            parms[0] + parms[1],
            2 * parms[1],
            parms[2],
            jnp.sqrt((parms[2] * parms[3])),
            parms[3],
        ]
    )
    sigma = jnp.take(parms, indexs, unique_indices=False)
    eps = jnp.take(parms, indexs + 3, unique_indices=False)
    LJE = lj(sigma, eps, dists)
    return LJE


@partial(jit, static_argnames=["num_segments"])
def ecol_seg(outE, dcm_dists_labels, num_segments=500):
    return jax.ops.segment_sum(outE, dcm_dists_labels, num_segments=num_segments)


@partial(jit, static_argnames=["num_segments"])
def LJRUN_LOSS(dists, indexs, groups, parms, target, num_segments=500):
    ERROR = LJRUN(dists, indexs, groups, parms, num_segments=num_segments) - target
    return jnp.mean(ERROR**2)


@partial(jit, static_argnames=["num_segments"])
def LJRUN_LOSS_GRAD(dists, indexs, groups, parms, target, num_segments):
    return grad(LJRUN_LOSS(dists, indexs, groups, parms, target, num_segments))
