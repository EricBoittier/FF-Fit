from functools import partial

from ff_energy.ffe.structure import atom_key_pairs

import jax
from jax import grad
from jax import jit
import jax.numpy as jnp
import jax.numpy as np

"""
SimTK_COULOMB_CONSTANT_IN_KCAL_ANGSTROM   2L
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

sig_bound = (0.05, 4.0)
ep_bound = (0.00001, 1.0)
LJ_bound = [(sig_bound), (sig_bound), (ep_bound), (ep_bound)]


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
    for i, (a, b) in enumerate(atom_key_pairs):
        print(a, b)
        aep, asig, bep, bsig = epsilons[a], rminhalfs[a], epsilons[b], rminhalfs[b]
        sig = asig + bsig
        ep = (aep * bep) ** 0.5
        sigs.append(sig)
        eps.append(ep)
    return sigs, eps


def LJ(sig, ep, r):
    """
    rmin = 2^(1/6) * sigma
        https://de.wikipedia.org/wiki/Lennard-Jones-Potential
    Lennard-Jones potential for a pair of atoms
    """
    a = 6
    b = 2
    # sig = sig / (2 ** (1 / 6))
    r6 = (sig / r) ** a
    return ep * (r6 ** b - 2 * r6)


@jit
def lj(sig, ep, r):
    """Lennard-Jones potential for a pair of atoms"""

    # sig = sig * (2 ** (1 / 6))

    r6 = (sig / r) ** 6

    return ep * (r6 ** 2 - 2 * r6)



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
def de(c, e, a, b, x):
    """
    Double exponential potential
    """
    return e * (
            ((b * np.exp(a)) / (a - b)) * np.exp(-a * (x / c))
            - ((a * np.exp(b)) / (a - b)) * np.exp(-b * (x / c))
    )


@jit
# def charge_penetration(cpt,r):
def charge_penetration(a1, a2, b1, b2, q1, q2, z1, z2, r):
    """
    https://pubs.acs.org/doi/10.1021/ct7000182
    """
    inv_r = 1 / r
    a1_term = 1 - jnp.exp(-a1 * r)
    a2_term = 1 - jnp.exp(-a2 * r)
    b1_term = 1 - jnp.exp(-b1 * r)
    b2_term = 1 - jnp.exp(-b2 * r)
    z1z2 = z1 * z2
    z1minq1 = z1 - q1
    z2minq2 = z2 - q2
    return (
            z1z2
            - (z1 * z2minq2 * a2_term + z2 * z1minq1 * a1_term)
            + z1minq1 * z2minq2 * b1_term * b2_term
    ) * inv_r


@partial(jit, static_argnames=["num_segments"])
def LJRUN(dists, indexs, groups, parms, num_segments):
    #  flat array of LJ energies
    LJE, sigma, epsilons = LJflat(dists, indexs, parms)
    #  sum the energies for each group
    OUT = jax.ops.segment_sum(LJE, groups, num_segments=num_segments)
    #  debug
    # jax.debug.print("shapeOUT {x}", x=OUT.shape)
    # jax.debug.print("out {x}", x=OUT)
    # jax.debug.print("shapeLJE {x}", x=LJE.shape)
    # jax.debug.print("LJE {x}", x=LJE)
    return OUT, sigma, epsilons


@partial(jit, static_argnames=["num_segments"])
def DERUN(dists, indexs, groups, parms, num_segments):
    DEE,_,_ = DEflat(dists, indexs, parms)
    OUT = jax.ops.segment_sum(DEE, groups, num_segments=num_segments)
    return OUT


@jit
def LJflat(dists, indexs, parms):
    # jax.debug.print("{p}", p=parms)
    n_parms = len(parms)
    n_types = n_parms // 2
    n_comb = n_types * (n_types + 1)
    #  above omits (ans/2) since the array is twice as long
    comb_parms = jnp.zeros(n_comb, dtype=jnp.float64)
    count = 0
    for a in range(n_types):
        for b in range(n_types):
            # jax.debug.print("count: {c} a: {a} b: {b}", a=a, b=b, c=count)
            if a <= b:
                # jax.debug.print("+{d} +{e}", d=n_types, e=n_comb // 2)
                comb_parms = comb_parms.at[count].set(
                    parms[a] + parms[b]
                )
                comb_parms = comb_parms.at[count + (n_comb // 2)].set(
                    jnp.sqrt(
                        parms[a + n_types] * parms[b + n_types]
                    )
                )
                # jax.debug.print("comb_parms: {c}", c=comb_parms)
                count += 1

    sigma = jnp.take(comb_parms, indexs,
                     unique_indices=False)

    eps = jnp.take(comb_parms, indexs + (n_comb // 2),
                   unique_indices=False)

    # #  run the flat LJ function
    LJE = lj(sigma, eps, dists)
    # #  debugging
    # jax.debug.print("indexs shape {x}", x=indexs.shape)
    # # jax.debug.print("indexs {x}", x=indexs)
    # jax.debug.print("dists shape {x}", x=dists.shape)
    # # jax.debug.print("dists {x}", x=dists.dtype)
    # jax.debug.print("comb_parms shape {x}", x=comb_parms.shape)
    # jax.debug.print("comb_parms {x}", x=comb_parms)
    # jax.debug.print("sigma_s {x}", x=sigma.shape)
    # jax.debug.print("sigma {x}", x=sigma)
    # jax.debug.print("eps_s {x}", x=eps.shape)
    # jax.debug.print("eps {x}", x=eps)

    return LJE, sigma, eps


# @jit
# def DEflat(dists, indexs, parms):
#     pparms = jnp.array(
#         [
#             2 * parms[0],
#             parms[0] + parms[1],
#             2 * parms[1],
#             parms[2],
#             jnp.sqrt((parms[2] * parms[3])),
#             parms[3],
#         ]
#     )
#     sigma = jnp.take(pparms, indexs, unique_indices=False)
#     eps = jnp.take(pparms, indexs + 3, unique_indices=False)
#     DEE = de(sigma, eps, parms[4], parms[5], dists)
#     return DEE

@jit
def DEflat(dists, indexs, parms):
    # jax.debug.print("{p}", p=parms)
    n_parms = len(parms) - 2
    n_types = n_parms // 2
    n_comb = n_types * (n_types + 1)
    #  above omits (ans/2) since the array is twice as long
    comb_parms = jnp.zeros(n_comb, dtype=jnp.float64)
    count = 0
    for a in range(n_types):
        for b in range(n_types):
            # jax.debug.print("count: {c} a: {a} b: {b}", a=a, b=b, c=count)
            if a <= b:
                # jax.debug.print("+{d} +{e}", d=n_types, e=n_comb // 2)
                comb_parms = comb_parms.at[count].set(
                    parms[a] + parms[b]
                )
                comb_parms = comb_parms.at[count + (n_comb // 2)].set(
                    jnp.sqrt(
                        parms[a + n_types] * parms[b + n_types]
                    )
                )
                # jax.debug.print("comb_parms: {c}", c=comb_parms)
                count += 1
    sigma = jnp.take(comb_parms, indexs,
                     unique_indices=False)
    eps = jnp.take(comb_parms, indexs + (n_comb // 2),
                   unique_indices=False)
    # #  run the flat function
    DEE = de(sigma, eps, parms[-2], parms[-1], dists)
    return DEE, sigma, eps

@partial(jit, static_argnames=["num_segments"])
def ecol_seg(outE, dcm_dists_labels, num_segments):
    return jax.ops.segment_sum(outE, dcm_dists_labels, num_segments=num_segments)


@partial(jit, static_argnames=["num_segments"])
def LJRUN_LOSS(dists, indexs, groups, parms, target, num_segments):
    # jax.debug.print("indexs {x}", x=indexs.shape)
    # # jax.debug.print(" {x}", x=indexs)
    # jax.debug.print("groups {x}", x=groups.shape)
    # # jax.debug.print(" {x}", x=groups)
    RES, _, _ = LJRUN(dists, indexs, groups, parms, num_segments=num_segments)
    # assert RES.shape == target.shape
    ERROR = RES - target
    LOSS = jnp.mean(ERROR ** 2)  # TODO:  dangerous to use nanmean here?
    # jax.debug.print("LOSS {x}", x=LOSS)
    return LOSS


@partial(jit, static_argnames=["num_segments"])
def DERUN_LOSS(dists, indexs, groups, parms, target, num_segments):
    # jax.debug.print("indexs {x}", x=indexs.shape)
    # # jax.debug.print(" {x}", x=indexs)
    # jax.debug.print("groups {x}", x=groups.shape)
    # # jax.debug.print(" {x}", x=groups)
    RES = DERUN(dists, indexs, groups, parms, num_segments=num_segments)
    # assert RES.shape == target.shape
    ERROR = RES - target
    LOSS = jnp.mean(ERROR ** 2)  # TODO:  dangerous to use nanmean here?
    # jax.debug.print("LOSS {x}", x=LOSS)
    return LOSS
    
"""ERROR = DERUN(dists, indexs, groups, parms, num_segments=num_segments) - target
    return jnp.mean(ERROR ** 2)


# @partial(jit, static_argnames=["num_segments"])
# def CHGPEN_LOSS(dists, indexs, groups, parms, target, num_segments):
#     ERROR = CHGPENRUN(dists, indexs, groups, parms, num_segments=num_segments) - target
#     return jnp.mean(ERROR**2)
"""

@partial(jit, static_argnames=["num_segments"])
def LJRUN_LOSS_GRAD(dists, indexs, groups, parms, target, num_segments):
    return grad(LJRUN_LOSS(dists, indexs, groups, parms, target, num_segments))
