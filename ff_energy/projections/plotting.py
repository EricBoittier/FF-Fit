import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ff_energy.projections.dimensionality import get_pca
from ff_energy.projections.dscribe_utils import weighting_exp


import patchworklib as pw



def plot_pca(pca, x, ax=None, c=None):
    """
    Plot PCA of data
    :param pca:
    :param x:
    :return:
    """
    if ax is None:
        plt.scatter(x[:, 0], x[:, 1], c=c)
        ax = plt.gca()
    else:
        ax.scatter(x[:, 0], x[:, 1], c=c)
    # label explained variance
    exp_var = pca.explained_variance_ratio_
    exp_var_cum = np.cumsum(exp_var)
    xlabel = f"PCA1 ({exp_var[0]:.2f},{exp_var_cum[0]:.2f})"
    ylabel = f"PCA2 ({exp_var[1]:.2f},{exp_var_cum[1]:.2f})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.title("PCA of SOAP")
    ax.grid(which="both")
    return ax


def plot_values_of_l(function, ase_atoms, c=None):
    l_values = range(1, 4)
    soaps = [
        function(average="inner", lmax=l, rcut=None, weighting=weighting_exp).create(
            ase_atoms
        )
        for l in l_values
    ]
    for s in soaps:
        print(s.shape)
        # print(s[:1])

    def reshape_soap(soap):
        if len(soap.shape) == 3:
            soap = soap.reshape((soap.shape[0], soap.shape[1] * soap.shape[2]))
        print(soap.shape)
        return soap


    reshaped = [reshape_soap(soap) for soap in soaps]

    by_l = []
    last_idx = 0
    for _ in reshaped:
        # print("last_idx", last_idx)
        by_l.append(_[:, last_idx:])
        # print("shape", by_l[-1].shape)
        last_idx = _[0].shape[0]

    pcas = [get_pca(_) for _ in by_l]
    axes = []
    if c is not None:
        for ci in range(len(c)):
            for i, (pca_output) in enumerate(zip(pcas)):
                pca, xnew = pca_output[0]
                ax = pw.Brick(figsize=(3, 2))
                plot_pca(pca, xnew, ax=ax, c=c[ci])
                ax.set_title(f"$l$={i+1}", fontsize=10)
                axes.append(ax)
    else:
        for i, (pca_output) in enumerate(zip(pcas)):
            pca, xnew = pca_output[0]
            ax = pw.Brick(figsize=(3, 2))
            plot_pca(pca, xnew, ax=ax,)
            ax.set_title(f"$l$={i + 1}", fontsize=10)
            axes.append(ax)

    return soaps, pcas, axes
