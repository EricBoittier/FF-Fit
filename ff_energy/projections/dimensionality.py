from sklearn.decomposition import PCA
from sklearn import random_projection
import numpy as np


def get_pca(data, n_components=2) -> (PCA, np.ndarray):
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(data)
    return pca, X_new


def get_random_projection(
    data, n_components=None, type="gaussian"
) -> (random_projection, np.ndarray):
    if n_components is None:
        n_components = "auto"

    if type == "gaussian":
        rp = random_projection.GaussianRandomProjection(n_components=n_components)
        X_new = rp.fit_transform(data)

    elif type == "sparse":
        rp = random_projection.SparseRandomProjection(n_components=n_components)
        X_new = rp.fit_transform(data)

    return rp, X_new


def get_rp_pca(data) -> (random_projection, PCA, np.ndarray):
    #  project to slightly lower dimensions
    rp = random_projection.SparseRandomProjection(eps=0.9, random_state=None)
    X_new1 = rp.fit_transform(data)
    # project back to 2D
    pca = PCA(n_components=2)
    X_new2 = pca.fit_transform(X_new1)
    return rp, pca, X_new1, X_new2
