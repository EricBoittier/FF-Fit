import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error

import scipy.stats
from sklearn.linear_model import LinearRegression


def calculate_std_uncertainty(data_dict):
    # Placeholder for the actual implementation
    return np.std(data_dict["residuals"])


def calculate_cross_val_uncertainty(data_dict):
    X = data_dict["X"]
    y = data_dict["y"]
    # Placeholder for the actual implementation
    est = LinearRegression()
    scores = cross_val_score(est, X, y, cv=10, scoring=make_scorer(mean_absolute_error))
    sem = scipy.stats.sem(scores)
    print(
        f"Mean absolute error with 95% CI: " f"{np.mean(scores):.3f} ± {sem * 1.96:.3f}"
    )
    return_dict = {"mean": np.mean(scores), "sem": sem, "ci": sem * 1.96}
    return return_dict
