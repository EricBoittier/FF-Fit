import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

import scipy.stats

from sklearn.linear_model import LinearRegression

from ff_energy.uncertainty.pipe import standardize
from ff_energy.uncertainty.bayes import calculate_bayesian_uncertainty


def calculate_conformal_uncertainty(data_dict):
    # Placeholder for the actual implementation
    return np.std(data_dict['residuals'])


def calculate_std_uncertainty(data_dict):
    # Placeholder for the actual implementation
    return np.std(data_dict['residuals'])


def calculate_bootstrap_uncertainty(df):
    # Placeholder for the actual implementation
    from scipy.stats import bootstrap

    rng = np.random.default_rng()
    data = (df["SE"],)  # samples must be in a sequence
    res = bootstrap(
        data, np.mean, confidence_level=0.95, random_state=rng, n_resamples=100
    )
    seMin, seMax = res.confidence_interval
    rseMin, rseMax = np.sqrt(seMin), np.sqrt(seMax)
    out_dict = {
        "seMin": seMin,
        "seMax": seMax,
        "rseMin": rseMin,
        "rseMax": rseMax,
    }
    return out_dict


def calculate_mapie_uncertainty(data_dict):
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_pred = data_dict['y_pred']
    y_test_pred = data_dict['y_test_pred']
    std = data_dict['std']
    y_err = np.vstack([std, std]) * 1.96
    # Print out statistics
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"MAPIE: MAE = {mae_test:.6f}")
    print(f"MAPIE: Width of 95% prediction interval = {np.mean(y_err) * 2:6f}")
    coverage = regression_coverage_score(
        y_test, y_test_pred - std * 1.96, y_test_pred + std * 1.96
    )
    print(f"MAPIE: Coverage of 95% prediction interval: {coverage:.6f}")
    est = RandomForestRegressor(n_estimators=10, random_state=42)
    mapie = MapieRegressor(est, cv=10, agg_function="median")
    mapie.fit(X_train, y_train)
    y_test_pred, y_test_pis = mapie.predict(X_test, alpha=[0.05])

    return_dict = {
        'mae_test': mae_test,
        'width': np.mean(y_err) * 2,
        'coverage': coverage,
        'y_test_pred': y_test_pred,
        'y_test_pis': y_test_pis
    }
    return return_dict


def calculate_cross_val_uncertainty(data_dict):
    X = data_dict['X']
    y = data_dict['y']
    # Placeholder for the actual implementation
    est = LinearRegression()
    scores = cross_val_score(est, X, y, cv=10,
                             scoring=make_scorer(mean_absolute_error))
    sem = scipy.stats.sem(scores)
    print(f"Mean absolute error with 95% CI: "
          f"{np.mean(scores):.3f} ± {sem * 1.96:.3f}")
    return_dict = {
        'mean': np.mean(scores),
        'sem': sem,
        'ci': sem * 1.96
    }
    return return_dict


class UncertaintyQuantifier:
    def __init__(self, data: pd.DataFrame, ref_key=None, key=None):
        self.cross_val_uncertainty = None
        self.std_uncertainty = None
        self.mapie_uncertainty = None
        self.bootstrap_uncertainty = None
        self.bayesian_uncertainty = None
        self.conformal_uncertainty = None

        self.data = data
        if key is None:
            self.key = "Ens"
        else:
            self.key = key
        if ref_key is None:
            self.ref_key = "Eref"
        else:
            self.ref_key = ref_key

        self.output_dict = self.calculate_uncertainty()

    def __repr__(self):
        return f"UncertaintyQuantifier(data={self.data})"

    def __str__(self):
        return f"UncertaintyQuantifier(data={self.data})"

    def calculate_uncertainty(self, split=0.8):
        self.data["TARGET"] = self.data[self.ref_key]
        self.data["FIT"] = self.data[self.key]
        self.data["SE"] = self.data["FIT"].std() / np.sqrt(self.data["FIT"].shape[0])
        # standardize the data
        self.data["FIT_scaled"] = self.data["FIT"].pipe(standardize)
        self.data["TARGET_scaled"] = self.data["TARGET"].pipe(standardize)
        # Calculate the residuals
        residuals = self.data[self.ref_key] - self.data[self.key]
        #  create a test-train split
        train_idx = np.random.choice(
            self.data.index, size=int(len(self.data) * split),
            replace=False
        )
        test_idx = self.data.index[~self.data.index.isin(train_idx)]

        data_dict = {
            'X': self.data.drop(columns=[self.ref_key]),
            'y': self.data[self.key],
            'residuals': residuals,
            "std": np.std(residuals),
            "y_train": self.data.loc[train_idx, self.key],
            "y_test": self.data.loc[test_idx, self.key],
            "y_pred": self.data.loc[train_idx, self.key],
            "y_test_pred": self.data.loc[test_idx, self.key],
            "X_train": self.data.loc[train_idx, self.data.columns != self.key],
            "X_test": self.data.loc[test_idx, self.data.columns != self.key],
        }

        # Calculate the uncertainty using conformal predictions
        conformal_uncertainty = calculate_conformal_uncertainty(data_dict)
        self.conformal_uncertainty = conformal_uncertainty

        # Calculate the uncertainty using Bayesian statistics
        bayesian_uncertainty = calculate_bayesian_uncertainty(self.data)
        self.bayesian_uncertainty = bayesian_uncertainty

        # Calculate the uncertainty using bootstrap resampling
        bootstrap_uncertainty = calculate_bootstrap_uncertainty(data_dict)
        self.bootstrap_uncertainty = bootstrap_uncertainty

        # Calculate the uncertainty using cross-validation
        cross_val_uncertainty = calculate_cross_val_uncertainty(data_dict)
        self.cross_val_uncertainty = cross_val_uncertainty

        # Calculate the uncertainty using MAPPIE
        mapie_uncertainty = calculate_mapie_uncertainty(data_dict)
        self.mapie_uncertainty = mapie_uncertainty

        # standard dev. uncertainty
        std_uncertainty = calculate_std_uncertainty(data_dict)
        self.std_uncertainty = std_uncertainty

        out_dict = {
            'conformal_uncertainty': conformal_uncertainty,
            'bayesian_uncertainty': bayesian_uncertainty,
            'bootstrap_uncertainty': bootstrap_uncertainty,
            'cross_val_uncertainty': cross_val_uncertainty,
            'mapie_uncertainty': mapie_uncertainty,
            'std_uncertainty': std_uncertainty,
        }

        return out_dict
