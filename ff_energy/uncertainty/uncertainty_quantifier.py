import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ff_energy import calculate_bootstrap_uncertainty
from ff_energy import calculate_cross_val_uncertainty, \
    calculate_std_uncertainty
from ff_energy import calculate_mapie_uncertainty
from ff_energy import calculate_bayesian_uncertainty
from ff_energy import calculate_conformal_uncertainty


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

        self.scaler = None

        self.output_dict = {
            'conformal_uncertainty': None,
            'bayesian_uncertainty': None,
            'bootstrap_uncertainty': None,
            'cross_val_uncertainty': None,
            'mapie_uncertainty': None,
            'std_uncertainty': None,
        }

    def calc(self):
        self.calculate_uncertainty()

    def __repr__(self):
        return f"UncertaintyQuantifier(data={self.data})"

    def __str__(self):
        return f"UncertaintyQuantifier(data={self.data})"

    def calculate_uncertainty(self, split=0.8, keys=None):

        self.scaler = StandardScaler()

        self.data["TARGET"] = self.data[self.ref_key]
        self.data["FIT"] = self.data[self.key]
        self.data["SE"] = self.data["FIT"].std() / np.sqrt(self.data["FIT"].shape[0])
        # standardize the data

        self.data["FIT_scaled"] = self.scaler.fit_transform(
            self.data[self.key].values.reshape(-1, 1)).T[0]

        self.data["TARGET_scaled"] = self.scaler.transform(
            self.data["TARGET"].values.reshape(-1, 1)).T[0]

        self.data["SE_scaled"]\
            = self.data["FIT_scaled"].std() / np.sqrt(self.data["FIT_scaled"].shape[0])

        print(self.data.describe())
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
            "test_std": np.std(residuals[test_idx]) * np.ones(len(test_idx)),
            "train_std": np.std(residuals[train_idx]) * np.ones(len(train_idx)),
            "y_train": self.data.loc[train_idx, self.key],
            "y_test": self.data.loc[test_idx, self.key],
            "y_pred": self.data.loc[train_idx, self.key],
            "y_test_pred": self.data.loc[test_idx, self.key],
            "X_train": self.data.loc[train_idx, self.ref_key],
            "X_test": self.data.loc[test_idx, self.ref_key],
        }

        if keys is None:
            keys = self.output_dict.keys()

        if "conformal_uncertainty" in keys:
            # Calculate the uncertainty using conformal predictions
            conformal_uncertainty = calculate_conformal_uncertainty(data_dict)
            self.conformal_uncertainty = conformal_uncertainty
            self.output_dict['conformal_uncertainty'] = conformal_uncertainty
        if "bayesian_uncertainty" in keys:
            # Calculate the uncertainty using Bayesian statistics
            bayesian_uncertainty = calculate_bayesian_uncertainty(self.data,
                                                                  scaler=self.scaler)
            self.bayesian_uncertainty = bayesian_uncertainty
            self.output_dict['bayesian_uncertainty'] = bayesian_uncertainty
        if "bootstrap_uncertainty" in keys:
            # Calculate the uncertainty using bootstrap resampling
            bootstrap_uncertainty = calculate_bootstrap_uncertainty(self.data)
            self.bootstrap_uncertainty = bootstrap_uncertainty
            self.output_dict['bootstrap_uncertainty'] = bootstrap_uncertainty
        if "cross_val_uncertainty" in keys:
            # Calculate the uncertainty using cross-validation
            cross_val_uncertainty = calculate_cross_val_uncertainty(data_dict)
            self.cross_val_uncertainty = cross_val_uncertainty
            self.output_dict['cross_val_uncertainty'] = cross_val_uncertainty
        if "mapie_uncertainty" in keys:
            # Calculate the uncertainty using MAPPIE
            mapie_uncertainty = calculate_mapie_uncertainty(data_dict)
            self.mapie_uncertainty = mapie_uncertainty
            self.output_dict['mapie_uncertainty'] = mapie_uncertainty
        if "std_uncertainty" in keys:
            # standard dev. uncertainty
            std_uncertainty = calculate_std_uncertainty(data_dict)
            self.std_uncertainty = std_uncertainty
            self.output_dict['std_uncertainty'] = std_uncertainty
