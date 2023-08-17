import numpy as np

from fortuna.conformal import OneDimensionalUncertaintyConformalRegressor


def calculate_conformal_uncertainty(data_dict, verbose=False, error=0.05):
    """

    :param data_dict:
    :return:
    """
    test_preds = np.stack([data_dict["X_test"]]).T
    test_uncertainties = np.stack([data_dict["test_std"]]).T

    val_preds = np.stack([data_dict["y_train"]]).T
    val_uncertainties = np.stack([data_dict["train_std"]]).T
    val_targets = np.stack([data_dict["X_train"]]).T

    if verbose:
        for k, v in data_dict.items():
            print(k, v.shape)
        print("val_targets", val_targets.shape)
        print(val_targets)
        print("val_uncertainties", val_uncertainties.shape)
        print(val_uncertainties)
        print("val_preds", val_preds.shape)
        print(val_preds)
        print("test_uncertainties", test_uncertainties.shape)
        print(test_uncertainties)
        print("test_preds", test_preds.shape)
        print(test_preds)

    conformal_intervals = (
        OneDimensionalUncertaintyConformalRegressor().conformal_interval(
            error=error,
            val_preds=val_preds,
            val_uncertainties=val_uncertainties,
            test_preds=test_preds,
            test_uncertainties=test_uncertainties,
            val_targets=val_targets,
        )
    )
    return conformal_intervals
