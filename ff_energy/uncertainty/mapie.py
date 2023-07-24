import numpy as np
from sklearn.metrics import mean_absolute_error
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def calculate_mapie_uncertainty(data_dict, verbose=False, error=0.05):
    y_train = np.array(data_dict['y_train'])
    y_test = np.array(data_dict['y_test'])
    X_train = np.array(data_dict['X_train'])
    X_test = np.array(data_dict['X_test'])
    y_pred = np.array(data_dict['y_pred'])
    y_test_pred = np.array(data_dict['y_test_pred'])
    std = np.array([data_dict['std']])
    y_err = np.vstack([std, std]) * 1.96

    # print shapes if in verbose
    if verbose:
        print("y_train", y_train.shape)
        print("y_test", y_test.shape)
        print("X_train", X_train.shape)
        print("X_test", X_test.shape)
        print("y_pred", y_pred.shape)
        print("y_test_pred", y_test_pred.shape)
        print("std", std.shape)
        print("y_err", y_err.shape)


    # Print out statistics
    mae_test = mean_absolute_error(y_test, y_test_pred)
    print(f"MAPIE: MAE = {mae_test:.6f}")
    print(f"MAPIE: Width of 95% prediction interval = "
          f"{np.mean(y_err) * 2:6f}")
    coverage = regression_coverage_score(
        y_test, y_test_pred - std * 1.96, y_test_pred + std * 1.96
    )
    print(f"MAPIE: Coverage of 95% prediction interval: {coverage:.6f}")

    est = LinearRegression()
    mapie = MapieRegressor(est, cv=10, agg_function="median")

    mapie.fit(X_train, y_train)

    y_test_pred, y_test_pis = mapie.predict(
        X_test, alpha=[error]
    )
    return_dict = {
        'mae_test': mae_test,
        'width': np.mean(y_err) * 2,
        'coverage': coverage,
        'y_test_pred': y_test_pred,
        'y_test_pis': y_test_pis
    }
    return return_dict
