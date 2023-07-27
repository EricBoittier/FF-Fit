import os
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import Predictive

plt.rcParams['axes.facecolor'] = 'white'

plt.style.use("bmh")
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats("svg")

assert numpyro.__version__.startswith("0.11.0")


def get_random_seed():
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    return rng_key_


def regression_model(fit, targets, targets_sd):
    #  std
    sigma = numpyro.sample("sigma",
                           dist.Exponential(1.0))

    #  sample the scale factor
    a = numpyro.sample("a", dist.Normal(0.0, 0.5))

    #  computer the mean
    mu = fit * a
    #  assume a distribution for the fit
    fit_dist = numpyro.sample("fit_dist",
                              dist.Normal(mu, sigma))
    #  compare to targets
    numpyro.sample("obs",
                   dist.Normal(fit_dist, targets_sd),
                   obs=targets)


def get_kernel(regression_model):
    kernel = NUTS(regression_model, target_accept_prob=0.9)
    return kernel


def get_mcmc(kernel, num_warmup=1000, num_samples=1000):
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    return mcmc


def run_mcmc(mcmc, rng_key_, data, target_sd=None):
    """
    Expects a dataframe with columns FIT_Scaled and TARGET_Scaled
    :param mcmc:
    :param rng_key_:
    :param data:
    :param target_sd:
    :return:
    """
    if target_sd is not None:
        target_sd = jnp.array(target_sd)
    else:
        target_sd = jnp.ones(data["TARGET_scaled"].shape[0])

    mcmc.run(
        rng_key_,
        fit=data["FIT_scaled"].values,
        targets=data["TARGET_scaled"].values,
        targets_sd=target_sd,
    )

    return mcmc


def get_predictions(mcmc, rng_key_, data, target_sd=None):
    if target_sd is not None:
        target_sd = jnp.array(target_sd)
    else:
        target_sd = jnp.ones(data["TARGET_scaled"].shape[0])

    predictive = Predictive(mcmc.sampler.model, mcmc.get_samples())
    predictions = predictive(rng_key_,
                             fit=data["FIT_scaled"].values,
                             targets=data["TARGET_scaled"].values,
                             targets_sd=target_sd,
                             )
    return predictions


def get_mean_residuals(predictions):
    return predictions["obs"].mean(axis=0)


def get_residuals(predictions, CI=0.9):
    residuals_dict = {}
    residuals_dict["obs"] = predictions["obs"]
    residuals_dict["obs_mean"] = predictions["obs"].mean(axis=0)
    residuals_dict["obs_hpdi"] = hpdi(predictions["obs"], CI)
    residuals_dict["obs_mean_hpdi"] = hpdi(
        predictions["obs"].mean(axis=0), CI)
    err = abs(residuals_dict["obs_hpdi"][0] - residuals_dict["obs_mean_hpdi"][0])
    residuals_dict["err"] = err
    return residuals_dict


def calculate_bayesian_uncertainty(df, verbose=False, scaler=None):
    kernel = get_kernel(regression_model)
    mcmc = get_mcmc(kernel)
    rnd_key = get_random_seed()
    mcmc = run_mcmc(mcmc, rnd_key, df, target_sd=df["FIT_scaled"].std())
    if verbose:
        mcmc.print_summary()
    predictions = get_predictions(mcmc, rnd_key, df)
    residuals = get_residuals(predictions)
    if scaler is None:
        unstandardize = lambda x: x * df["FIT"].std() + df["FIT"].mean()
    else:
        #  Assume the scaler is a MinMaxScaler and reshape accordingly
        def unstandardize(x):
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            return scaler.inverse_transform(x)

    out_dict = {
        "predictions_scaled": predictions,
        "residuals_scaled": residuals,
        "df": df,
        "unstandarize": unstandardize,
        "residuals_unscaled": {
            "obs": unstandardize(residuals["obs"]),
            "obs_mean": unstandardize(residuals["obs_mean"]),
            "obs_hpdi": unstandardize(residuals["obs_hpdi"]),
            "obs_mean_hpdi": unstandardize(residuals["obs_mean_hpdi"]),
            "err": unstandardize(residuals["err"]),
        },
    }

    return out_dict
