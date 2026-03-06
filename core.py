import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

from pathlib import Path
import time

import warnings
warnings.filterwarnings('ignore')

# CONSTANTS
DEFAULT_LEN_SCALE = 0.3

# Resolve paths relative to this file so the script works from any CWD
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

load_input = lambda x, type: np.load(f'./initial_data/function_{x}/initial_{type}.npy')

def upper_confidence_bound(mu, sigma, kappa):
    return mu + sigma * kappa

# EI for Maximisation
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """
    Expected Improvement (EI) acquisition function.

    EI = E[max(f(x) - f(x_best), 0)]

    Parameters:
    -----------
    mu : predicted mean
    sigma : predicted standard deviation
    y_best : best observed value so far
    xi : exploration parameter
    """
    with np.errstate(divide='warn'):
        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def evaluate_acquisition(x, gp, acquisition_strategy, y_best):
    # Accept a single candidate or a candidate matrix.
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)

    match acquisition_strategy:
        case "ei":
            acquisition_value = expected_improvement(mu, sigma, y_best)
        case "ucb":
            acquisition_value = upper_confidence_bound(mu, sigma, kappa=2)
        case _:
            raise ValueError(f"Unsupported acquisition strategy: {acquisition_strategy}")

    return acquisition_value


def acq_objective(x, gp, acquisition_strategy, y_best):
    # `minimize` requires a scalar objective.
    acquisition_value = evaluate_acquisition(x, gp, acquisition_strategy, y_best)

    # We minimise, so return the negative acquisition
    return float(-acquisition_value[0])

def fit_gp(X, y):
    kernel = ConstantKernel(1.0) * RBF(length_scale=DEFAULT_LEN_SCALE)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
    gp.fit(X, y)
    return gp

def propose_next(
    gp, X, Y, 
    func_id, 
    acquisition='ucb'):
    # set bounds
    _, n_dims = X.shape
    bounds = [(0.000001, 0.999999) for _ in range(n_dims)]

    # Normalise acquisition string once so we don't reference an undefined name
    acquisition_strategy = acquisition.lower()

    # Optimise acquisition function (multi-start)
    best_acq = np.inf
    X_next = None

    # optimise to obtain next best random input variable
    for _ in range(10):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(
            acq_objective,
            x0,
            args=(gp, acquisition_strategy, Y.max()),
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result.fun < best_acq:
            best_acq = result.fun
            X_next = np.round(result.x, 6)

    return X, Y, X_next

def propose_next_rnd_sampling(
    gp, X, Y,
    func_id,
    n_candidates = 50_000, acquisition='ucb', seed=0):
    rng = np.random.default_rng(seed)
    _, n_dims = X.shape

    X_cand = rng.uniform(0., 1., size=(n_candidates, n_dims))

    y_best = Y.max()
    acq = evaluate_acquisition(X_cand, gp, acquisition.lower(), y_best)

    best_idx = int(np.argmax(acq))
    return X, Y, X_cand[best_idx]

