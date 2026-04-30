import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.stats import norm

from pathlib import Path
import time

from utils.cycle_parameters import ( get_cycle_parameters )

import warnings
warnings.filterwarnings('ignore')

# CONSTANTS
DEFAULT_LEN_SCALE = 0.3
DEFAULT_ACQUISITION_STRATEGY = "ucb"
DEFAULT_XI = 0.01
DEFAULT_KAPPA = 2
DEFAULT_NOISE_FLOOR = 1e-6
DEFAULT_NOISE_SCALE = 1e-4

# Resolve paths relative to this file so the script works from any CWD
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

load_input = lambda x, type: np.load(f'./initial_data/function_{x}/initial_{type}.npy')

def upper_confidence_bound(mu, sigma, kappa):
    return mu + sigma * kappa

# EI for Maximisation
def expected_improvement(mu, sigma, y_best, xi):
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

def evaluate_acquisition(x, gp, acq_dict, y_best):
    if not isinstance(acq_dict, dict):
        raise TypeError("`acq_dict` must be a dict containing `strategy` and `params`.")

    # Accept a single candidate or a candidate matrix.
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mu, sigma = gp.predict(x, return_std=True)

    strategy = acq_dict.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("Missing/invalid required parameter: `strategy`.")
    strategy = strategy.lower()

    params = acq_dict.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError("`acq_dict['params']` must be a dict.")

    match strategy:
        case "ei":
            xi = params.get("xi")
            if xi is None:
                raise ValueError(
                    "Missing required parameter: `params.xi` for strategy `ei`."
                )
            return expected_improvement(mu, sigma, y_best, xi)
        case "ucb":
            kappa = params.get("kappa")
            if kappa is None:
                raise ValueError(
                    "Missing required parameter: `params.kappa` for strategy `ucb`."
                )
            return upper_confidence_bound(mu, sigma, kappa)
        case _:
            raise ValueError(f"Unsupported acquisition strategy: {strategy}")

def acq_objective(x, gp, acq_dict, y_best):
    # `minimize` requires a scalar objective.
    acquisition_value = evaluate_acquisition(x, gp, acq_dict, y_best)

    # We minimise, so return the negative acquisition
    return float(-acquisition_value[0])

def fit_gp(X, y, kernel, n_restarts_optimizer=10):
    y = np.asarray(y, dtype=float).ravel()
    y_var = float(np.var(y))
    alpha = max(DEFAULT_NOISE_FLOOR, DEFAULT_NOISE_SCALE * y_var)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True
    )
    gp.fit(X, y)
    return gp

def propose_next(
    gp, X, Y, 
    func_id, 
    acquisition_cfg):
    # set bounds
    _, n_dims = X.shape
    bounds = [(0.000001, 0.999999) for _ in range(n_dims)]

    # Optimise acquisition function (multi-start)
    best_acq = np.inf
    X_next = None

    # optimise to obtain next best random input variable
    for _ in range(10):
        x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(
            acq_objective,
            x0,
            args=(gp, acquisition_cfg, Y.max()),
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
    acquisition_cfg, 
    n_candidates = 50_000, seed=0):

    rng = np.random.default_rng(seed)
    _, n_dims = X.shape

    X_cand = rng.uniform(0., 1., size=(n_candidates, n_dims))

    y_best = Y.max()
    acq = evaluate_acquisition(X_cand, gp, acquisition_cfg, y_best)

    best_idx = int(np.argmax(acq))
    return X, Y, X_cand[best_idx]
