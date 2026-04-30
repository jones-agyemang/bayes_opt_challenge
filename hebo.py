from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PowerTransformer

NUMERIC_EPS = 1e-12
BOUNDS_EPS = 1e-6
DEFAULT_NOISE_FLOOR = 1e-6
DEFAULT_NOISE_SCALE = 1e-4
DEFAULT_KAPPA = 2.0
DEFAULT_XI = 0.0
DEFAULT_STOCHASTIC_MEAN_XI = 1.0
DEFAULT_WARP_LOG_BOUNDS = (-3.0, 3.0)
DEFAULT_WARP_MULTISTARTS = 4

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class OutputTransform:
    method: str
    offset: float
    transformer: PowerTransformer

    def transform(self, y: np.ndarray) -> np.ndarray:
        values = (np.asarray(y, dtype=float) + self.offset).reshape(-1, 1)
        return self.transformer.transform(values).ravel()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).reshape(-1, 1)
        restored = self.transformer.inverse_transform(values).ravel()
        return restored - self.offset


@dataclass
class KumaraswamyWarp:
    a: np.ndarray
    b: np.ndarray
    eps: float = BOUNDS_EPS

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=float), self.eps, 1.0 - self.eps)
        return 1.0 - np.power(1.0 - np.power(X, self.a), self.b)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(np.asarray(X, dtype=float), self.eps, 1.0 - self.eps)
        inner = 1.0 - np.power(1.0 - X, 1.0 / self.b)
        return np.power(np.clip(inner, self.eps, 1.0), 1.0 / self.a)


@dataclass
class HeboSurrogate:
    gp: GaussianProcessRegressor
    output_transform: OutputTransform
    input_warp: KumaraswamyWarp
    noise_variance: float
    stochastic_mean_xi: float
    y_transformed: np.ndarray
    X_original: np.ndarray
    X_warped: np.ndarray

    @property
    def y_best(self) -> float:
        return float(np.max(self.y_transformed))

    def predict_warped(self, X_warped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_warped = np.atleast_2d(np.asarray(X_warped, dtype=float))
        mu, sigma = self.gp.predict(X_warped, return_std=True)
        mu = mu + (self.stochastic_mean_xi * self.noise_variance)
        sigma = np.maximum(sigma, NUMERIC_EPS)
        return mu, sigma

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_warped(self.input_warp.transform(X))


def select_output_transform(y: np.ndarray, mode: str = "auto") -> OutputTransform:
    y = np.asarray(y, dtype=float).ravel()
    if mode != "auto":
        raise ValueError(f"Unsupported output transform mode: {mode}")

    all_positive = np.all(y > 0.0)
    all_negative = np.all(y < 0.0)

    if all_positive:
        method = "box-cox"
        offset = 0.0
    elif all_negative:
        method = "box-cox"
        offset = -float(np.min(y)) + 1.0
    else:
        method = "yeo-johnson"
        offset = 0.0

    transformer = PowerTransformer(method=method, standardize=True)
    transformer.fit((y + offset).reshape(-1, 1))
    return OutputTransform(method=method, offset=offset, transformer=transformer)


def identity_warp(n_dims: int) -> KumaraswamyWarp:
    ones = np.ones(n_dims, dtype=float)
    return KumaraswamyWarp(a=ones, b=ones)


def _coerce_length_scale(length_scale: float | list[float] | np.ndarray, n_dims: int) -> np.ndarray:
    values = np.asarray(length_scale, dtype=float)
    if values.ndim == 0:
        return np.full(n_dims, float(values), dtype=float)
    if values.shape == (n_dims,):
        return values.astype(float)
    raise ValueError(
        f"length_scale must be a scalar or have shape ({n_dims},), got {values.shape}"
    )


def build_hebo_kernel(
    n_dims: int,
    kernel_cfg: str | dict | None = None,
):
    if kernel_cfg is None:
        kernel_cfg = "linear_plus_matern32"
    if isinstance(kernel_cfg, str):
        kernel_cfg = {"type": kernel_cfg}
    if not isinstance(kernel_cfg, dict):
        raise ValueError(f"Unsupported HEBO kernel config: {kernel_cfg!r}")

    kernel_type = str(kernel_cfg.get("type", "linear_plus_matern32")).lower()
    amplitude = ConstantKernel(
        float(kernel_cfg.get("amplitude", 1.0)),
        tuple(kernel_cfg.get("amplitude_bounds", (1e-3, 1e3))),
    )
    length_scale = _coerce_length_scale(kernel_cfg.get("length_scale", 0.3), n_dims)
    length_scale_bounds = tuple(kernel_cfg.get("length_scale_bounds", (1e-2, 1e2)))
    sigma_0 = float(kernel_cfg.get("sigma_0", 1.0))
    sigma_0_bounds = tuple(kernel_cfg.get("sigma_0_bounds", (1e-5, 1e3)))

    linear = amplitude * DotProduct(sigma_0=sigma_0, sigma_0_bounds=sigma_0_bounds)

    if kernel_type == "linear":
        return linear
    if kernel_type == "rbf":
        return amplitude * RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
        )
    if kernel_type in {"matern", "matern32", "matern52"}:
        default_nu = 1.5 if kernel_type != "matern52" else 2.5
        return amplitude * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=float(kernel_cfg.get("nu", default_nu)),
        )
    if kernel_type in {"linear_plus_matern", "linear_plus_matern32", "linear+matern32"}:
        matern = amplitude * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=float(kernel_cfg.get("nu", 1.5)),
        )
        return linear + matern
    if kernel_type in {"linear_plus_matern52", "linear+matern52"}:
        matern = amplitude * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=float(kernel_cfg.get("nu", 2.5)),
        )
        return linear + matern
    raise ValueError(f"Unsupported HEBO kernel type: {kernel_type}")


def estimate_noise_alpha(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    return max(DEFAULT_NOISE_FLOOR, DEFAULT_NOISE_SCALE * float(np.var(y)))


def fit_gp_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_restarts_optimizer: int,
    random_state: int,
    kernel_cfg: str | dict | None = None,
    optimize_hyperparams: bool = True,
):
    gp = GaussianProcessRegressor(
        kernel=build_hebo_kernel(X.shape[1], kernel_cfg=kernel_cfg),
        alpha=alpha,
        normalize_y=True,
        optimizer="fmin_l_bfgs_b" if optimize_hyperparams else None,
        n_restarts_optimizer=n_restarts_optimizer if optimize_hyperparams else 0,
        random_state=random_state,
    )
    gp.fit(X, y)
    return gp


def _params_to_warp(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = params.shape[0] // 2
    a = np.exp(params[:half])
    b = np.exp(params[half:])
    return a, b


def fit_input_warp(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    random_state: int,
    kernel_cfg: str | dict | None = None,
    multistarts: int = DEFAULT_WARP_MULTISTARTS,
    maxiter: int = 30,
) -> KumaraswamyWarp:
    X = np.asarray(X, dtype=float)
    n_dims = X.shape[1]
    bounds = [DEFAULT_WARP_LOG_BOUNDS] * (2 * n_dims)
    rng = np.random.default_rng(random_state)

    def objective(params: np.ndarray) -> float:
        try:
            a, b = _params_to_warp(params)
            warp = KumaraswamyWarp(a=a, b=b)
            X_warped = warp.transform(X)
            gp = fit_gp_model(
                X_warped,
                y,
                alpha=alpha,
                n_restarts_optimizer=0,
                random_state=random_state,
                kernel_cfg=kernel_cfg,
                optimize_hyperparams=False,
            )
            return float(-gp.log_marginal_likelihood_value_)
        except Exception:
            return np.inf

    starts = [np.zeros(2 * n_dims, dtype=float)]
    for _ in range(max(0, multistarts - 1)):
        starts.append(rng.uniform(*DEFAULT_WARP_LOG_BOUNDS, size=2 * n_dims))

    best_value = np.inf
    best_params = starts[0]
    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        candidate_value = float(result.fun if np.isfinite(result.fun) else np.inf)
        if candidate_value < best_value:
            best_value = candidate_value
            best_params = np.asarray(result.x, dtype=float)

    a, b = _params_to_warp(best_params)
    return KumaraswamyWarp(a=a, b=b)


def fit_hebo_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    surrogate_cfg: dict | None = None,
    random_state: int = 0,
) -> HeboSurrogate:
    surrogate_cfg = surrogate_cfg or {}
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    output_transform = select_output_transform(
        y,
        mode=surrogate_cfg.get("output_transform", "auto"),
    )
    y_transformed = output_transform.transform(y)
    alpha = estimate_noise_alpha(y_transformed)

    if surrogate_cfg.get("input_warping", True):
        input_warp = fit_input_warp(
            X,
            y_transformed,
            alpha=alpha,
            random_state=random_state,
            kernel_cfg=surrogate_cfg.get("kernel"),
            multistarts=int(surrogate_cfg.get("warp_multistarts", DEFAULT_WARP_MULTISTARTS)),
            maxiter=int(surrogate_cfg.get("warp_maxiter", 30)),
        )
    else:
        input_warp = identity_warp(X.shape[1])

    X_warped = input_warp.transform(X)
    gp = fit_gp_model(
        X_warped,
        y_transformed,
        alpha=alpha,
        n_restarts_optimizer=int(surrogate_cfg.get("gp_restarts", 10)),
        random_state=random_state,
        kernel_cfg=surrogate_cfg.get("kernel"),
    )
    return HeboSurrogate(
        gp=gp,
        output_transform=output_transform,
        input_warp=input_warp,
        noise_variance=alpha,
        stochastic_mean_xi=float(
            surrogate_cfg.get("stochastic_mean_xi", DEFAULT_STOCHASTIC_MEAN_XI)
        ),
        y_transformed=y_transformed,
        X_original=X,
        X_warped=X_warped,
    )


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float = DEFAULT_KAPPA):
    return mu + (kappa * sigma)


def probability_of_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = DEFAULT_XI,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), NUMERIC_EPS)
    improvement = np.asarray(mu, dtype=float) - y_best - xi
    z = improvement / sigma
    pi = norm.cdf(z)
    zero_sigma = sigma <= NUMERIC_EPS
    if np.any(zero_sigma):
        pi[zero_sigma] = (improvement[zero_sigma] > 0.0).astype(float)
    return np.clip(pi, 0.0, 1.0)


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = DEFAULT_XI,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), NUMERIC_EPS)
    improvement = np.asarray(mu, dtype=float) - y_best - xi
    z = improvement / sigma
    ei = (improvement * norm.cdf(z)) + (sigma * norm.pdf(z))
    zero_sigma = sigma <= NUMERIC_EPS
    if np.any(zero_sigma):
        ei[zero_sigma] = np.maximum(improvement[zero_sigma], 0.0)
    return np.maximum(ei, 0.0)


def log_expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    xi: float = DEFAULT_XI,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), NUMERIC_EPS)
    improvement = np.asarray(mu, dtype=float) - y_best - xi
    z = improvement / sigma
    ei = expected_improvement(mu, sigma, y_best, xi)
    log_ei = np.log(np.maximum(ei, NUMERIC_EPS))

    approx_mask = z < -6.0
    if np.any(approx_mask):
        z_sq = np.square(z[approx_mask])
        approx = (
            np.log(sigma[approx_mask])
            - (0.5 * z_sq)
            - np.log(np.maximum(z_sq - 1.0, NUMERIC_EPS))
            - (0.5 * np.log(2.0 * np.pi))
        )
        log_ei[approx_mask] = approx
    return log_ei


def evaluate_mace_acquisitions(
    surrogate: HeboSurrogate,
    X_warped: np.ndarray,
    objectives: list[str] | tuple[str, ...] | None = None,
    xi: float = DEFAULT_XI,
    kappa: float = DEFAULT_KAPPA,
) -> dict[str, np.ndarray]:
    objectives = list(objectives or ["log_ei", "pi", "ucb"])
    mu, sigma = surrogate.predict_warped(X_warped)
    scores: dict[str, np.ndarray] = {}

    for objective in objectives:
        if objective == "log_ei":
            scores[objective] = log_expected_improvement(mu, sigma, surrogate.y_best, xi=xi)
        elif objective == "pi":
            scores[objective] = probability_of_improvement(mu, sigma, surrogate.y_best, xi=xi)
        elif objective == "ucb":
            scores[objective] = upper_confidence_bound(mu, sigma, kappa=kappa)
        else:
            raise ValueError(f"Unsupported MACE objective: {objective}")
    return scores


class MACEProblem(Problem):
    def __init__(self, surrogate: HeboSurrogate, proposal_cfg: dict):
        self.surrogate = surrogate
        self.proposal_cfg = proposal_cfg
        self.objectives = proposal_cfg.get("objectives", ["log_ei", "pi", "ucb"])
        super().__init__(
            n_var=surrogate.X_original.shape[1],
            n_obj=len(self.objectives),
            xl=np.full(surrogate.X_original.shape[1], BOUNDS_EPS),
            xu=np.full(surrogate.X_original.shape[1], 1.0 - BOUNDS_EPS),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        scores = evaluate_mace_acquisitions(
            surrogate=self.surrogate,
            X_warped=X,
            objectives=self.objectives,
            xi=float(self.proposal_cfg.get("xi", DEFAULT_XI)),
            kappa=float(self.proposal_cfg.get("kappa", DEFAULT_KAPPA)),
        )
        out["F"] = np.column_stack([-scores[name] for name in self.objectives])


def min_distance_to_observed(X_candidates: np.ndarray, X_observed: np.ndarray) -> np.ndarray:
    X_candidates = np.atleast_2d(np.asarray(X_candidates, dtype=float))
    X_observed = np.atleast_2d(np.asarray(X_observed, dtype=float))
    deltas = X_candidates[:, None, :] - X_observed[None, :, :]
    return np.linalg.norm(deltas, axis=2).min(axis=1)


def normalize_for_geometric_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.maximum(maxs - mins, NUMERIC_EPS)
    return np.clip((values - mins) / spans, NUMERIC_EPS, 1.0)


def select_balanced_pareto_candidate(
    X_original: np.ndarray,
    acquisitions: dict[str, np.ndarray],
    X_observed: np.ndarray,
    min_distance: float,
    dedupe_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    objective_names = list(acquisitions.keys())
    scores = np.column_stack([acquisitions[name] for name in objective_names])
    distances = min_distance_to_observed(X_original, X_observed)

    dupes = np.any(
        np.max(np.abs(X_original[:, None, :] - X_observed[None, :, :]), axis=2) <= dedupe_tol,
        axis=1,
    )
    valid_mask = (~dupes) & (distances > min_distance)
    if not np.any(valid_mask):
        valid_mask = ~dupes
    if not np.any(valid_mask):
        valid_mask = np.ones(X_original.shape[0], dtype=bool)

    candidate_scores = scores[valid_mask]
    candidate_points = X_original[valid_mask]
    candidate_distances = distances[valid_mask]

    normalized = normalize_for_geometric_mean(candidate_scores)
    geometric_scores = np.exp(np.mean(np.log(normalized), axis=1))

    best_local_index = int(np.argmax(geometric_scores))
    top_score = geometric_scores[best_local_index]
    tied_mask = np.isclose(geometric_scores, top_score)
    if np.sum(tied_mask) > 1:
        tie_indices = np.where(tied_mask)[0]
        best_tie = int(tie_indices[np.argmax(candidate_distances[tied_mask])])
        best_local_index = best_tie

    return candidate_points[best_local_index], candidate_scores[best_local_index]


def propose_next_hebo(
    X: np.ndarray,
    y: np.ndarray,
    surrogate_cfg: dict | None = None,
    proposal_cfg: dict | None = None,
    seed: int = 0,
) -> tuple[HeboSurrogate, np.ndarray]:
    surrogate_cfg = surrogate_cfg or {}
    proposal_cfg = proposal_cfg or {}
    surrogate = fit_hebo_surrogate(X, y, surrogate_cfg=surrogate_cfg, random_state=seed)

    algorithm = NSGA2(pop_size=int(proposal_cfg.get("population_size", 96)))
    problem = MACEProblem(surrogate=surrogate, proposal_cfg=proposal_cfg)
    result = pymoo_minimize(
        problem,
        algorithm,
        get_termination("n_gen", int(proposal_cfg.get("generations", 60))),
        seed=seed,
        save_history=False,
        verbose=False,
    )

    X_pareto_warped = np.atleast_2d(np.asarray(result.X, dtype=float))
    X_pareto = surrogate.input_warp.inverse_transform(X_pareto_warped)
    X_pareto = np.clip(X_pareto, BOUNDS_EPS, 1.0 - BOUNDS_EPS)

    acquisitions = evaluate_mace_acquisitions(
        surrogate=surrogate,
        X_warped=X_pareto_warped,
        objectives=proposal_cfg.get("objectives", ["log_ei", "pi", "ucb"]),
        xi=float(proposal_cfg.get("xi", DEFAULT_XI)),
        kappa=float(proposal_cfg.get("kappa", DEFAULT_KAPPA)),
    )
    X_next, _ = select_balanced_pareto_candidate(
        X_original=X_pareto,
        acquisitions=acquisitions,
        X_observed=np.asarray(X, dtype=float),
        min_distance=float(proposal_cfg.get("min_distance", 1e-3)),
        dedupe_tol=float(proposal_cfg.get("dedupe_tol", 1e-6)),
    )
    return surrogate, np.round(np.asarray(X_next, dtype=float), 6)
