import argparse
from copy import deepcopy

import numpy as np

from bayesian_optimiser import build_baseline_kernel, load_dataset
from core import evaluate_acquisition, fit_gp
from hebo import (
    evaluate_mace_acquisitions,
    fit_hebo_surrogate,
    select_balanced_pareto_candidate,
)
from utils.cycle_parameters import get_cycle_parameters

NUMBER_OF_FUNCTIONS = 8


def weakly_dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left >= right) and np.any(left > right))


def apply_replay_overrides(function_cfg: dict, args) -> dict:
    cfg = deepcopy(function_cfg)
    for key in ("population_size", "generations", "kappa", "xi"):
        value = getattr(args, key, None)
        if value is not None:
            cfg.setdefault("proposal", {})[key] = value
    for key in ("warp_multistarts", "warp_maxiter", "gp_restarts", "stochastic_mean_xi"):
        value = getattr(args, key, None)
        if value is not None:
            cfg.setdefault("surrogate", {})[key] = value
    return cfg


def run_replay_sanity(cycle: int = 7, pool_size: int = 5000, seed: int = 123, cli_args=None):
    rng = np.random.default_rng(seed)
    dataset = load_dataset()
    cycle_parameters = get_cycle_parameters()

    results = []
    for func_id in range(1, NUMBER_OF_FUNCTIONS + 1):
        function_cfg = cycle_parameters[cycle][f"function_{func_id}"]
        if cli_args is not None:
            function_cfg = apply_replay_overrides(function_cfg, cli_args)
        X = dataset[func_id]["x"]
        y = dataset[func_id]["y"]
        dims = X.shape[1]
        X_pool = rng.uniform(1e-6, 1.0 - 1e-6, size=(pool_size, dims))

        baseline_cfg = function_cfg["baseline"]
        baseline_gp = fit_gp(
            X,
            y,
            build_baseline_kernel(baseline_cfg),
            n_restarts_optimizer=int(getattr(cli_args, "baseline_gp_restarts", 0) or 0),
        )
        baseline_acq = evaluate_acquisition(
            X_pool,
            baseline_gp,
            baseline_cfg["acquisition"],
            float(np.max(y)),
        )
        baseline_idx = int(np.argmax(baseline_acq))
        baseline_point = X_pool[baseline_idx]

        surrogate = fit_hebo_surrogate(
            X,
            y,
            surrogate_cfg=function_cfg["surrogate"],
            random_state=(cycle * NUMBER_OF_FUNCTIONS) + func_id,
        )
        X_pool_warped = surrogate.input_warp.transform(X_pool)
        hebo_acq = evaluate_mace_acquisitions(
            surrogate=surrogate,
            X_warped=X_pool_warped,
            objectives=function_cfg["proposal"]["objectives"],
            xi=float(function_cfg["proposal"]["xi"]),
            kappa=float(function_cfg["proposal"]["kappa"]),
        )
        hebo_point, hebo_scores = select_balanced_pareto_candidate(
            X_original=X_pool,
            acquisitions=hebo_acq,
            X_observed=X,
            min_distance=float(function_cfg["proposal"]["min_distance"]),
            dedupe_tol=float(function_cfg["proposal"]["dedupe_tol"]),
        )

        hebo_idx = int(np.argmin(np.linalg.norm(X_pool - hebo_point, axis=1)))
        baseline_scores = np.array(
            [
                hebo_acq["log_ei"][baseline_idx],
                hebo_acq["pi"][baseline_idx],
                hebo_acq["ucb"][baseline_idx],
            ],
            dtype=float,
        )

        results.append(
            {
                "function_id": func_id,
                "baseline_point": baseline_point,
                "hebo_point": hebo_point,
                "baseline_scores": baseline_scores,
                "hebo_scores": np.asarray(hebo_scores, dtype=float),
                "dominates": weakly_dominates(np.asarray(hebo_scores, dtype=float), baseline_scores),
                "pool_index": hebo_idx,
            }
        )

    dominates_count = sum(item["dominates"] for item in results)
    print(f"HEBO weakly dominates baseline on {dominates_count}/{len(results)} functions")
    for item in results:
        print(
            f"Function {item['function_id']}: "
            f"dominates={item['dominates']} "
            f"baseline_scores={np.round(item['baseline_scores'], 4)} "
            f"hebo_scores={np.round(item['hebo_scores'], 4)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=7)
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--population-size", type=int, dest="population_size")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--xi", type=float)
    parser.add_argument("--warp-multistarts", type=int, dest="warp_multistarts")
    parser.add_argument("--warp-maxiter", type=int, dest="warp_maxiter")
    parser.add_argument("--gp-restarts", type=int, dest="gp_restarts")
    parser.add_argument("--stochastic-mean-xi", type=float, dest="stochastic_mean_xi")
    parser.add_argument("--baseline-gp-restarts", type=int, default=0, dest="baseline_gp_restarts")
    args = parser.parse_args()
    run_replay_sanity(
        cycle=args.cycle,
        pool_size=args.pool_size,
        seed=args.seed,
        cli_args=args,
    )
