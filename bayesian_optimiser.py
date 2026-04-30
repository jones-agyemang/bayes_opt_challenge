import argparse
import time
import warnings
from copy import deepcopy

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

from core import fit_gp, propose_next, propose_next_rnd_sampling
from hebo import propose_next_hebo
from kernel_config import KernelConfig
from utils.cycle_parameters import (
    PROPOSE_NEXT,
    PROPOSE_NEXT_RND,
    get_cycle_parameters,
)
from utils.loader import load_input_data, load_output_data

warnings.filterwarnings("ignore")

DEFAULT_LEN_SCALE_BOUND = (1e-3, 1e3)
NUMBER_OF_FUNCTIONS = 8
DEFAULT_CYCLE = 8


def load_dataset():
    return {
        i: {"x": load_input_data(i), "y": load_output_data(i)}
        for i in range(1, NUMBER_OF_FUNCTIONS + 1)
    }


def build_baseline_kernel(baseline_cfg: dict):
    kernel_cfg = baseline_cfg.get("kernel") or {}
    kernel_type = kernel_cfg.get("type", "Matern")
    kernel = KernelConfig(
        kernel_type=kernel_type,
        length_scale=kernel_cfg.get("length_scale", 0.3),
        length_scale_bounds=kernel_cfg.get("length_scale_bounds", (1e-2, 1e2)),
        nu=kernel_cfg.get("nu", 2.5 if kernel_type == "Matern" else None),
    )
    amplitude = ConstantKernel(1.0, DEFAULT_LEN_SCALE_BOUND)

    if kernel.kernel_type == "Matern":
        return amplitude * Matern(
            kernel.length_scale,
            kernel.length_scale_bounds,
            kernel.nu,
        )
    if kernel.kernel_type == "RBF":
        return amplitude * RBF(
            kernel.length_scale,
            kernel.length_scale_bounds,
        )
    raise ValueError(f"Unsupported baseline kernel type: {kernel.kernel_type}")


def propose_baseline_candidate(
    X: np.ndarray,
    y: np.ndarray,
    baseline_cfg: dict,
    seed: int,
):
    gp = fit_gp(X, y, build_baseline_kernel(baseline_cfg))
    proposer = baseline_cfg.get("proposer", PROPOSE_NEXT)
    acquisition_cfg = baseline_cfg.get("acquisition", {})

    if proposer == PROPOSE_NEXT:
        _, _, candidate = propose_next(gp, X, y, None, acquisition_cfg)
        return candidate
    if proposer == PROPOSE_NEXT_RND:
        _, _, candidate = propose_next_rnd_sampling(
            gp,
            X,
            y,
            None,
            acquisition_cfg,
            seed=seed,
        )
        return np.round(candidate, 6)
    raise ValueError(f"Unsupported baseline proposer: {proposer}")


def propose_candidate(
    func_id: int,
    X: np.ndarray,
    y: np.ndarray,
    function_cfg: dict,
    cycle: int,
):
    proposal_cfg = function_cfg.get("proposal", {})
    mode = str(proposal_cfg.get("mode", "hebo")).lower()
    seed = (cycle * NUMBER_OF_FUNCTIONS) + func_id

    if mode == "hebo":
        _, candidate = propose_next_hebo(
            X,
            y,
            surrogate_cfg=function_cfg.get("surrogate", {}),
            proposal_cfg=proposal_cfg,
            seed=seed,
        )
        return candidate
    if mode == "baseline":
        return propose_baseline_candidate(
            X=X,
            y=y,
            baseline_cfg=function_cfg.get("baseline", {}),
            seed=seed,
        )
    raise ValueError(f"Unsupported proposal mode: {mode}")


def apply_cli_overrides(function_cfg: dict, args) -> dict:
    cfg = deepcopy(function_cfg)
    if args.mode is not None:
        cfg.setdefault("proposal", {})["mode"] = args.mode
    for key in ("population_size", "generations", "kappa", "xi"):
        value = getattr(args, key, None)
        if value is not None:
            cfg.setdefault("proposal", {})[key] = value
    for key in ("warp_multistarts", "warp_maxiter", "gp_restarts", "stochastic_mean_xi"):
        value = getattr(args, key, None)
        if value is not None:
            cfg.setdefault("surrogate", {})[key] = value
    return cfg


def bayesian_loop(
    cycle: int = DEFAULT_CYCLE,
    plot_evaluations: bool = False,
    cli_args=None,
):
    dataset = load_dataset()
    cycle_parameters = get_cycle_parameters()
    optimal_cycle_values = []

    for func_id, data in dataset.items():
        print()
        print(f"Function: {func_id}")
        X = data["x"]
        y = data["y"]
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")

        if plot_evaluations:
            from log_eval_plts import signed_log_plot
            signed_log_plot(func_id, y)

        function_cfg = cycle_parameters.get(cycle, {}).get(f"function_{func_id}", {})
        
        print(f'Function {func_id} config')
        print(function_cfg)
        print("-"*20)
        if cli_args is not None:
            function_cfg = apply_cli_overrides(function_cfg, cli_args)
        proposal_mode = function_cfg.get("proposal", {}).get("mode", "hebo")
        print(f"Proposal mode: {proposal_mode}")

        optimal_cycle_values.append(
            propose_candidate(func_id=func_id, X=X, y=y, function_cfg=function_cfg, cycle=cycle)
        )
    return optimal_cycle_values


def print_cycle_values(optimal_cycle_values):
    for idx, arr in enumerate(optimal_cycle_values):
        arr = np.asarray(arr, dtype=float)
        formatted = [f"{value:.6f}" for value in arr]
        print(f"Function {idx + 1}: " + "-".join(formatted))


def main(cycle: int = DEFAULT_CYCLE, plot_evaluations: bool = False, cli_args=None):
    print("Running Bayesian Optimisation...")
    start = time.time()
    print("--------")

    optimal_cycle_values = bayesian_loop(
        cycle=cycle,
        plot_evaluations=plot_evaluations,
        cli_args=cli_args,
    )
    print_cycle_values(optimal_cycle_values)

    print("--------")
    print(f"Time elapsed: {time.time() - start}secs")
    print("Challenge completed!")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=DEFAULT_CYCLE)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--mode", choices=["hebo", "baseline"])
    parser.add_argument("--population-size", type=int, dest="population_size")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--xi", type=float)
    parser.add_argument("--warp-multistarts", type=int, dest="warp_multistarts")
    parser.add_argument("--warp-maxiter", type=int, dest="warp_maxiter")
    parser.add_argument("--gp-restarts", type=int, dest="gp_restarts")
    parser.add_argument("--stochastic-mean-xi", type=float, dest="stochastic_mean_xi")
    args = parser.parse_args()
    main(cycle=args.cycle, plot_evaluations=args.plot, cli_args=args)
