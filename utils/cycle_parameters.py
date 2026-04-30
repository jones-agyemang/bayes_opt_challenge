PROPOSE_NEXT = "propose_next"
PROPOSE_NEXT_RND = "propose_next_rnd_sampling"

DEFAULT_LEN_SCALE = 0.3
DEFAULT_LEN_SCALE_BOUND = (1e-2, 1e2)
DEFAULT_KERNEL_SMOOTHNESS_CTRL = 2.5

DEFAULT_HEBO_SURROGATE = {
    "output_transform": "auto",
    "input_warping": True,
    "kernel": "linear_plus_matern32",
    "stochastic_mean_xi": 1.0,
}

DEFAULT_HEBO_PROPOSAL = {
    "mode": "hebo",
    "optimizer": "mace_nsga2",
    "objectives": ["log_ei", "pi", "ucb"],
    "population_size": 96,
    "generations": 60,
    "min_distance": 1e-3,
    "dedupe_tol": 1e-6,
    "selection": "balanced_pareto",
    "xi": 0.0,
    "kappa": 2.0,
}

DEFAULT_BASELINE = {
    "proposer": PROPOSE_NEXT,
    "acquisition": {
        "strategy": "ucb",
        "params": {"kappa": 2.0},
    },
    "kernel": {
        "type": "Matern",
        "length_scale": DEFAULT_LEN_SCALE,
        "length_scale_bounds": DEFAULT_LEN_SCALE_BOUND,
        "nu": DEFAULT_KERNEL_SMOOTHNESS_CTRL,
    },
}


def build_default_config():
    return {
        "surrogate": dict(DEFAULT_HEBO_SURROGATE),
        "proposal": {
            **DEFAULT_HEBO_PROPOSAL,
            "objectives": list(DEFAULT_HEBO_PROPOSAL["objectives"]),
        },
        "baseline": {
            "proposer": DEFAULT_BASELINE["proposer"],
            "acquisition": {
                "strategy": DEFAULT_BASELINE["acquisition"]["strategy"],
                "params": dict(DEFAULT_BASELINE["acquisition"]["params"]),
            },
            "kernel": dict(DEFAULT_BASELINE["kernel"]),
        },
    }


def build_hebo_override(
    *,
    surrogate_updates: dict | None = None,
    proposal_updates: dict | None = None,
    baseline_updates: dict | None = None,
):
    cfg = build_default_config()
    if surrogate_updates:
        cfg["surrogate"].update(surrogate_updates)
    if proposal_updates:
        cfg["proposal"].update(proposal_updates)
    if baseline_updates:
        cfg["baseline"].update(baseline_updates)
    return cfg


CYCLE_FUNCTION_OVERRIDES = {
    8: {
        1: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.35},
                "stochastic_mean_xi": 0.75,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 128,
                "generations": 80,
                "kappa": 2.0,
                "xi": 0.0,
            },
        ),
        2: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern32", "length_scale": 0.2},
                "stochastic_mean_xi": 1.5,
                "warp_multistarts": 6,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 90,
                "kappa": 3.0,
                "xi": 0.01,
                "min_distance": 5e-3,
            },
        ),
        3: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern32", "length_scale": 0.15},
                "stochastic_mean_xi": 1.5,
                "warp_multistarts": 8,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 100,
                "kappa": 3.0,
                "xi": 0.01,
                "min_distance": 5e-3,
            },
        ),
        4: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern52", "length_scale": 0.5},
                "stochastic_mean_xi": 0.5,
                "warp_multistarts": 4,
                "warp_maxiter": 30,
                "gp_restarts": 10,
            },
            proposal_updates={
                "population_size": 96,
                "generations": 70,
                "kappa": 1.5,
                "xi": 0.0,
            },
        ),
        5: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.6},
                "stochastic_mean_xi": 0.75,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 128,
                "generations": 80,
                "kappa": 1.75,
                "xi": 0.0,
            },
        ),
        6: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern32", "length_scale": 0.12},
                "stochastic_mean_xi": 1.25,
                "warp_multistarts": 6,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 90,
                "kappa": 2.5,
                "xi": 0.01,
                "min_distance": 5e-3,
            },
        ),
        7: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern32", "length_scale": 0.25},
                "stochastic_mean_xi": 1.25,
                "warp_multistarts": 8,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 192,
                "generations": 90,
                "kappa": 3.0,
                "xi": 0.02,
                "min_distance": 1e-2,
            },
        ),
        8: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.4},
                "stochastic_mean_xi": 1.0,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 128,
                "generations": 100,
                "kappa": 2.0,
                "xi": 0.01,
                "min_distance": 5e-3,
            },
        ),
    },
    9: {
        1: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.35},
                "stochastic_mean_xi": 0.75,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 128,
                "generations": 80,
                "kappa": 2.5,
                "xi": 0.0,
            },
        ),
        2: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern32", "length_scale": 0.2},
                "stochastic_mean_xi": 1.5,
                "warp_multistarts": 6,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 90,
                "kappa": 5.0,
                "xi": 0.01,
                "min_distance": 5e-3,
            },
        ),
        3: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern32", "length_scale": 0.15},
                "stochastic_mean_xi": 1.5,
                "warp_multistarts": 8,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 100,
                "kappa": 3.0,
                "xi": 0.05,
                "min_distance": 5e-3,
            },
        ),
        4: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern52", "length_scale": 0.5},
                "stochastic_mean_xi": 0.5,
                "warp_multistarts": 4,
                "warp_maxiter": 30,
                "gp_restarts": 10,
            },
            proposal_updates={
                "population_size": 96,
                "generations": 70,
                "kappa": 2.5,
                "xi": 0.0,
            },
        ),
        5: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.6},
                "stochastic_mean_xi": 0.75,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 128,
                "generations": 80,
                "kappa": 1.75,
                "xi": 0.1,
            },
        ),
        6: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "matern32", "length_scale": 0.12},
                "stochastic_mean_xi": 1.25,
                "warp_multistarts": 6,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 160,
                "generations": 90,
                "kappa": 3.0,
                "xi": 0.09,
                "min_distance": 5e-3,
            },
        ),
        7: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern32", "length_scale": 0.25},
                "stochastic_mean_xi": 1.25,
                "warp_multistarts": 8,
                "warp_maxiter": 50,
                "gp_restarts": 15,
            },
            proposal_updates={
                "population_size": 300,
                "generations": 180,
                "kappa": 5.0,
                "xi": 0.7,
                "min_distance": 1e-2,
            },
        ),
        8: build_hebo_override(
            surrogate_updates={
                "kernel": {"type": "linear_plus_matern52", "length_scale": 0.4},
                "stochastic_mean_xi": 1.0,
                "warp_multistarts": 6,
                "warp_maxiter": 40,
                "gp_restarts": 12,
            },
            proposal_updates={
                "population_size": 256,
                "generations": 100,
                "kappa": 2.5,
                "xi": 0.03,
                "min_distance": 5e-3,
            },
        ),
    }
}

def build_function_config(cycle, func_id):
    cycle_function_override = CYCLE_FUNCTION_OVERRIDES.get(cycle, {}).get(func_id, {})

    default_config = build_default_config()

    return cycle_function_override if bool(cycle_function_override) else default_config


def get_cycle_parameters(cycle_budget: int = 12):
    return {
        cycle: {
            f"function_{func_id}": build_function_config(cycle, func_id)
            for func_id in range(1, 9)
        }
        for cycle in range(1, cycle_budget + 1)
    }
