import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import time

import warnings
warnings.filterwarnings('ignore')

from utils.loader import (
    load_input_data,
    load_output_data
)

from core import (
    fit_gp,
    propose_next,
    propose_next_rnd_sampling
)

from log_eval_plts import (signed_log_plot)

# CONSTANTS
DEFAULT_LEN_SCALE = 0.3
PROPOSE_NEXT     = "propose_next"
PROPOSE_NEXT_RND = "propose_next_rnd_sampling"

"""
Run Bayesian Optimisation Challenge 
"""
print("Running Bayesian Optimisation...")
start = time.time()
print('--------')

# Settings
NUMBER_OF_FUNCTIONS = 8 # 8
CYCLE_BUDGET = 12

dataset = {
    i: {"x": load_input_data(i), "y": load_output_data(i)}
    for i in range(1, NUMBER_OF_FUNCTIONS + 1)
}

cycle_parameters = {
    1: {
        "acquisition": { 
            "strategy": "ucb",
            "params": { "kappa": 2.0 }
        },
        "proposer": PROPOSE_NEXT,
    },
    2: {
        "acquisition": { 
            "strategy": "ucb",
            "params": { "kappa": 20.0 }
        },
        "proposer": PROPOSE_NEXT,
    },
    3: {
        "acquisition": {
            "strategy": "ei",
            "params": {}
        },
        "proposer": PROPOSE_NEXT,
    },
    4: {
        "acquisition": {
            "strategy": "ei",
            "params": {}
        },
        "proposer": PROPOSE_NEXT,
    },
    5: {
        "function_1": {
            "acquisition": {
                "strategy": "ei",
                "params": {} # TODO: Implement `xi`
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_2": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_3": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT_RND
        },
        "function_4": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT
        },
        "function_5": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT
        },
        "function_6": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT
        },
        "function_7": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT
        },
        "function_8": {
            "acquisition": {
                "strategy": "ei",
                "params": {}
            },
            "proposer": PROPOSE_NEXT
        },
    }
}

optimal_cycle_values = []
optimal_cycle_values_rnd = []
plot_evaluations = True

cycle = 5
DEFAULT_PROPOSER = 'propose_next'

def bayesian_loop():
    for func_id, data in dataset.items():
        print("\n")
        print(f'Function: {func_id}')
        print(f'Buiding a surrogate model with the following input and output priors')
        X = data['x']
        Y = data['y']
        print(X.shape)
        print(Y.shape)
        if func_id == 1:
            print(X)
            print(Y)
        if plot_evaluations:
            signed_log_plot(func_id, Y)
        assert X.shape[0], Y.shape
        
        # train GP with prior data
        gp = fit_gp(X, Y)

        # Run acquisition based on proposer type
        cycle_cfg = cycle_parameters.get(cycle, {})
        function_cfg = cycle_cfg.get(f"function_{func_id}", {})
        acquisition_cfg = function_cfg.get('acquisition', {})
        proposer = function_cfg.get("proposer", DEFAULT_PROPOSER)
        print(f"Proposer: {proposer}")

        match proposer:
            case "propose_next":
                # Use Gradient method for acquiring optimal next proposal
                _, _, opt_cyc_vals = propose_next(
                    gp, X, Y, func_id, acquisition_cfg
                )
                optimal_cycle_values.append(opt_cyc_vals)
            case "propose_next_rnd_sampling":
                # Use heuristics for acquiring best next sample
                seed = (cycle * NUMBER_OF_FUNCTIONS) + func_id
                _, _, opt_cyc_vals_rnd = propose_next_rnd_sampling(
                    gp, X, Y, func_id, acquisition_cfg, seed=seed
                )
                optimal_cycle_values_rnd.append(opt_cyc_vals_rnd)
            case _:
                raise ValueError(f"Unsupported proposer: {proposer}")

bayesian_loop()

def print_cycle_values(optimal_cycle_values):
    for idx, arr in enumerate(optimal_cycle_values):
        arr = np.asarray(arr)
        formatted = [f"{float(val):.6f}" for val in arr]
        print(f"Function {idx+1}: " + "-".join(formatted))

print_cycle_values(optimal_cycle_values)
print_cycle_values(optimal_cycle_values_rnd)

print('--------')
print(f'Time elapsed: {time.time() - start}secs')
print("Challenge completed!")
print("\n")
