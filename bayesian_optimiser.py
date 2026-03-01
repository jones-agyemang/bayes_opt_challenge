import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

import time

import warnings
warnings.filterwarnings('ignore')

from utils.loader import (
    load_input_data,
    load_output_data
)

from core import (
    bayesian_optimisation_nd
)

from log_eval_plts import (signed_log_plot)

# CONSTANTS
DEFAULT_LEN_SCALE = 0.3

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
        "n_iterations": 1,
        "acquisition": { 
            "strategy": "ucb",
            "params": { "kappa": 2.0 }
        }
    },
    2: {
        "n_iterations": 1,
        "acquisition": { 
            "strategy": "ucb",
            "params": { "kappa": 20.0 }
        }
    },
    3: {
        "acquisition": {
            "strategy": "ei",
            "params": {}
        }
    }
}

optimal_cycle_values = []
plot_evaluations = True

cycle = 3
for func_id, data in dataset.items():
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
    
    # Acquisition
    acq_stra = cycle_parameters[cycle]['acquisition']['strategy']
    print(f'Acqusition strategy: {acq_stra}')

    _, _, _, opt_cyc_vals = bayesian_optimisation_nd(X, Y, func_id, acquisition=acq_stra)
    optimal_cycle_values.append(opt_cyc_vals)

def print_cycle_values(optimal_cycle_values):
    for arr in optimal_cycle_values:
        arr = np.asarray(arr)
        formatted = [f"{float(val):.6f}" for val in arr]
        print("-".join(formatted))

print_cycle_values(optimal_cycle_values)

print('--------')
print(f'Time elapsed: {time.time() - start}secs')
print("Challenge completed!")
print("\n")