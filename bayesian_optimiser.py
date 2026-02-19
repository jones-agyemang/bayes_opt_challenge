import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.optimize import minimize

from pathlib import Path
import time

import warnings
warnings.filterwarnings('ignore')

from utils.loader import (
    load_evaluation_data,
    load_sample_data
)

# CONSTANTS
DEFAULT_LEN_SCALE = 0.3

# Resolve paths relative to this file so the script works from any CWD
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

load_input = lambda x, type: np.load(f'./initial_data/function_{x}/initial_{type}.npy')

def upper_confidence_bound(mu, sigma, kappa):
    return mu + sigma * kappa

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

"""
Aim: acquisition function to be used for optimisation
"""

def bayesian_optimisation_nd(X, Y, func_id, n_iterations=10, acquisition='ucb'):
    # set bounds
    _, n_dims = X.shape
    bounds = [(0.000001, 0.999999) for _ in range(n_dims)]

    best_values = [Y.max()]

    # Normalise acquisition string once so we don't reference an undefined name
    acquisition_strategy = acquisition.lower()

    for iteration in range(n_iterations):
        kernel = ConstantKernel(1.0) * RBF(length_scale=DEFAULT_LEN_SCALE)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

        gp.fit(X, Y)

        def acq_objective(x):
            # `minimize` passes a 1D array of parameters; convert to 2D for GP
            x = np.atleast_2d(x)
            mu, sigma = gp.predict(x, return_std=True)

            match acquisition_strategy:
                case "ei":
                    acquisition_value = expected_improvement(mu, sigma, Y.max())
                case "ucb":
                    acquisition_value = upper_confidence_bound(mu, sigma, kappa=2)
                case _:
                    raise ValueError(f"Unsupported acquisition strategy: {acquisition_strategy}")

            # We minimise, so return the negative acquisition
            return -acquisition_value

        # Optimise acquisition function (multi-start)
        best_acq = np.inf
        X_next = None
        # optimise to obtain next best random input variable
        for _ in range(1):
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            result = minimize(acq_objective, x0, bounds=bounds, method='L-BFGS-B')

            if result.fun < best_acq:
                best_acq = result.fun
                X_next = np.round(result.x, 6)

        # Evaluate
        ## Submit X_next
        final_iteration = iteration == n_iterations - 1
        if final_iteration:
            print(f'Next optimal input(x) for:')
            print(X_next)
        
        ## Obtain Y_next
        path = PROCESSED_DATA_DIR / 'week_1' / 'outputs.txt'
        Y_next = load_evaluation_data(path)[func_id - 1]

        # Update
        X_samples = np.vstack([X, X_next])
        print(f'X_samples: {X_samples}')

        Y_samples = np.append(Y, Y_next)
        print(f'Y_samples: {Y_samples}')

        best_values.append(Y_samples.max())
        print(best_values)
    
    return X, Y, best_values, X_next

# ------
"""
Run Bayesian Optimisation Challenge 
"""
print("Running Bayesian Optimisation...")
start = time.time()
print('--------')

# Settings
NUMBER_OF_FUNCTIONS = 8 # 8
CYCLE_BUDGET = 12

# Load initial samples
dataset = {
    i: {"x": load_input(i, 'inputs'), "y": load_input(i, 'outputs')}
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
    }
}

optimal_cycle_values = []

for func_id, data in dataset.items():
    print(f'Function {func_id}')
    
    # Run cycles
    for cycle in range(1, CYCLE_BUDGET + 1):
        print(f'Running cycle#{cycle}')
        # check if there's existing data from previous cycle
        prev_week = cycle - 0
        if prev_week >= 1:
            prior_input_data_path = PROCESSED_DATA_DIR / f"week_{prev_week}" / "inputs.txt"
            prior_output_data_path = PROCESSED_DATA_DIR / f"week_{prev_week}" / "outputs.txt"
            
            if prior_input_data_path.exists() & prior_output_data_path.exists():
                print(f'Found prior inputs at {prior_input_data_path}')
                prior_input = load_sample_data(prior_input_data_path)[func_id - 1]
                print(f'Input from week: {prev_week}')
                print(prior_input)
                
                # append previous prior data to initial sample
                x = np.vstack([data["x"], prior_input])

                print(f'Found prior outputs at {prior_output_data_path}')
                prior_output = load_evaluation_data(prior_output_data_path)[func_id - 1]
                print(f'Output from week: {prev_week}')
                print(prior_output)
                
                # append previous prior data to initial output
                y = np.append(data['y'], prior_output)
            
                print(f'Buiding GP with the following input and output priors')
                print(x)
                print(y)


                n_iter = cycle_parameters[cycle]['n_iterations']
                acq_stra = cycle_parameters[cycle]['acquisition']['strategy']

                _, _, _, opt_cyc_vals = bayesian_optimisation_nd(x, y, func_id, n_iterations=n_iter, acquisition=acq_stra)
                print(f'Appending next vlaues to cycle: {opt_cyc_vals}')
                optimal_cycle_values.append(opt_cyc_vals)
            else:
                print(f'No prior inputs for week_{prev_week}.')
        else:
            print(f'No previous week to load (cycle {prev_week})')
        print('\n') 

def print_cycle_values(optimal_cycle_values):
    for arr in optimal_cycle_values:
        # Ensure it's a NumPy array
        arr = np.asarray(arr)
        
        # Format each value to 6 decimal places (no scientific notation)
        formatted = [f"{float(val):.6f}" for val in arr]
        
        # Join with hyphens and print
        print("-".join(formatted))

print_cycle_values(optimal_cycle_values)

print('--------')
print(f'Time elapsed: {time.time() - start}secs')
print("Challenge completed!")
print("\n")