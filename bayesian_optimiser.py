import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel

import time

import warnings
warnings.filterwarnings('ignore')

from utils.loader import (
    load_input_data,
    load_output_data
)

DEFAULT_LEN_SCALE_BOUND = (1e-3, 1e3)

from utils.cycle_parameters import ( get_cycle_parameters )

from core import (
    fit_gp,
    propose_next,
    propose_next_rnd_sampling
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

cycle_parameters = get_cycle_parameters()
optimal_cycle_values = []
plot_evaluations = True

cycle = 6
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
        
        # Extract Cycle Parameters
        cycle_cfg = cycle_parameters.get(cycle, {})
        function_cfg = cycle_cfg.get(f"function_{func_id}", {})
        
        kernel_cfg              = function_cfg.get('kernel')
        kernel_type             = kernel_cfg.get('type')
        kernel_len_scale        = kernel_cfg.get('length_scale')
        kernel_len_scale_bounds = kernel_cfg.get('length_scale_bounds')
        kernel_nu               = kernel_cfg.get('nu')

        constant_kernel = ConstantKernel(1.0, DEFAULT_LEN_SCALE_BOUND)

        match kernel_type:
            case 'Matern':
                kernel = constant_kernel * Matern(kernel_len_scale, kernel_len_scale_bounds, kernel_nu)
            case 'RBF':
                kernel = constant_kernel * RBF(kernel_len_scale, kernel_len_scale_bounds)
            case _:
                raise ValueError(f'Invalid kernel type: {kernel_type}')

        # train GP with prior data
        gp = fit_gp(X, Y, kernel)

        # Run acquisition based on proposer type
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
                optimal_cycle_values.append(opt_cyc_vals_rnd)
            case _:
                raise ValueError(f"Unsupported proposer: {proposer}")

bayesian_loop()

def print_cycle_values(optimal_cycle_values):
    for idx, arr in enumerate(optimal_cycle_values):
        arr = np.asarray(arr)
        formatted = [f"{float(val):.6f}" for val in arr]
        print(f"Function {idx+1}: " + "-".join(formatted))

print_cycle_values(optimal_cycle_values)

print('--------')
print(f'Time elapsed: {time.time() - start}secs')
print("Challenge completed!")
print("\n")
