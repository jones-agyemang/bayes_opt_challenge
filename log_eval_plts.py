# Plot outputs on a signed log scale and save to PNG
import numpy as np
import matplotlib.pyplot as plt

from consts import (EVALUATIONS_DIR)

def signed_log_plot(func_id, outputs, n_expensive=3):
    # Signed log transform (avoid log(0))
    epsilon = 1e-300
    signed_log_outputs = np.sign(outputs) * np.log10(np.abs(outputs) + epsilon)

    indices = np.arange(1, len(outputs) + 1)

    # Create single plot (no custom colours)
    plt.figure()
    plt.plot(indices, signed_log_outputs)
    plt.xlabel("Evaluation Index")
    plt.ylabel("Signed log10(Objective Output)")
    plt.title(f"Func_{func_id}: Signed Log-Scale Bayesian Optimisation Outputs")

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.6)

    # Mark expensive evaluations (assumed last n_expensive points)
    if n_expensive > 0:
        expensive_indices = indices[-n_expensive:]
        for idx in expensive_indices:
            plt.axvline(x=idx, linestyle=":", alpha=0.8)

    # Save to PNG
    file_path = f"{EVALUATIONS_DIR}/signed_log_outputs_{func_id}.png"
    plt.savefig(file_path)
    plt.close()

def multi_signed_log_plot(outputs, n_expensive=3):
    # Signed log transform (avoid log(0))
    epsilon = 1e-300
    signed_log_outputs = np.sign(outputs) * np.log10(np.abs(outputs) + epsilon)

    indices = np.arange(1, len(outputs) + 1)

    # Create single plot (no custom colours)
    plt.figure()
    plt.plot(indices, signed_log_outputs)
    plt.xlabel("Evaluation Index")
    plt.ylabel("Signed log10(Objective Output)")
    plt.title(f"Signed Log-Scale Bayesian Optimisation Outputs")

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.6)

    # Mark expensive evaluations (assumed last n_expensive points)
    if n_expensive > 0:
        expensive_indices = indices[-n_expensive:]
        for idx in expensive_indices:
            plt.axvline(x=idx, linestyle=":", alpha=0.8)

    # Save to PNG
    file_path = f"{EVALUATIONS_DIR}/multi_signed_log_outputs.png"
    plt.savefig(file_path)
    plt.close()