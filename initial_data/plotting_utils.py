import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_gp_1d(X_train, y_train, X_test, mu, sigma, title="Gaussian Process"):
    """
    Plot 1D Gaussian Process with uncertainty bands.
    
    Parameters:
    -----------
    X_train : array-like, training points
    y_train : array-like, training observations
    X_test : array-like, test points for prediction
    mu : array-like, predicted mean
    sigma : array-like, predicted standard deviation
    title : str, plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(X_test, mu, 'b-', label='Mean prediction', linewidth=2)
    ax.fill_between(X_test.ravel(), 
                     mu - 1.96 * sigma, 
                     mu + 1.96 * sigma, 
                     alpha=0.3, 
                     label='95% confidence')
    ax.scatter(X_train, y_train, c='red', s=50, zorder=10, 
               edgecolors='black', label='Observations')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_acquisition_1d(X_test, acquisition_values, X_next, title="Acquisition Function"):
    """
    Plot 1D acquisition function and next sampling point.
    
    Parameters:
    -----------
    X_test : array-like, test points
    acquisition_values : array-like, acquisition function values
    X_next : float, next point to sample
    title : str, plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(X_test, acquisition_values, 'g-', linewidth=2, label='Acquisition')
    ax.axvline(X_next, color='red', linestyle='--', linewidth=2, 
               label=f'Next sample: x={X_next:.3f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Acquisition Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_bo_iteration_1d(X_train, y_train, X_test, mu, sigma, 
                         acquisition_values, X_next, true_func=None,
                         iteration=0, acq_name="Acquisition"):
    """
    Combined plot showing GP and acquisition function for one BO iteration.
    
    Parameters:
    -----------
    X_train : array-like, current training points
    y_train : array-like, current observations
    X_test : array-like, test points
    mu : array-like, GP mean predictions
    sigma : array-like, GP standard deviations
    acquisition_values : array-like, acquisition function values
    X_next : float, next point to sample
    true_func : callable, optional true function to overlay
    iteration : int, iteration number
    acq_name : str, name of acquisition function
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot GP
    ax1.plot(X_test, mu, 'b-', label='GP Mean', linewidth=2)
    ax1.fill_between(X_test.ravel(), 
                      mu - 1.96 * sigma, 
                      mu + 1.96 * sigma, 
                      alpha=0.3, 
                      label='95% CI')
    if true_func is not None:
        y_true = true_func(X_test)
        ax1.plot(X_test, y_true, 'k--', alpha=0.4, label='True function')
    ax1.scatter(X_train, y_train, c='red', s=50, zorder=10, 
                edgecolors='black', label='Observations')
    ax1.axvline(X_next, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('f(x)', fontsize=11)
    ax1.set_title(f'Iteration {iteration}: GP Surrogate', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot Acquisition
    ax2.plot(X_test, acquisition_values, 'g-', linewidth=2)
    ax2.axvline(X_next, color='red', linestyle='--', linewidth=2, 
                label=f'Next: x={X_next:.3f}')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('Acquisition Value', fontsize=11)
    ax2.set_title(f'{acq_name}', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_2d_function(X1, X2, Z, title="2D Function"):
    """
    Plot a 2D function as a contour plot.
    
    Parameters:
    -----------
    X1, X2 : 2D arrays, meshgrid coordinates
    Z : 2D array, function values
    title : str, plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    contour = ax.contourf(X1, X2, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig, ax


def plot_2d_bo_state(X1, X2, Z, X_samples, y_samples, title="Bayesian Optimization Progress"):
    """
    Plot 2D function with sampled points overlay.
    
    Parameters:
    -----------
    X1, X2 : 2D arrays, meshgrid coordinates
    Z : 2D array, function values
    X_samples : array-like, sampled points (n_samples, 2)
    y_samples : array-like, function values at samples
    title : str, plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    contour = ax.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='f(x1, x2)')
    
    # Plot samples with color based on function value
    scatter = ax.scatter(X_samples[:, 0], X_samples[:, 1], 
                        c=y_samples, s=100, 
                        cmap='coolwarm', edgecolors='black', 
                        linewidths=2, zorder=10)
    
    # Mark the best point found so far
    best_idx = np.argmax(y_samples)
    ax.scatter(X_samples[best_idx, 0], X_samples[best_idx, 1], 
              marker='*', s=500, c='gold', edgecolors='black', 
              linewidths=2, zorder=11, label='Best found')
    
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax


def plot_convergence(iterations, best_values, true_optimum=None):
    """
    Plot convergence of best found value over iterations.
    
    Parameters:
    -----------
    iterations : array-like, iteration numbers
    best_values : array-like, best value found at each iteration
    true_optimum : float, optional true optimal value
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(iterations, best_values, 'b-o', linewidth=2, markersize=6, 
            label='Best value found')
    if true_optimum is not None:
        ax.axhline(true_optimum, color='red', linestyle='--', 
                   linewidth=2, label=f'True optimum: {true_optimum:.4f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best f(x) found', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_parallel_coordinates(X_samples, y_samples, n_best=5):
    """
    Plot parallel coordinates for high-dimensional optimization results.
    
    Parameters:
    -----------
    X_samples : array-like, sampled points (n_samples, n_dims)
    y_samples : array-like, function values
    n_best : int, number of best samples to highlight
    """
    n_dims = X_samples.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Get indices of best samples
    best_indices = np.argsort(y_samples)[-n_best:]
    
    # Plot all samples in gray
    for i in range(len(X_samples)):
        if i not in best_indices:
            ax.plot(range(n_dims), X_samples[i], 'gray', alpha=0.2, linewidth=1)
    
    # Plot best samples in color
    colors = cm.viridis(np.linspace(0.3, 1, n_best))
    for idx, color in zip(best_indices, colors):
        ax.plot(range(n_dims), X_samples[idx], color=color, 
               linewidth=2, alpha=0.8, 
               label=f'f(x)={y_samples[idx]:.3f}')
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Parallel Coordinates - Top {n_best} Solutions', fontsize=14)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'x{i}' for i in range(n_dims)])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax
