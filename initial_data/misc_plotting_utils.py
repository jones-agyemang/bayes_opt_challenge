import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_gp_2d(X_train, y_train, X1, X2, mu, sigma, title="2D Gaussian Process"):
    """
    Plot 2D Gaussian Process with uncertainty visualization.
    
    ...existing code...
    """
    # ...existing code...

def plot_gp_2d(X_train, y_train, X1, X2, mu, sigma, title="2D Gaussian Process"):
    """
    Plot 2D Gaussian Process with uncertainty visualization.
    
    Parameters:
    -----------
    X_train : array-like, training points (n_samples, 2)
    y_train : array-like, training observations
    X1, X2 : 2D arrays, meshgrid coordinates for predictions
    mu : 2D array, predicted mean values
    sigma : 2D array, predicted standard deviation (uncertainty)
    title : str, plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: GP Mean Predictions
    contour_mean = ax1.contourf(X1, X2, mu, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour_mean, ax=ax1, label='Predicted Mean')
    
    # Overlay training points
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, 
               cmap='coolwarm', edgecolors='black', linewidths=2, 
               zorder=10, label='Training observations')
    
    ax1.set_xlabel('x1', fontsize=12)
    ax1.set_ylabel('x2', fontsize=12)
    ax1.set_title(f'{title} - Mean Prediction', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty (Standard Deviation)
    contour_sigma = ax2.contourf(X1, X2, sigma, levels=20, cmap='plasma', alpha=0.8)
    plt.colorbar(contour_sigma, ax=ax2, label='Uncertainty (σ)')
    
    # Overlay training points
    ax2.scatter(X_train[:, 0], X_train[:, 1], c='white', s=100, 
               edgecolors='black', linewidths=2, zorder=10, 
               label='Training observations')
    
    ax2.set_xlabel('x1', fontsize=12)
    ax2.set_ylabel('x2', fontsize=12)
    ax2.set_title(f'{title} - Uncertainty', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)