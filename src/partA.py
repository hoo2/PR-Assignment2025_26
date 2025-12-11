# ------------------------------------------------------------
# Part A - Gaussian Parameter Estimation (MLE) & Visualization
# Pattern Recognition – Semester Assignment
#
# Author:
#   Christos Choutouridis (ΑΕΜ 8997)
#   cchoutou@ece.auth.gr
#
# Description:
#   This module implements Part A of the assignment:
#   - Loading and splitting the dataset into classes
#   - MLE estimation of mean vectors and covariance matrices
#   - Construction of Gaussian pdf surfaces
#   - 3D visualization of class-conditional densities
#
# Notes:
#   The implementation follows the theoretical formulation of
#   multivariate Gaussian distributions and MLE parameter
#   estimation as taught in class.
# ------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from toolbox import load_csv, split_dataset_by_class, dataset1

from typing import Tuple, Dict

def mle_mean(X: np.ndarray) -> np.ndarray:
    """
    MLE estimate of the mean vector.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Data samples.

    Returns
    -------
    mu : ndarray, shape (d,)
        Estimated mean vector.
    """
    return np.sum(X, axis=0) / X.shape[0]


def mle_covariance(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    MLE estimate of the covariance matrix.
    (Divide by N, not N-1)

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Data samples.
    mu : ndarray, shape (d,)
        Mean vector.

    Returns
    -------
    cov : ndarray, shape (d, d)
        Covariance matrix.
    """
    N = X.shape[0]
    diff = X - mu
    cov = (diff.T @ diff) / N
    return cov


def estimate_gaussians_mle(classes: Dict[int, np.ndarray]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimates mean and covariance (MLE) for each class.

    Parameters
    ----------
    classes : dict
        Dictionary mapping class label -> samples of that class.

    Returns
    -------
    params : dict
        Dictionary mapping class label -> (mu, cov),
        where mu has shape (d,) and cov has shape (d,d).
    """
    params: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for c, Xc in classes.items():
        mu_c = mle_mean(Xc)
        cov_c = mle_covariance(Xc, mu_c)
        params[c] = (mu_c, cov_c)

    return params


# --------------------------------------------------
# Part A: Gaussian pdf and grid computation
# --------------------------------------------------
def gaussian_pdf(point: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """
    Multivariate Gaussian pdf at a single point (general dimension).

    Parameters
    ----------
    point : ndarray, shape (d,)
        feature data of the point
    mu : ndarray, shape (d,)
        mean vector
    cov : ndarray, shape (d,d)
        covariance array

    Returns
    -------
    value : float
        pdf value at `point`.
    """
    d = mu.shape[0]              # dimension
    diff = point - mu
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)

    # (2π)^(d/2) * sqrt(det Σ)
    norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
    exponent = -0.5 * diff.T @ inv @ diff

    return float(norm_const * np.exp(exponent))


def compute_gaussian_grid(
    X: np.ndarray, mu: np.ndarray, cov: np.ndarray, grid_size: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 2D grid over the range of the first two dimensions of X
    and computes pdf values using the multivariate Gaussian pdf.

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Data samples (only used to define plotting range for dims 0 and 1).
    mu : ndarray, shape (d,)
        mean vector value
    cov : ndarray, shape (d,d)
        covariance
    grid_size : int
        Number of points per axis.

    Returns
    -------
    tuple:
        Xgrid: ndarray, shape (grid_size)
            X Meshgrid coordinates for dimensions 0 and 1
        Ygrid: ndarray, shape (grid_size)
            Y Meshgrid coordinates for dimensions 0 and 1,
        Z: ndarray, shape (grid_size)
            pdf values at each grid point.
    """
    # Range only on the first two dimensions
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), grid_size)
    y_vals = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), grid_size)

    Xgrid, Ygrid = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(Xgrid, dtype=float)

    for i in range(Xgrid.shape[0]):
        for j in range(Xgrid.shape[1]):
            point = np.array([Xgrid[i, j], Ygrid[i, j]])
            Z[i, j] = gaussian_pdf(point, mu, cov)

    return Xgrid, Ygrid, Z


# --------------------------------------------------
# Part A: 3D plotting for multiple classes
# --------------------------------------------------
def plot_gaussians_3d(
    X: np.ndarray, params: Dict[int, Tuple[np.ndarray, np.ndarray]], grid_size: int = 50
) -> None:
    """
    Plots the Gaussian pdfs (MLE estimates) for all classes on a single 3D figure.

    Parameters
    ----------
    X : ndarray, shape (N, 2)
        All data samples (used to define the plotting range).
    params : dict
        Dictionary mapping class label -> (mu, cov).
    grid_size : int
        Resolution of the grid for pdf evaluation.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, (c, (mu_c, cov_c)) in enumerate(params.items()):
        Xgrid, Ygrid, Z = compute_gaussian_grid(X, mu_c, cov_c, grid_size=grid_size)
        ax.plot_surface(Xgrid, Ygrid, Z, alpha=0.6, label=f"Class {c}")

    ax.set_title("MLE Estimated 2D Gaussians (all classes)")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("pdf")
    plt.show()



# --------------------------------------------------
# Part A: convenience runner (optional)
# --------------------------------------------------
if __name__ == "__main__":
    """
    Convenience function to run the whole Part A pipeline:
    - load dataset
    - split by class
    - estimate Gaussian parameters (MLE) per class
    - plot 3D pdf surfaces
    """
    df1 = load_csv(dataset1, header=None)

    X, y, classes = split_dataset_by_class(df1)
    params = estimate_gaussians_mle(classes)

    # Optional parameters printing
    for c, (mu_c, cov_c) in params.items():
        print(f"Class {c}:")
        print("  mu  =", mu_c)
        print("  cov =\n", cov_c)
        print()

    # Plot 3D surfaces
    plot_gaussians_3d(X, params, grid_size=50)