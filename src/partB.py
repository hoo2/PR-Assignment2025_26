# ------------------------------------------------------------
# Part B - Parzen Window Density Estimation (1D)
# Pattern Recognition – Semester Assignment
#
# Author:
#   Christos Choutouridis (ΑΕΜ 8997)
#   cchoutou@ece.auth.gr
#
# Description:
#   This module implements Part B of the assignment:
#   - 1D Parzen window density estimation using uniform and
#     Gaussian kernels
#   - Computation of predicted likelihood for varying bandwidth h
#   - Comparison with the true N(1, 4) distribution
#   - MSE analysis and optimal bandwidth selection
#
# ------------------------------------------------------------

import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence
from toolbox import load_csv, dataset2


# --------------------------------------------------
# Optional: D-dimensional Bishop-style kernels (not used in Part B)
# --------------------------------------------------
def kernel_hypercube(u: np.ndarray) -> float:
    """
    D-dimensional uniform kernel (hypercube).

    Bishop eq. (2.247):
        k(u) = 1 if |u_i| <= 1/2 for all i, else 0

    In 1D this reduces to:
        k(u) = 1 if |u| <= 1/2 else 0

    This kernel integrates to 1:
        ∫_{-1/2}^{1/2} 1 du = 1
    """
    return float(np.all(np.abs(u) <= 0.5))


def kernel_gaussian(u: np.ndarray) -> float:
    """
    D-dimensional Gaussian kernel.

    k(u) = (2π)^(-D/2) * exp(-||u||^2 / 2)

    Integral over R^D is 1.
    """
    d = u.shape[0]
    norm_const = 1.0 / ((2.0 * np.pi) ** (d / 2.0))
    return float(norm_const * np.exp(-0.5 * np.dot(u, u)))


# --------------------------------------------------
# 1D Parzen kernels (used in this Part)
# --------------------------------------------------
def parzen_kernel_uniform(u: np.ndarray) -> np.ndarray:
    """
    1D uniform Parzen kernel (box).

    Bishop-style in 1D:
        k(u) = 1 if |u| <= 1/2
             = 0 otherwise

    Integral:
        ∫_{-1/2}^{1/2} 1 du = 1

    Parameters
    ----------
    u : ndarray
        Array of values where the kernel is evaluated.

    Returns
    -------
    values : ndarray
        Kernel values at u.
    """
    return (np.abs(u) <= 0.5).astype(float)


def parzen_kernel_gaussian(u: np.ndarray) -> np.ndarray:
    """
    1D Gaussian kernel with mean 0, variance 1.

    k(u) = (1 / sqrt(2π)) * exp(-u^2 / 2)

    Integral over R is 1.

    Parameters
    ----------
    u : ndarray

    Returns
    -------
    values : ndarray
    """
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (u ** 2))


# --------------------------------------------------
# Parzen estimator (1D, point-wise)
# --------------------------------------------------
def parzen_estimate_1d(x_eval: float, data: np.ndarray, h: float, kernel_fn) -> float:
    """
    Parzen window density estimate in 1D, for a single point x_eval.

    Implements:
        p_hat(x_eval) = (1 / (N h)) * sum_n k((x_eval - x_n) / h)

    Parameters
    ----------
    x_eval : float
        Point where the density is estimated.
    data : ndarray, shape (N,)
        1D data samples.
    h : float
        Bandwidth (window width).
    kernel_fn : callable
        Kernel function K(u), applied elementwise on u.

    Returns
    -------
    f_hat : float
        Estimated pdf value at x_eval.
    """
    N = data.shape[0]
    u = (x_eval - data) / h
    return float(np.sum(kernel_fn(u)) / (N * h))


def evaluate_parzen(data: np.ndarray, h: float, kernel_fn) -> np.ndarray:
    """
    Evaluates the Parzen estimate at each sample in 'data' itself.

    For each x_i in data:
        p_hat(x_i) = (1 / (N h)) * sum_n k((x_i - x_n) / h)

    Parameters
    ----------
    data : ndarray, shape (N,)
        1D data samples.
    h : float
        Bandwidth.
    kernel_fn : callable
        Kernel function K(u).

    Returns
    -------
    estimates : ndarray, shape (N,)
        Estimated pdf values at each data point.
    """
    N = data.shape[0]
    estimates = np.zeros(N, dtype=float)

    for i in range(N):
        estimates[i] = parzen_estimate_1d(data[i], data, h, kernel_fn)

    return estimates


# --------------------------------------------------
# True pdf and error
# --------------------------------------------------
def true_normal_pdf_1d(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    True normal pdf N(mu, var) at points x (array).

    Parameters
    ----------
    x : ndarray
        Points where the pdf is evaluated.
    mu : float
        Mean
    var : float
        Variance

    Returns
    -------
    pdf : ndarray
        The normal pdf N(mu, var)
    """
    sigma = np.sqrt(var)
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    z = (x - mu) / sigma
    return coef * np.exp(-0.5 * z * z)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error between two arrays.

    Parameters
    ----------
    y_true : ndarray
    y_pred : ndarray

    Returns
    -------
    err : float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def scan_bandwidths_parzen(
    data: np.ndarray, h_values: Sequence[float], kernel_fn, mu_true: float = 1.0, var_true: float = 4.0
) -> np.ndarray:
    """
    For each h in h_values, computes:

    - estimated pdf via Parzen (predicted likelihood)
    - true pdf via N(mu_true, var_true) (true likelihood)
    - MSE between them

    Parameters
    ----------
    data : ndarray, shape (N,)
        1D data samples.
    h_values : sequence of float
        Bandwidth values to test.
    kernel_fn : callable
        Kernel function K(u).
    mu_true : float
        True mean, default to 1.0.
    var_true : float
        True variance, default to 4.0.

    Returns
    -------
    errors : ndarray, shape (len(h_values),)
        MSE between estimated and true pdf as array of len(h_values)
    """
    true_values = true_normal_pdf_1d(data, mu=mu_true, var=var_true)
    errors_list = []

    for h in h_values:
        est_values = evaluate_parzen(data, h, kernel_fn)
        err = mse(true_values, est_values)
        errors_list.append(err)

    return np.array(errors_list, dtype=float)


# --------------------------------------------------
# Plotting helpers
# --------------------------------------------------
def plot_h_vs_error(h_values: np.ndarray, errors: np.ndarray, title: str) -> None:
    """
    Simple plot of bandwidth vs error.

    Parameters
    ----------
    h_values : ndarray
    errors : ndarray
    title : str
    """
    plt.figure(figsize=(8, 5))
    plt.plot(h_values, errors, marker='o')
    plt.xlabel("h")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_histogram_with_pdf(
    data: np.ndarray, mu_true: float = 1.0, var_true: float = 4.0, bins: int = 30
) -> None:
    """
    Plots a histogram of the data and overlays the true N(mu_true, var_true) pdf.
    """
    plt.figure(figsize=(8, 5))

    plt.hist(data, bins=bins, density=True, alpha=0.5, label="Data histogram")

    x_min, x_max = np.min(data), np.max(data)
    x_plot = np.linspace(x_min, x_max, 200)
    pdf_true = true_normal_pdf_1d(x_plot, mu=mu_true, var=var_true)

    plt.plot(x_plot, pdf_true, label=f"True N({mu_true}, {var_true}) pdf")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Dataset2 histogram vs true N({mu_true}, {var_true}) pdf")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# Part B: main runner
# --------------------------------------------------
if __name__ == "__main__":
    # Load dataset2 (from GitHub via toolbox)
    df2 = load_csv(dataset2, header=None)
    data2 = df2.iloc[:, 0].values

    mu = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    var = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

    # Optional: histogram + true pdf
    plot_histogram_with_pdf(data2, mu_true=mu, var_true=var, bins=30)

    # Range of h: [0.1, 10] with step 0.1
    h_values = np.arange(0.1, 10.1, 0.1)

    # Uniform kernel (parzen)
    errors_uniform = scan_bandwidths_parzen(data2, h_values, parzen_kernel_uniform, mu_true=mu, var_true=var)
    best_h_uniform = h_values[np.argmin(errors_uniform)]

    # Gaussian kernel
    errors_gaussian = scan_bandwidths_parzen(data2, h_values, parzen_kernel_gaussian, mu_true=mu, var_true=var)
    best_h_gaussian = h_values[np.argmin(errors_gaussian)]

    print("Best h (uniform):", best_h_uniform, " with error: ", errors_uniform[np.argmin(errors_uniform)])
    print("Best h (gaussian):", best_h_gaussian, " with error: ", errors_gaussian[np.argmin(errors_gaussian)])

    plot_h_vs_error(h_values, errors_uniform, "Uniform kernel: h vs MSE")
    plot_h_vs_error(h_values, errors_gaussian, "Gaussian kernel: h vs MSE")
