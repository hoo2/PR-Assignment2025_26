# ------------------------------------------------------------
# Part C - k-Nearest Neighbors Classifier (k-NN)
# Pattern Recognition – Semester Assignment
#
# Author:
#   Christos Choutouridis (ΑΕΜ 8997)
#   cchoutou@ece.auth.gr
#
# Description:
#   This module implements Part C of the assignment:
#   - Implementation of a simple k-NN classifier in 2D
#   - Manual computation of Euclidean distances (no ML libraries)
#   - Probability estimation for any number of classes
#   - Accuracy evaluation for k ∈ [1, 30]
#   - Decision boundary visualization for the best k
# ------------------------------------------------------------

from typing import Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas import DataFrame

from toolbox import load_csv, split_dataset_by_class, dataset3, testset


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------
def load_data(dataset: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads dataset and splits it into features and labels.

    Returns
    -------
    tuple:
        X (ndarray, shape (N, d)):
            Feature vectors.
        y (ndarray, shape (N,)):
            Corresponding class labels.
    """
    df = load_csv(dataset, header=None)
    X, y, _ = split_dataset_by_class(df)
    return X, y


# --------------------------------------------------
# k-NN core functions
# --------------------------------------------------
def eucl(x: np.ndarray, trainData: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance of x from all training samples.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Query point.
    trainData : ndarray, shape (N, d)
        Training feature vectors.

    Returns
    -------
    distances : ndarray, shape (N,)
        Euclidean distance from x to each training point.
    """
    diff = trainData - x              # shape (N, d)
    sq_dist = np.sum(diff * diff, axis=1)
    distances = np.sqrt(sq_dist)
    return distances


def neighbors(x: np.ndarray, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the indices and distances of the k nearest neighbors of x.

    Parameters
    ----------
    x : ndarray, shape (d,)
        data point
    data : ndarray, shape (N, d)
        dataset to search neighbors
    k : int
        Number of neighbors to consider

    Returns
    -------
    tuple:
        neighbor_indices : ndarray, shape (k,)
            Indices of the k nearest neighbors.
        neighbor_distances : ndarray, shape (k,)
            Distances of the k nearest neighbors (ascending order).
    """
    distances = eucl(x, data)
    sorted_indices = np.argsort(distances)
    neighbor_indices = sorted_indices[:k]
    neighbor_distances = distances[neighbor_indices]
    return neighbor_indices, neighbor_distances


def predict(
    X_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int
):
    """
    Predicts class probabilities and labels for each test sample using k-NN.
    Supports an arbitrary number of classes.

    Parameters
    ----------
    X_test : ndarray, shape (N_test, d)
        test features
    X_train : ndarray, shape (N_train, d)
        train features
    y_train : ndarray, shape (N_train,)
        Class labels (may be any discrete integers).
    k : int
        number of neighbors to consider

    Returns
    -------
    tuple:
        probs (ndarray, shape (N_test, C)):
            probs[i, j] = estimated probability of class classes[j] for sample i.
        y_pred (ndarray, shape (N_test,)):
            Predicted label for each test sample.
    """
    classes = np.unique(y_train)
    C = len(classes)
    N_test = X_test.shape[0]

    probs = np.zeros((N_test, C))
    y_pred = np.zeros(N_test, dtype=classes.dtype)

    for i in range(N_test):
        x = X_test[i]
        neighbor_indices, _ = neighbors(x, X_train, k)
        neighbor_labels = y_train[neighbor_indices]

        # Probabilities per class
        for j, c in enumerate(classes):
            probs[i, j] = np.sum(neighbor_labels == c) / k

        # Winner class
        y_pred[i] = classes[np.argmax(probs[i])]

    return probs, y_pred


# --------------------------------------------------
# Accuracy & model evaluation
# --------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy.

    Parameters
    ----------
    y_true : ndarray
        actual labels
    y_pred : ndarray
        predicted labels

    Returns
    -------
    acc : float
        Fraction of correctly classified samples.
    """
    return float(np.mean(y_true == y_pred))


def evaluate_over_k(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    k_values: Sequence[int],
) -> np.ndarray:
    """
    Evaluates k-NN accuracy for multiple values of k.

    Parameters
    ----------
    X_train, y_train:
        training set
    X_test, y_test:
        test set
    k_values :
        sequence of int

    Returns
    -------
    accuracies : ndarray, shape (len(k_values),)
        Accuracy for each value of k.
    """
    accuracies = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        _, y_pred = predict(X_test, X_train, y_train, k)
        accuracies[i] = accuracy(y_test, y_pred)

    return accuracies


def plot_accuracy_vs_k(k_values: np.ndarray, accuracies: np.ndarray) -> None:
    """
    Plots k on the x-axis and accuracy on the y-axis.

    Parameters
    ----------
    k_values: np.ndarray
        sequence of int
    accuracies: np.ndarray
        accuracies array
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker="o")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("k-NN accuracy over k")
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# Decision boundary visualization
# --------------------------------------------------
def plot_decision_boundaries_2d(
    X_train: np.ndarray, y_train: np.ndarray, k: int, grid_size: int = 200
) -> None:
    """
    Plots the decision boundaries of the k-NN classifier in 2D using contourf.
    Supports any number of classes, but requires **exactly 2 features**.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, 2)
        training features
    y_train : ndarray, shape (N_train,)
        training labels
    k : int
        Number of neighbors.
    grid_size : int
        Grid resolution for the contour.
    """
    # --- Check for 2D features ---
    if X_train.shape[1] != 2:
        raise ValueError(
            f"plot_decision_boundaries_2d supports only 2D features, "
            f"but got X_train with shape {X_train.shape}"
        )

    classes = np.unique(y_train)
    C = len(classes)
    class_to_idx = {c: idx for idx, c in enumerate(classes)}

    # Grid limits
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    _, y_pred_grid = predict(grid_points, X_train, y_train, k)

    Z_idx = np.vectorize(class_to_idx.get)(y_pred_grid).reshape(xx.shape)

    # Discrete colormap
    cmap = plt.cm.get_cmap("Set2", C)
    levels = np.arange(C + 1) - 0.5

    plt.figure(figsize=(12, 8))

    # Filled boundaries
    plt.contourf(xx, yy, Z_idx, levels=levels, cmap=cmap, alpha=0.3)

    # Plot samples
    for c, idx in class_to_idx.items():
        mask = (y_train == c)
        plt.scatter(
            X_train[mask, 0], X_train[mask, 1],
            c=[cmap(idx)], edgecolors="k", s=30
        )

    # --- Custom legend: Region + Samples per class ---
    legend_elements = []
    for c, idx in class_to_idx.items():
        color = cmap(idx)
        legend_elements.append(Patch(facecolor=color, edgecolor="none",
                                     alpha=0.3, label=f"Region: class {c}"))
        legend_elements.append(Line2D([], [], marker="o", linestyle="",
                                      markerfacecolor=color,
                                      markeredgecolor="k",
                                      label=f"Samples: class {c}"))

    plt.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"k-NN decision boundaries (k = {k})")
    plt.grid(True)
    plt.show()


# --------------------------------------------------
# Main runner
# --------------------------------------------------
if __name__ == "__main__":
    # Load training and test sets
    X_train, y_train = load_data(dataset=dataset3)
    X_test, y_test   = load_data(dataset=testset)

    # Evaluate over k
    k_values = np.arange(1, 31, 1)
    accuracies = evaluate_over_k(X_train, y_train, X_test, y_test, k_values)

    # Best k
    best_idx = np.argmax(accuracies)
    best_k = int(k_values[best_idx])
    best_acc = accuracies[best_idx]

    print(f"Best k: {best_k} with accuracy: {best_acc:.4f}")

    # Plots
    plot_accuracy_vs_k(k_values, accuracies)
    plot_decision_boundaries_2d(X_train, y_train, best_k, grid_size=200)
