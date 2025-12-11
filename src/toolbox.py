# ------------------------------------------------------------
# Common tools for the entire assignment
#
# Author:
#   Christos Choutouridis (ΑΕΜ 8997)
#   cchoutou@ece.auth.gr
# ------------------------------------------------------------

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame


def github_raw(user, repo, branch, path):
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"


dataset1 = github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset1.csv")
dataset2 = github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset2.csv")
dataset3 = github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset3.csv")
testset  = github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/testset.csv")


def load_csv(path, header=None):
    """
    Loads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(path, header=header)


def split_dataset_by_class(df: DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Splits a dataset into features, labels and per-class subsets with the assumptions that:

    - All columns except the last are feature columns.
    - The last column is the class label.

    Parameters
    ----------
    df: DataFrame
        Data samples as DataFrame.

    Returns
    -------
    tuple:
        X : ndarray, shape (N, d)
            Feature matrix.
        y : ndarray, shape (N,)
            Labels.
        classes : dict
            Dictionary mapping each class label to the subset of X that belongs to that class.

    Example
    -------
        X, y, classes = split_dataset_by_class(df)
    """
    n_cols = df.shape[1]                # Number of columns
    X = df.iloc[:, :n_cols - 1].values  # Features = all columns except last
    y = df.iloc[:, n_cols - 1].values   # Labels = last column

    classes = {c: X[y == c] for c in np.unique(y)}

    return X, y, classes
