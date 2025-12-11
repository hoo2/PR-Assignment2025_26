# ------------------------------------------------------------
# Part D - TV Dataset Classifier
# Pattern Recognition – Semester Assignment
#
# Author:
#   Christos Choutouridis (ΑΕΜ 8997)
#   cchoutou@ece.auth.gr
#
# Description:
#   This module implements a complete classification pipeline
#   for the high-dimensional TV dataset (Part D):
#   - Loading training and test data
#   - Basic preprocessing (scaling, optional dimensionality reduction)
#   - Training a supervised classifier
#   - Evaluating on a validation split
#   - Predicting labels for the provided test set
#   - Saving labels to labelsX.npy as required by the assignment
#
# Notes:
#   The exact choice of classifier and preprocessing steps can
#   be modified. The current skeleton uses a RandomForest model
#   as a robust default for high-dimensional data.
# ------------------------------------------------------------

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# from sklearn.decomposition import PCA  # Optional, if you decide to use PCA

from toolbox import load_csv, datasetTV, datasetTest


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_tv_training() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the TV training dataset (Part D) and splits it into
    features and labels.

    Returns
    -------
    tuple:
        X_train (ndarray, shape (N_train, D)):
            Training feature matrix.
        y_train (ndarray, shape (N_train,)):
            Training class labels (1..5).
    """
    df = load_csv(datasetTV, header=None)
    data = df.values
    X_train = data[:, :-1]
    y_train = data[:, -1].astype(int)
    return X_train, y_train


def load_tv_test() -> np.ndarray:
    """
    Loads the TV test dataset (Part D) without labels.

    Returns
    -------
    X_test (ndarray, shape (N_test, D)):
        Test feature matrix (no labels).
    """
    df = load_csv(datasetTest, header=None)
    X_test = df.values
    return X_test


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_features(
    X_train: np.ndarray,
    X_test: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, StandardScaler]:
    """
    Applies basic preprocessing to the feature matrices.
    By default, standardizes features (zero mean, unit variance).

    Parameters
    ----------
    X_train : ndarray, shape (N_train, D)
        Training features.
    X_test : ndarray, shape (N_test, D) or None
        Test features, if available.

    Returns
    -------
    tuple:
        X_train_proc (ndarray):
            Preprocessed training features.
        X_test_proc (ndarray or None):
            Preprocessed test features (if X_test is not None).
        scaler (StandardScaler):
            Fitted scaler object (can be reused later).
    """
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_proc = scaler.transform(X_test)
    else:
        X_test_proc = None

    # If later θέλεις PCA:
    # pca = PCA(n_components=some_k)
    # X_train_proc = pca.fit_transform(X_train_proc)
    # if X_test_proc is not None:
    #     X_test_proc = pca.transform(X_test_proc)

    return X_train_proc, X_test_proc, scaler


# --------------------------------------------------
# Model training & evaluation
# --------------------------------------------------
def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a supervised classifier on the given features and labels.

    Currently uses a RandomForestClassifier as a robust default,
    but this can be replaced with any other model.

    Parameters
    ----------
    X_train : ndarray, shape (N_train, D)
    y_train : ndarray, shape (N_train,)

    Returns
    -------
    model (RandomForestClassifier):
        Trained classifier.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Evaluates a trained classifier on a validation set.

    Parameters
    ----------
    model :
        Any scikit-learn-like classifier with .predict method.
    X_val : ndarray, shape (N_val, D)
    y_val : ndarray, shape (N_val,)

    Returns
    -------
    acc : float
        Classification accuracy on the validation set.
    """
    y_pred = model.predict(X_val)
    acc = float(np.mean(y_pred == y_val))
    return acc


# --------------------------------------------------
# Prediction & saving labels
# --------------------------------------------------
def predict_labels(
    model,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Predicts labels for the TV test set.

    Parameters
    ----------
    model :
        Trained classifier.
    X_test : ndarray, shape (N_test, D)

    Returns
    -------
    labels (ndarray, shape (N_test,)):
        Predicted class labels for each test sample.
    """
    labels = model.predict(X_test)
    return labels.astype(int)


def save_labels(labels: np.ndarray, filename: str = "labelsX.npy") -> None:
    """
    Saves predicted labels to a .npy file as required by the assignment.

    Parameters
    ----------
    labels : ndarray, shape (N_test,)
        Predicted class labels.
    filename : str
        Output filename (default: "labelsX.npy").
    """
    np.save(filename, labels)
    print(f"Saved labels to {filename} with shape {labels.shape}")


# --------------------------------------------------
# Main pipeline for Part D
# --------------------------------------------------
if __name__ == "__main__":
    # 1. Load training and test sets
    X_train_raw, y_train = load_tv_training()
    X_test_raw = load_tv_test()

    # 2. Train/validation split on the training data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        random_state=0,
        stratify=y_train,
    )

    # 3. Preprocess features (scaling, optional PCA)
    X_tr_proc, X_val_proc, scaler = preprocess_features(X_tr, X_val)

    # 4. Train classifier
    model = train_classifier(X_tr_proc, y_tr)

    # 5. Evaluate on validation set
    val_acc = evaluate_classifier(model, X_val_proc, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # 6. Retrain on full training set (optional but συνήθως καλό)
    X_full_proc, X_test_proc, _ = preprocess_features(X_train_raw, X_test_raw)
    final_model = train_classifier(X_full_proc, y_train)

    # 7. Predict labels for official test set
    labels = predict_labels(final_model, X_test_proc)

    # 8. Save labels to labelsX.npy
    save_labels(labels, filename="labelsX.npy")
