# -----------------------------
# Toolbox
# -----------------------------

import pandas as pd


def github_raw(user, repo, branch, path):
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"

dataset1 = (github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset1.csv"))
dataset2 = (github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset2.csv"))
dataset3 = (github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/dataset3.csv"))
testset = (github_raw("hoo2", "PR-Assignment2025_26", "master", "datasets/testset.csv"))

def load_csv(path, header=None):
    """
    Loads a CSV file and returns a pandas DataFrame.
    """
    return pd.read_csv(path, header=header)