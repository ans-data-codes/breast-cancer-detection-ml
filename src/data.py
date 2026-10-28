"""Load and split the Wisconsin Diagnostic Breast Cancer dataset."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


RANDOM_STATE = 42


def load_dataset():
    """Return measurements and labels, with malignant encoded as 1.

    Scikit-learn encodes malignant as 0 and benign as 1. Invert that coding
    so recall and precision treat malignant tumours as the positive class.
    """
    bunch = load_breast_cancer(as_frame=True)
    features = bunch.data.copy()
    # Present scikit-learn's space-separated labels in the conventional WDBC
    # suffix style used by the report and app: radius_mean, radius_se, etc.
    normalized_names = []
    for name in features.columns:
        parts = name.split(" ")
        if parts[0] in {"mean", "worst"}:
            normalized_names.append(" ".join(parts[1:]) + "_" + parts[0])
        elif parts[-1] == "error":
            normalized_names.append(" ".join(parts[:-1]) + "_se")
        else:
            normalized_names.append(name.replace(" ", "_"))
    features.columns = normalized_names
    target = (bunch.target == 0).astype(int).rename("malignant")

    if features.isna().any().any() or target.isna().any():
        raise ValueError("The breast cancer dataset contains missing values.")

    return features, target


def get_train_test_split(test_size=0.2):
    """Return a stratified train/test split with a fixed random state."""
    X, y = load_dataset()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
