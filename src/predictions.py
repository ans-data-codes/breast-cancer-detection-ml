"""Shared input validation and prediction helpers for the Streamlit app."""

import numpy as np
import pandas as pd


def predict_csv(uploaded_file, model, feature_names):
    """Validate an uploaded CSV and return its predictions and probabilities."""
    try:
        uploaded = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read this CSV file: {exc}") from exc

    missing = [name for name in feature_names if name not in uploaded.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    if uploaded.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    numeric = uploaded[feature_names].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna().any()
    invalid_columns = invalid[invalid].index.tolist()
    if invalid_columns:
        raise ValueError("These required columns contain missing or non-numeric values: " + ", ".join(invalid_columns))
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Required measurements must contain only finite numbers.")

    try:
        labels = model.predict(numeric)
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        malignant_position = classes.index(1)
        malignant_probabilities = model.predict_proba(numeric)[:, malignant_position]
    except Exception as exc:
        raise ValueError(f"CSV prediction failed: {exc}") from exc

    output = uploaded.copy()
    output["prediction"] = pd.Series(labels, index=output.index).map({0: "Benign", 1: "Malignant"})
    output["malignant_probability"] = malignant_probabilities
    return output
