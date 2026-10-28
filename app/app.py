"""Streamlit interface for the saved breast tumour classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictions import predict_csv


st.set_page_config(page_title="Breast Tumour Classifier", page_icon="🩺", layout="wide")
st.title("Breast Tumour Classification")
st.write(
    "Enter the 30 cell nucleus measurements from a fine needle aspirate to see "
    "the saved model's prediction. The model uses the Wisconsin Diagnostic "
    "Breast Cancer dataset."
)
st.warning("Educational tool only. This is not medical advice and must not be used for diagnosis or treatment decisions.")

MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_names.json"

if not MODEL_PATH.exists():
    st.error("The trained model is missing. From the repository root, run `python src/train.py` first.")
    st.stop()
if not FEATURES_PATH.exists():
    st.error("The model feature list is missing. From the repository root, run `python src/train.py` first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    from src.data import load_dataset

    dataset_X, dataset_y = load_dataset()
except Exception as exc:  # Keep startup problems understandable in the app.
    st.error(f"Could not load the model or its feature schema: {exc}")
    st.stop()

if not isinstance(feature_names, list) or not feature_names:
    st.error("The model feature list is empty or invalid. Run the training script again.")
    st.stop()
missing_from_dataset = [name for name in feature_names if name not in dataset_X.columns]
if missing_from_dataset:
    st.error(f"The saved model feature list does not match the dataset: {', '.join(missing_from_dataset)}")
    st.stop()

defaults = dataset_X[feature_names].mean()
groups = {
    "Mean measurements": [name for name in feature_names if name.endswith("_mean")],
    "Standard error measurements": [name for name in feature_names if name.endswith("_se")],
    "Worst measurements": [name for name in feature_names if name.endswith("_worst")],
}
assigned = sum(groups.values(), [])
unassigned = [name for name in feature_names if name not in assigned]
if unassigned:
    groups["Other measurements"] = unassigned

st.subheader("Enter measurements")
benign_rows = np.flatnonzero(dataset_y.to_numpy() == 0)
malignant_rows = np.flatnonzero(dataset_y.to_numpy() == 1)
button_cols = st.columns(2)
if button_cols[0].button("Load benign example", use_container_width=True):
    example = dataset_X.iloc[int(benign_rows[0])]
    for name in feature_names:
        st.session_state[f"measurement_{name}"] = float(example[name])
if button_cols[1].button("Load malignant example", use_container_width=True):
    example = dataset_X.iloc[int(malignant_rows[0])]
    for name in feature_names:
        st.session_state[f"measurement_{name}"] = float(example[name])

with st.form("prediction_form"):
    columns = st.columns(len(groups))
    for column, (group_name, names) in zip(columns, groups.items()):
        with column:
            st.markdown(f"#### {group_name}")
            for name in names:
                key = f"measurement_{name}"
                st.number_input(
                    name.replace("_", " "),
                    value=float(defaults[name]),
                    key=key,
                    format="%.6f",
                    help=f"Dataset mean: {defaults[name]:.6f}",
                )
    predict_clicked = st.form_submit_button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    values = {name: float(st.session_state[f"measurement_{name}"]) for name in feature_names}
    sample = pd.DataFrame([values], columns=feature_names)
    try:
        prediction = int(model.predict(sample)[0])
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        malignant_position = classes.index(1)
        probability = float(model.predict_proba(sample)[0][malignant_position])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        if prediction == 1:
            st.error(f"Prediction: Malignant — estimated malignant probability: {probability:.1%}")
        else:
            st.success(f"Prediction: Benign — estimated malignant probability: {probability:.1%}")
        st.caption("This probability is a model output, not a clinical risk estimate.")

st.divider()
st.subheader("Predict from a CSV")
st.write("Upload a CSV containing one row per sample and a column for each required measurement. Additional columns are kept in the downloadable output.")
upload = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
if upload is not None:
    try:
        output = predict_csv(upload, model, feature_names)
    except Exception as exc:
        st.error(str(exc))
    else:
        st.success(f"Predictions created for {len(output)} row(s).")
        st.dataframe(output, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="breast_cancer_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
