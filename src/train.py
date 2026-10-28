"""Tune, evaluate, save, and report breast tumour classifiers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from .data import RANDOM_STATE, get_train_test_split, load_dataset
except ImportError:  # Allows `python src/train.py` from the repository root.
    from data import RANDOM_STATE, get_train_test_split, load_dataset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

MODEL_SPECS = [
    ("Logistic Regression", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), {"model__C": [0.01, 0.1, 1, 10, 100]}),
    ("K-Nearest Neighbors", KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7, 9, 11], "model__weights": ["uniform", "distance"]}),
    ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE), {"model__max_depth": [3, 5, 7, None], "model__min_samples_split": [2, 5, 10]}),
    ("Gaussian Naive Bayes", GaussianNB(), {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
    ("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE), {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 5, 10]}),
    ("Support Vector Machine", SVC(probability=True, random_state=RANDOM_STATE), {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.001], "model__kernel": ["rbf", "linear"]}),
]


def make_pipeline(k, estimator):
    """Keep scaling and feature selection inside CV to prevent leakage."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            ("model", estimator),
        ]
    )


def evaluate(estimator, X_test, y_test):
    """Calculate held-out metrics once for the fitted best estimator."""
    predicted = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics, probabilities


def save_eda_figures(X, y, X_train):
    """Create the four exploratory figures and return PCA summary values."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    class_names = y.map({0: "Benign", 1: "Malignant"})

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = class_names.value_counts().reindex(["Benign", "Malignant"])
    ax.bar(counts.index, counts.values, color=["#4c9f70", "#cf5c60"])
    ax.set(title="Class distribution in the WDBC dataset", xlabel="Tumour class", ylabel="Number of samples")
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height())}", (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_4_1_class_distribution.png", dpi=160)
    plt.close(fig)

    key_features = ["radius_mean", "area_mean", "concavity_mean", "concave points_mean"]
    long_data = X[key_features].copy()
    long_data["Class"] = class_names.values
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feature in zip(axes.flat, key_features):
        grouped = [long_data.loc[long_data["Class"] == label, feature] for label in ("Benign", "Malignant")]
        artists = ax.boxplot(grouped, patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Benign", "Malignant"])
        for artist, color in zip(artists["boxes"], ("#4c9f70", "#cf5c60")):
            artist.set_facecolor(color)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Tumour class")
    fig.suptitle("Selected feature distributions by tumour class", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_4_2_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, square=True, linewidths=0.15, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Feature correlation heatmap")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_4_3_correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # Fit the PCA display basis on training rows only, then project all rows.
    # The test rows are transformed for the plot but never used to fit it.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    scaled_train = scaler.transform(X_train)
    pca = PCA(random_state=RANDOM_STATE).fit(scaled_train)
    projection = pca.transform(scaler.transform(X))[:, :2]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=y, cmap="coolwarm", alpha=0.75, edgecolors="none")
    handles, _ = scatter.legend_elements()
    ax.legend(handles, ["Benign", "Malignant"], title="Class")
    ax.set(
        title="PCA projection of standardized tumour measurements",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_4_4_pca_projection.png", dpi=160)
    plt.close(fig)

    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")
    print(f"PCA components needed for 95% variance: {components_95}")
    return {"pc1_explained_variance": float(pca.explained_variance_ratio_[0]), "pc2_explained_variance": float(pca.explained_variance_ratio_[1]), "components_for_95_percent": components_95}


def save_model_figures(experiments, best, y_test):
    """Create the four model evaluation figures."""
    best_metrics = best["metrics"]

    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array([[best_metrics["confusion_matrix"]["tn"], best_metrics["confusion_matrix"]["fp"]], [best_metrics["confusion_matrix"]["fn"], best_metrics["confusion_matrix"]["tp"]]])
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set(title=f"Best model: {best['model']} (K={best['k']})", xlabel="Predicted class", ylabel="Actual class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_6_1_confusion_matrix.png", dpi=160)
    plt.close(fig)

    # For a single ROC line per model, show that model's CV-best feature count.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in [spec[0] for spec in MODEL_SPECS]:
        choices = [item for item in experiments if item["model"] == name]
        chosen = max(choices, key=lambda item: item["cv_f1"])
        probs = chosen["test_probabilities"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = chosen["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (K={chosen['k']}, AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set(title="ROC curves for the CV-best configuration of each model", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_6_2_roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    score_data = pd.DataFrame([{"Model": item["model"], "Selected features (K)": str(item["k"]), "CV F1": item["cv_f1"]} for item in experiments])
    sns.barplot(data=score_data, x="Model", y="CV F1", hue="Selected features (K)", ax=ax, palette="Set2")
    ax.set(title="Cross-validation F1 by model and feature count", xlabel="Model", ylabel="Mean CV F1")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_6_3_cv_scores.png", dpi=160)
    plt.close(fig)

    # The tuned Random Forest's GridSearchCV refit used all training rows.
    rf = max((item for item in experiments if item["model"] == "Random Forest"), key=lambda item: item["cv_f1"])
    rf_pipeline = rf["estimator"]
    selected_names = np.asarray(rf_pipeline.named_steps["selector"].get_feature_names_out())
    importances = rf_pipeline.named_steps["model"].feature_importances_
    top_indices = np.argsort(importances)[-10:][::-1]
    importance_data = pd.DataFrame({"Feature": selected_names[top_indices], "Importance": importances[top_indices]}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_data["Feature"], importance_data["Importance"], color="#4c78a8")
    ax.set(title=f"Top 10 Random Forest feature importances (K={rf['k']})", xlabel="Mean decrease in impurity", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_6_4_feature_importance.png", dpi=160)
    plt.close(fig)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(metrics, pca_summary, feature_names):
    """Generate a ready-to-copy set of report tables from measured results."""
    rows = metrics["experiments"]
    lines = [
        "# Report numbers (generated by `python src/train.py`)",
        "",
        "All test metrics below are measured on the same stratified 20% holdout. Malignant is the positive class. Training time includes the five-fold grid search and refit.",
        "",
        "## Best parameters (report Table 5.2)",
        "",
        "| Model | Features (K) | Best parameters | Best mean CV F1 |",
        "|---|---:|---|---:|",
    ]
    for item in rows:
        lines.append(f"| {item['model']} | {item['k']} | `{json.dumps(item['best_params'], sort_keys=True)}` | {item['cv_f1']:.4f} |")

    for k in (30, 17):
        lines += [
            "",
            f"## Test metrics: {k} features (report Table 6.{3 if k == 30 else 4})",
            "",
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in rows:
            if item["k"] != k:
                continue
            m = item["metrics"]
            lines.append(f"| {item['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {item['training_time_seconds']:.3f} |")

    best = metrics["best_model"]
    cm = best["metrics"]["confusion_matrix"]
    alternatives = {item["k"]: item for item in rows if item["model"] == best["model"]}
    t30 = alternatives[30]["training_time_seconds"]
    t17 = alternatives[17]["training_time_seconds"]
    lines += [
        "",
        "## Champion details",
        "",
        f"- Champion: **{best['model']}** with K={best['k']} features.",
        f"- Confusion matrix counts: TP={cm['tp']}, FN={cm['fn']}, FP={cm['fp']}, TN={cm['tn']}.",
        f"- Champion model training time for K=30: {t30:.3f} seconds; for K=17: {t17:.3f} seconds.",
        f"- PCA components needed for 95% variance: {pca_summary['components_for_95_percent']}.",
        f"- Model input has {len(feature_names)} measurements; SelectKBest applies the selected K inside the saved pipeline.",
        "",
        f"The champion is **{best['model']}**, with test accuracy **{best['metrics']['accuracy']:.4f}** and recall **{best['metrics']['recall']:.4f}**.",
        "",
    ]
    (RESULTS_DIR / "report_numbers.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split()
    pca_summary = save_eda_figures(X, y, X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    experiments = []
    for model_name, prototype, grid in MODEL_SPECS:
        for k in (30, 17):
            print(f"Tuning {model_name} with K={k}...", flush=True)
            search = GridSearchCV(
                make_pipeline(k, prototype),
                param_grid=grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            started = time.perf_counter()
            search.fit(X_train, y_train)
            training_time = time.perf_counter() - started
            # This is the only held-out test evaluation for this tuned run.
            heldout, test_probabilities = evaluate(search.best_estimator_, X_test, y_test)
            experiments.append(
                {
                    "model": model_name,
                    "k": k,
                    "best_params": search.best_params_,
                    "cv_f1": float(search.best_score_),
                    "training_time_seconds": float(training_time),
                    "metrics": heldout,
                    "test_probabilities": test_probabilities.tolist(),
                    "estimator": search.best_estimator_,
                }
            )
            print(f"  CV F1={search.best_score_:.4f}; test recall={heldout['recall']:.4f}; {training_time:.2f}s", flush=True)

    # Champion selection follows the requested rule: test recall first, then F1.
    best = max(experiments, key=lambda item: (item["metrics"]["recall"], item["metrics"]["f1"]))
    feature_names = list(X.columns)
    dump(best["estimator"], MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    save_model_figures(experiments, best, y_test)

    json_experiments = [
        {key: value for key, value in item.items() if key not in {"estimator", "test_probabilities"}}
        for item in experiments
    ]
    metrics = {
        "dataset": "Wisconsin Diagnostic Breast Cancer (scikit-learn)",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "positive_class": "malignant",
        "feature_names": feature_names,
        "pca": pca_summary,
        "selection_rule": "Highest held-out test recall, then highest held-out test F1.",
        "experiments": json_experiments,
        "best_model": {
            key: value for key, value in best.items() if key not in {"estimator", "test_probabilities"}
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics, pca_summary, feature_names)
    print(f"\nChampion: {best['model']} (K={best['k']}) — accuracy={best['metrics']['accuracy']:.4f}, recall={best['metrics']['recall']:.4f}")
    print(f"Saved model, metrics, report, and figures under {ROOT}")


if __name__ == "__main__":
    main()
