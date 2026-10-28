# Prompt for a coding agent (Claude Code, Cursor, etc.)

Copy everything below the line into the agent, with this repo as the working directory.

---

You are building a small, fully working machine learning mini project inside this repository. Read `README.md` first; it describes the goal and the folder layout. Use only openly available data and standard open-source libraries. Keep the code simple, readable and commented, because a student must be able to explain it.

## Goal
Classify breast tumours as benign or malignant from the Wisconsin Diagnostic Breast Cancer dataset, compare six models, save the best one, and serve it through a Streamlit app.

## Rules
- Data: `sklearn.datasets.load_breast_cancer()` only. No downloads, no API keys, no invented data.
- Python 3.10+, dependencies limited to `requirements.txt` (add only if truly needed).
- Recode the target so **malignant = 1** (scikit-learn has malignant = 0). Use malignant as the positive class for precision/recall.
- Use a fixed `random_state=42` everywhere.
- Never fit the scaler or feature selector on test data: they must be steps inside a scikit-learn `Pipeline`.
- Do not hard-code or guess any results. Every number must come from actually running the code.
- Make the code runnable from the repo root with the commands in the README.

## Build steps

1. **`src/data.py`**: load the data into a pandas DataFrame, recode the target, check for missing values, and return a stratified 80/20 train/test split.

2. **`src/train.py`**: for each of the 6 models (LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, RandomForestClassifier, SVC with `probability=True`):
   - Build `Pipeline([StandardScaler, SelectKBest(f_classif, k=K), model])`.
   - Run **two experiments**: K=30 (all features) and K=17.
   - Tune with `GridSearchCV`, 5-fold `StratifiedKFold`, `scoring="f1"`, using these grids:
     - LR: `C` in {0.01, 0.1, 1, 10, 100}
     - KNN: `n_neighbors` in {3,5,7,9,11}, `weights` in {uniform, distance}
     - DT: `max_depth` in {3,5,7,None}, `min_samples_split` in {2,5,10}
     - NB: `var_smoothing` in {1e-9, 1e-8, 1e-7}
     - RF: `n_estimators` in {100,200,300}, `max_depth` in {None,5,10}
     - SVM: `C` in {0.1,1,10,100}, `gamma` in {scale, 0.01, 0.001}, `kernel` in {rbf, linear}
   - Evaluate each tuned model **once** on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and training time in seconds.
   - Choose the best model by test recall first, then F1 (state this rule in a comment). Save it with `joblib` to `models/best_model.joblib`, and also save the list of feature names it expects to `models/feature_names.json`.
   - Write everything to `results/metrics.json` (best params per model, all metrics, both experiments, which model won).

3. **Figures** (save PNGs to `results/figures/`, 150+ dpi, clear titles and axis labels):
   - `fig_4_1_class_distribution.png`
   - `fig_4_2_feature_distributions.png` (radius_mean, area_mean, concavity_mean, concave points_mean by class)
   - `fig_4_3_correlation_heatmap.png`
   - `fig_4_4_pca_projection.png` (PC1 vs PC2 coloured by class; print explained variance and the number of components needed for 95% variance)
   - `fig_6_1_confusion_matrix.png` (best model)
   - `fig_6_2_roc_curves.png` (all 6 tuned models, AUC in legend)
   - `fig_6_3_cv_scores.png` (cross-validation F1 per model)
   - `fig_6_4_feature_importance.png` (top 10, from a Random Forest fitted on the training data)

4. **`results/report_numbers.md`**: auto-generate a ready-to-copy sheet with:
   - the best-params table (matches report Table 5.2),
   - test metrics tables for 30 features and for 17 features (matches report Tables 6.3 and 6.4),
   - the best model's confusion matrix counts (TP, FN, FP, TN),
   - training time for 30 vs 17 features,
   - the number of PCA components for 95% variance,
   - one plain sentence naming the champion model with its accuracy and recall.

5. **`app/app.py`** (Streamlit):
   - Load `models/best_model.joblib` and `models/feature_names.json`.
   - Title, short description, and a visible disclaimer: educational tool, not medical advice.
   - Input form with a number input for every feature the model needs, grouped into mean / SE / worst columns, defaulting to the dataset mean.
   - Two buttons, "Load benign example" and "Load malignant example", that fill the form with real rows from the dataset.
   - A Predict button showing the class (Benign / Malignant) with a colour-coded message and the malignant probability.
   - A CSV upload section that predicts for every row and offers a download of the results.
   - Handle errors gracefully (missing model file, wrong CSV columns) with clear messages.

6. **`notebooks/exploration.ipynb`**: a short notebook for EDA only (head, info, class balance, key plots) with markdown explanations.

7. Update `README.md` only where facts changed (e.g. add the real final accuracy table). Do not remove existing sections.

## Verify before you finish
- Run `python src/train.py` from a clean virtual environment and confirm it completes without errors.
- Confirm all 8 figures, `metrics.json`, `report_numbers.md`, and `models/best_model.joblib` exist.
- Start the app with `streamlit run app/app.py`, confirm it loads, and check that a benign and a malignant example both return sensible predictions.
- Test that the CSV upload works using a small CSV exported from the dataset.
- Confirm there is no data leakage: scaler and selector are inside the Pipeline and are never fitted on test data.

## Final response
Summarise: which model won and its test accuracy/recall, the exact commands to run everything, anything that did not work, and any assumption you made. Do not claim anything you did not run.
