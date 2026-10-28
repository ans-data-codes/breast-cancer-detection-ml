# Breast Cancer Detection Using Machine Learning

A mini project for **Machine Learning Lab (PCC353AID)**, Third Year Artificial Intelligence and Data Science.

It trains and compares six machine learning models that decide whether a breast tumour is **benign** or **malignant**, then serves the best one in a small web app.

> Educational project only. It is **not** a medical tool and must not be used for real diagnosis.

---

## What this project does (in plain English)

1. A doctor takes a tiny sample of cells from a breast lump (a fine needle aspirate). A computer measures the cell nuclei in the image: size, shape, texture and so on.
2. Those measurements become **30 numbers** per patient. This is the public **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset: 569 patients, 357 benign, 212 malignant.
3. We show a model many past cases with known answers so it can learn the pattern. This is called *supervised classification*.
4. We train **six different models**, compare them fairly, and keep the best.
5. The best model is put behind a simple **Streamlit** web page: type in measurements, get a prediction and a probability.

The key idea: missing a cancer (false negative) is much worse than a false alarm, so we care about **recall**, not just accuracy.

## Data

- Source: UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic). Free and public.
- It also ships inside scikit-learn: `sklearn.datasets.load_breast_cancer()`, so **no download is needed**.
- Note: scikit-learn codes malignant as 0 and benign as 1. This project flips it so **malignant = 1**.

## Pipeline

```
Load data -> Check/clean -> Stratified 80/20 split -> StandardScaler -> SelectKBest (ANOVA)
          -> Train + tune 6 models (GridSearchCV, 5-fold) -> Evaluate on held-out test set
          -> Save best model (joblib) -> Streamlit app
```

Scaling and feature selection sit **inside a scikit-learn Pipeline**, so they only ever learn from training data (no data leakage).

Models: Logistic Regression, KNN, Decision Tree, Naive Bayes, Random Forest, SVM.
Each is run with all 30 features and with the top 17 features.
Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

## Project structure

```
breast-cancer-detection-ml/
├── README.md
├── AGENT_PROMPT.md        # prompt to hand to a coding agent to build everything
├── requirements.txt
├── src/                   # data loading, training, evaluation code
├── app/                   # Streamlit app (app.py)
├── models/                # saved best model (best_model.joblib)
├── notebooks/             # exploration notebook
├── results/
│   ├── figures/           # plots used in the report
│   ├── metrics.json       # all numbers for the report tables
│   └── report_numbers.md  # fill-in sheet matching the report placeholders
└── docs/                  # the written mini project report (.docx)
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py              # trains, tunes, evaluates, saves model + figures + metrics
streamlit run app/app.py         # opens the web app in your browser
```

## Using the results in the report

Open `results/report_numbers.md` and copy values into the yellow placeholders in `docs/`. Insert the images from `results/figures/` into the matching figure boxes. In Word, press Ctrl+A then F9 to refresh the contents and figure lists.

## Papers this work is based on

1. Elsadig et al., "Breast cancer detection using machine learning approaches: a comparative study," IJECE, 2023.
2. Mohammed et al., "Analysis of breast cancer detection using different machine learning techniques," DMBD 2020.
3. Abunasser et al., "Breast cancer detection and classification using deep learning Xception algorithm," IJACSA, 2022.
4. Asri et al., "Using machine learning algorithms for breast cancer risk prediction and diagnosis," Procedia Computer Science, 2016.

## Authors

[Your names, roll numbers/PRNs, guide, college, academic year]
