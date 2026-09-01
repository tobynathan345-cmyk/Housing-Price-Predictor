# House Price Predictor

A scikit-learn regression project predicting Ames, Iowa house sale prices, comparing five model types across a properly cross-validated, feature-engineered pipeline.

**Result: 0.13335 RMSLE** on Kaggle's official [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) leaderboard.

## Overview

This project tackles a genuinely messy, high-dimensional dataset (80+ raw columns, ~15 requiring individual missing-data judgment calls) with a fully modular, reusable preprocessing pipeline built with `ColumnTransformer` and nested `Pipeline`s — designed from the outset to generalize correctly between training and test data, informed directly by a validation-methodology gap discovered in an earlier project.

Five models were built and rigorously compared via 5-fold cross-validation:

| Model | Best cross-validated RMSE |
|---|---|
| Linear Regression (baseline) | 29,706 |
| Random Forest (tuned) | 29,069 |
| XGBoost (tuned) | 27,786 |
| Lasso (alpha=200) | 27,341 |
| **Ridge (alpha=10)** | **26,996** |

Both Ridge and XGBoost were submitted to Kaggle to compare against the competition's actual scoring metric (log-scale RMSE) — revealing that model rankings shifted between the raw-dollar metric used for local development and the log-scale metric used for scoring, with **XGBoost (0.13335) outperforming Ridge (0.14144)** on the real leaderboard despite scoring worse locally.

## Key features

- **Modular, leakage-safe preprocessing**: a `ColumnTransformer` combining targeted imputation strategies (median, mode, and constant fills based on individual investigation of *why* each column was missing data) with a general-purpose catch-all imputer, protecting against train/test missingness mismatches (a real issue this dataset exhibits)
- **Feature engineering**: combined `TotalSF`, `SaleAge`, `TotalBath`, and `LastRemod` from raw component columns, validated with before/after cross-validation rather than assumed to help
- **A rejected experiment, documented rather than discarded**: log-transforming the target improved average RMSE but worsened fold-to-fold consistency, and was rejected in favor of a more reliable, generalizable model — a decision later informative once the competition's actual log-scale metric was understood
- **Root-cause error analysis**: investigated the model's largest individual prediction errors and traced them to a specific, documented dataset quirk (atypical `SaleCondition` values like Partial and Family sales), confirming the issue was a genuine data characteristic rather than a pipeline bug
- **Systematic hyperparameter tuning** across all five models, including a full three-way sweep (learning rate, tree depth, estimator count) for XGBoost

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter

## Project structure

```
├── data/
│   ├── raw/              # Kaggle housing data (not tracked — see Setup)
│   └── processed/         # (not tracked)
├── models/                 # trained model pipelines (generated, not tracked)
├── notebooks/               # exploratory data analysis
├── src/
│   ├── preprocessing.py    # clean_raw_data() and build_preprocessor()
│   ├── train_ridge.py      # Ridge regression, alpha sweep, final pipeline
│   ├── train_xgboost.py    # XGBoost, learning_rate/max_depth/n_estimators sweep
│   ├── generate_submission.py  # predict on Kaggle's test set
│   └── paths.py             # shared project path configuration
├── requirements.txt
└── .gitignore
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset via the [Kaggle API](https://www.kaggle.com/docs/api) and place the CSVs in `data/raw/`:
   ```
   kaggle competitions download -c house-prices-advanced-regression-techniques
   ```

## Usage

```
python src/train_ridge.py         # trains and saves the Ridge pipeline
python src/train_xgboost.py       # trains and saves the XGBoost pipeline
python src/generate_submission.py # generates a Kaggle submission.csv
```

Each training script builds its own preprocessing pipeline fresh and fits it jointly with the model in a single `Pipeline.fit()` call — no separately pre-fit preprocessing artifacts are shared between scripts, avoiding a redundant-refit bug identified and fixed during development.

## What this project demonstrates

- End-to-end handling of a large, genuinely messy tabular dataset with targeted, individually-reasoned missing-data strategies (not blanket defaults)
- Building leakage-safe, reusable preprocessing with `ColumnTransformer` and nested `Pipeline`s, including defensive handling of train/test distribution mismatches
- Proper cross-validation methodology, applied specifically to correct a validation-reliability issue identified in prior work
- Feature engineering validated by measurement rather than assumption
- Root-cause investigation of model errors, tracing them to a specific, confirmed data characteristic
- Recognizing and correctly interpreting a mismatch between a development-time evaluation metric and the actual target metric — and adjusting conclusions accordingly once the real, better-aligned result was available
- Systematic hyperparameter tuning across five distinct model families (linear, regularized linear, and tree-based/ensemble)

## Next steps

- Ensembling Ridge and XGBoost predictions
- Ordinal encoding for genuinely ordered quality-rating columns (currently one-hot encoded for simplicity)
- Re-running model comparison and tuning directly against the log-transformed target, given the competition's actual scoring metric
