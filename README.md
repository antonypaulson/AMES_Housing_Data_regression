# Ames Housing Data Regression

Jupyter notebook project that fits linear, Ridge, and Lasso models to the Ames, Iowa housing data and predicts sale price. Evaluation metric is RMSE.

This is a maintenance pass of a 2019 General Assembly DSI project. It does **not** retrain for a new Kaggle score and does not claim results beyond what was already recorded in this repo.

## What is here

| File | Role |
| --- | --- |
| `Regression & EDA.ipynb` | EDA, cleaning, dummy encoding, Linear / Ridge / Lasso models, Kaggle-style `submission.csv` |
| `train.csv` | Training set (2,051 rows) included in this repo |
| `test.csv` | Test set without `SalePrice` (879 rows) included in this repo |
| `Ames Housing Challenge.pdf` | Original project write-up / challenge PDF |
| `requirements.txt` | Python packages that actually run the notebook |

Models: `LinearRegression`, `LassoCV`, and `RidgeCV` from scikit-learn. Features are numeric columns plus one-hot encodings of categoricals. The final exported predictions come from Lasso on log sale price.

## Results already claimed (2019)

These are the original claims from this repository. This update does not invent new leaderboard numbers.

From the GitHub repo description:

- Linear regression model, **6th position on the Kaggle leaderboard** (DSI-US-8 Project 2 Regression Challenge).

From the original README:

1. The sklearn **Lasso** model had the best accuracy among the models tried.
2. About **91%** of the change in sale price could be accounted for by the variables in the model.
3. The columns most correlated with sale price were **quality** and **living area**.

From the original saved notebook outputs (not re-scored on Kaggle):

- Lasso 5-fold `cross_val_score` mean ≈ **0.911** (this is the ~91% figure).
- Ridge ≈ 0.902, ordinary least squares ≈ 0.902.
- Hold-out RMSE for the log-linear model on 10% of training rows ≈ **18,208**.

Re-running today can produce slightly different numeric output because pandas 2.x / scikit-learn 1.x are not the 2019 stack. Treat the numbers above as the historical record.

## Data source

The CSVs in this repo are the General Assembly DSI split of the Ames Housing dataset, originally compiled by Dean De Cock:

- Challenge (may now be invitation-only): [DSI-US-8 Project 2 Regression Challenge](https://www.kaggle.com/c/dsi-us-8-project-2-regression-challenge/data)
- Variable documentation: [Ames data documentation](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)
- Public Ames / Kaggle analog: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

If `train.csv` or `test.csv` are missing, put them next to the notebook. Do not substitute the public Kaggle “House Prices” files without checking columns: this notebook expects DSI names such as `Gr Liv Area`, `SalePrice`, `Id`, and `PID`.

The files are small (~1 MB total) and already committed here, so no extra download is required for a normal clone.

## How to run

Python **3.10–3.12** (developed / verified on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook "Regression & EDA.ipynb"
```

Run all cells from top to bottom. The last cell writes `submission.csv` (`Id`, `SalePrice`).

Headless check:

```bash
python -m jupyter nbconvert --to notebook --execute "Regression & EDA.ipynb" \
  --inplace --ExecutePreprocessor.timeout=600
```

## Notes on the 2019 code

The notebook is the original analysis with the smallest changes needed to run on current pandas / seaborn / scikit-learn:

- `DataFrame.corr(numeric_only=True)` (pandas 2 no longer silently skips object columns)
- `DataFrame.drop(..., axis=1)` / `pd.concat(..., axis=1)` instead of the positional `1`
- `Series.ffill()` instead of `fillna(method="ffill")`
- `sns.histplot(..., kde=True)` instead of removed `sns.distplot`
- Clear errors if the CSVs are missing, empty, or lack expected columns
- Named constants for the outlier cutoffs (`Gr Liv Area` 4000, `Lot Area` 60000) and the missing-column cutoff (1000)

The modeling choices, log-target, dummy encoding, and Lasso-as-final-model path are unchanged.
