# Housing Prices — Model Comparison (Ames, Iowa)

## Overview
This project benchmarks linear, regularized, and tree-based models for predicting residential house prices using the Ames Housing dataset, focused on **model comparison and evaluation**.

## Data & Target
- Dataset: Ames, Iowa Housing Market
- Observations: 1,460 training samples
- Features: 79 explanatory variables 
- Target: `log(SalePrice)`
- Dataset source (not included due to license):  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Evaluation
- Metric: **10-fold cross-validated RMSE (log scale)**
- Mean and standard deviation across folds are reported to assess both accuracy and stability.

## Models Compared
- Linear models: OLS, Ridge, Lasso, Elastic Net
- Tree-based models: Random Forest, Gradient Boosting, XGBoost

## Key Result
XGBoost achieved the lowest cross-validated RMSE (≈ **0.118**), suggesting that **regularization, shallow trees, and subsampling** help reduce generalization error in high-dimensional, sparse tabular data.

## Lessons Learned
- Regularization stabilizes linear models under multicollinearity but delivers limited gains without strong feature engineering.
- Boosting methods outperform bagging and linear models when tuned conservatively.
- XGBoost can be interpreted as **regularized boosting**, rather than a black-box improvement.

## Limitations
- No extensive feature engineering or stacking was performed.
- Further work could include interpretability (e.g. SHAP), error analysis, and feature construction.

## Results Summary
![CV RMSE comparison](https://github.com/user-attachments/assets/58c96bc2-794b-4a85-8a81-b317ecb664e9)
