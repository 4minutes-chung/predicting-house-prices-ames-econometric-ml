# House Prices (Ames Housing) — Model Comparison

This project benchmarks linear and tree-based models on the Ames Housing dataset.

**Target**
- log(SalePrice)

**Evaluation**
- 10-fold cross-validated RMSE (log scale)

**Models**
- OLS, Ridge, Lasso, Elastic Net
- Random Forest
- Gradient Boosting
- XGBoost

**Best result**
- XGBoost: CV RMSE ≈ 0.117 (stable across folds)

**Notes**
- Dataset source: House Prices - Advanced Regression Techniques, Kaggle.
- It has 79 explanatory variables describing aspect of residential homes in Ames, Iowa.
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
