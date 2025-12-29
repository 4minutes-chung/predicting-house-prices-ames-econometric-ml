# Housing Prices — Model Comparison

This project benchmarks linear and tree-based models on the Ames, Iowa Housing dataset.

**Target**
- log(SalePrice)

**Evaluation**
- 10-fold cross-validated RMSE (log scale)

**Models**
- OLS, Ridge, Lasso, Elastic Net
- Random Forest
- Gradient Boosting
- XGBoost

### Key Result
XGBoost achieved the lowest CV RMSE (≈ 0.117), reflecting the benefits
of shrinkage, shallow trees, and subsampling.

**Notes**
- Dataset sourced from Kaggle, Data not included due to licensing.
- It has 79 explanatory variables describing aspect of residential homes in Ames, Iowa.
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

**Major Results**
- <img width="721" height="234" alt="image" src="https://github.com/user-attachments/assets/58c96bc2-794b-4a85-8a81-b317ecb664e9" />

