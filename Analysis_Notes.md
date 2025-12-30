> econ-style reasoning notes for the original notebook.  
> Notebook contains code + results; this file keeps interpretation + modeling logic.

## Phase 1 — Linear Models
### 1) 80/20 train-test baseline + residuals (OLS)

**Validation RMSE (log scale):** `0.13222726287064454`  
- That’s not terrible for a dumb baseline linear model with lots of dummies.

**Interpretation**
- Confirmed there exist relationship/ correlation of the features with house prices. Here as a sanity check.

**Residuals**
- Residual table with mean near 0 is good
- Linear regression assumption fulfiled: mean zero errors term.

---

### 2) OLS K-Fold RMSE Interpretation
- I trained the same OLS model 10 times. Each time, a different 10% of the data was held out, and measured prediction error on `log(SalePrice)`.
- **Mean RMSE:** `0.148` (larger than the 80/20 split: `0.1322`)
- **Std RMSE:** `0.045` (reflect model is unstable)

**Main reason (multicollinearity / near-singular X'X)**
- After `get_dummies`, there is ~300 regressors, which many of them are highly correlated, redundant, and mutually exclusive (dummy variables).
- This violates the “well-conditioned \(X′X\)” assumption.
- If \(X'X\) is nearly singular (non-invertible):
  - coefficients explode
  - small data changes → big prediction swings
  - CV variance increases
- Therefore, OLS is unbiased but OLS is high-variance, explaining a higher value here. This is textbook bias–variance tradeoff.

**Conclusion**
- While higher, K-fold is the better estimate of true performance of the OLS model, and will be used in the following as model comparison.

---

### 3) Ridge regression interpretation
It is slightly worse than OLS, due to:
- CV is harsher
- Ridge trades bias ↑ for variance ↓
- Goal is stability, not best lucky split

**\(\alpha \approx 429\) mean?**
- Strong multicollinearity (expected with 288 dummies) -> OLS coefficients were unstable
- Ridge shrinks them aggressively, sacrifices unbiasedness to reduce variance.

---

### 4) Interpretation: OLS vs Ridge Coefficient
➡ Variable's coefficient shrink a lot as it is highly correlated with other regressors.

#### Why Ridge helps
- Many housing characteristics describe the same underlying concepts (size, quality, garage presence, amenities) in this model.
- OLS tries to assign a separate marginal effect to each of them.
- When regressors overlap heavily, OLS coefficients become inflated and unstable.

##### However
- Ridge adds a penalty on coefficient size.
- When OLS attempts to give a large coefficient to a variable that is not uniquely identified, Ridge pushes it back toward zero and forces correlated variables to share explanatory power.
- This can be seen in the above table where variables with large abs_ols but tiny abs_ridge were over-credited by OLS.
- Other regressors were saying almost the same thing.

#### What multicollinearity looks like here
- Multicollinearity here shows up as:
  - Large OLS coefficients
  - Extreme shrinkage under Ridge
  - Shrink ratios close to zero
- This implies some variables looked important alone, but once other correlated features are present, it adds little unique information.
- The table can then be regarded as a diagnostic for collinearity, not just a comparison of models.

---

### 5) Top 5 most-shrunk features — interpretation

##### 1. GarageYrBlt
- Heavily shrunk because house age + garage existence/quality already explain the same variation.
- Predictive message: knowing the exact garage year adds little once age and garage indicators are in.

##### 2. PoolArea
- Shrunk because pool existence/quality and neighborhood already dominate the signal.
- Predictive message: pool size doesn’t add much once you know whether there is a pool and where the house is.

##### 3. PoolQC_None
- Almost wiped out because it’s mechanically tied to other pool indicators.
- Predictive message: multiple dummies encode the same “no pool” information.

##### 4. Garage “None” dummies (GarageQual_None, GarageCond_None, etc.)
- All shrink together because they encode the same binary fact.
- Predictive message: OLS splits credit across them; Ridge compresses them into one weak signal.

##### 5. Exterior2nd_CmentBd
- Shrunk because exterior type overlaps with overall quality + neighborhood.
- Predictive message: exterior material alone doesn’t add much beyond broader quality/location signals.

- Conclusion: Ridge improves prediction by shrinking coefficients that OLS inflated due to overlapping regressors, revealing which variables add little unique predictive information once related features are present.

---

### 6) Interpretation: Lasso

- Lasso achieves similar predictive performance to Ridge and OLS but enforces sparsity by setting many coefficients exactly to zero (223 out of 303).
- This confirms that much of the one-hot encoded feature space is redundant.
- Compared to Ridge, which shrinks correlated predictors together, Lasso selects one representative variable and drops the rest.
- In this dataset, sparsity does not meaningfully improve RMSE, indicating that prediction accuracy is limited more by linear model capacity than by overfitting alone.

---

### 7) Interpretation: Comparison of ALL Linear Model

- RMSEs are very close (≈ 0.135–0.145).
- Ridge and Lasso don’t magically win—they stabilize.
- Differences are smaller than CV noise → expected.

---

### 8) Interpretation: Elastic Net and all Linear Models

- Lasso, Ridge, and Elastic Net deliver very similar cross-validated RMSE on this dataset.
- Elastic Net performs marginally best, but the difference is smaller than CV noise.
- Lasso aggressively sets many coefficients to zero (≈ 75%), acting as variable selection, while Ridge keeps all variables with shrinkage.
- Elastic Net interpolates between the two.
- Since prediction accuracy is the goal, and gains are minimal, this confirms that regularization mainly stabilizes estimation rather than dramatically improving performance in linear models.

---

### 9) Why Lasso and Ridge matters despite low RMSE improvement in CV

- K-fold cross-validation averages prediction error, not fitted models.
- Regularization remains meaningful because it stabilizes each individual estimator, leading to consistently lower variance across folds.
- If Ridge/Lasso have low RMSE across all folds, then a single refit using both the training and validation set after tuning can be assumed low-variance.

---

## Phase 2 — Nonlinear Models

### 10) Interpretation of Random Forest

**Why nonlinear models do not improve dramatically?**
- Nonlinear models such as Random Forest do not deliver large performance gains here because the underlying data-generating process is already well approximated by an additive, monotone structure: house prices are largely driven by size, quality, age, and location effects that enter almost linearly once categorical variables are expanded.
- In econometric terms, the conditional expectation function is close to a sparse linear index, so allowing flexible nonlinearities reduces bias only marginally.

**What Random Forest is actually doing — and why its variance is lower?**
- Random Forest improves performance not by discovering strong nonlinearities, but by averaging many high-variance tree estimators, which stabilizes predictions through bagging.
- The lower RMSE standard deviation across 10 folds reflects this variance-reduction mechanism: while individual trees are unstable, their ensemble average (of 300 trees) is more robust to sample perturbations.
- Compared to linear models, RF therefore trades interpretability for a smoother approximation of the conditional mean with better out-of-sample stability, even when gains in mean RMSE are modest (0.1354 vs 0.1361).

**Overall conclusion**
- Taken together, the results indicate that nonlinear interactions exist but are weakly influential in housing price predictions, while the dominant gains from Random Forest arise from variance reduction rather than fundamentally different functional form discovery.

---
### 11) Why Gradient Boosting Outperforms RF and Linear Models

**Core mechanism difference**
- **Linear / Regularized models** reduce variance but impose a global linear structure, leaving functional bias when relationships are nonlinear or threshold-based.
- **Random Forest (RF)** mainly reduces variance via **bagging**. It stabilizes predictions but does not actively correct residual bias.
- **Gradient Boosting (GBR)** explicitly targets **bias reduction** by fitting trees sequentially to residuals.

**Why GBR works well here**
- Housing prices are largely **additive with weak nonlinearities and strong thresholds** (quality, size, age).
- GBR with **shallow trees (depth = 2)** acts like a nonparametric series estimator, gradually approximating \( E[Y \mid X] \).
- CV favors **many trees + moderate learning rate**, indicating bias—not variance—is the main bottleneck.

**Interpretation of results**
- RF improves stability but delivers modest gains.
- GBR captures remaining structure missed by RF and linear models, producing a large RMSE drop.

**Limitations / next step**
- GBR can overfit and lacks strong regularization.
- This motivates **XGBoost**, which adds shrinkage, subsampling, and explicit regularization for further gains.

---

## XGBoost — Tuning Notes (Process Log)

### 12) CV / Scoring Setup
- CV: KFold(n_splits=10, shuffle=True, random_state=42)
- Target: y_log = log(SalePrice)
- Metric: RMSE on log target (scoring="neg_root_mean_squared_error")
- Base params (fixed unless tuned):
  - objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=-1

### 13) Stepwise (Coarse-to-Fine) Search Log

**Step A: learning_rate × n_estimators**
- grid:
  - learning_rate: [0.04, 0.05, 0.06]
  - n_estimators: [800, 1000, 1200, 1400]
- best: learning_rate=0.05, n_estimators=1000
- CV RMSE (mean): 0.117920

**Step B: subsample × colsample_bytree**
- grid:
  - subsample: [0.80, 0.85, 0.90]
  - colsample_bytree: [0.60, 0.65, 0.70]
- best: subsample=0.85, colsample_bytree=0.70
- CV RMSE (mean): 0.117548

**Step C: min_child_weight × reg_lambda**
- grid:
  - min_child_weight: [1, 3, 5]
  - reg_lambda: [1, 3, 5, 10]
- best: min_child_weight=1, reg_lambda=1
- CV RMSE (mean): 0.117548

**Step D (sanity check): max_depth / gamma / reg_alpha**
- grid:
  - max_depth: [2, 3, 4]
  - gamma: [0, 0.03, 0.06, 0.10]
  - reg_alpha: [0, 0.05, 0.10]
- best: max_depth=3, gamma=0, reg_alpha=0
- CV RMSE (mean): 0.117548

### 14) Final Params + Final CV
- final params:
  - max_depth=3, learning_rate=0.05, n_estimators=1000
  - subsample=0.85, colsample_bytree=0.70
  - min_child_weight=1, reg_lambda=1, gamma=0, reg_alpha=0
  - objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=-1
- Final 10-fold RMSE:
  - mean = 0.117548
  - std  = 0.019275
- Notes:
  - std ~ 0.019 suggests performance is fairly stable across folds.
---

## Phase 2: Nonlinear models conclusion

### 15) Model Comparison
- We compare linear models (OLS, Ridge, Lasso, Elastic Net), bagging (Random Forest), classical boosting (Gradient Boosting), and XGBoost using 10-fold cross-validated RMSE on log prices.
- XGBoost achieves the lowest CV RMSE while maintaining stable variance across folds
- XGBoost provides a clear, interpretable improvement over linear and ensemble baselines by acting as a regularized nonlinear extension rather than a black box.

### 16) Why XGBoost helps: a regularized gradient boosting
1. Additive boosting structure
   - The model builds predictions as a sum of many shallow trees, each correcting residual errors left by previous trees.
   - This allows flexible nonlinear modeling without deep trees.

2. Explicit regularization (key difference)
   - Unlike classical Gradient Boosting, XGBoost penalizes: tree complexity (depth, number of leaves), leaf weights (via L2 regularization), subsampling of rows and features.
   - This controls overfitting by design, not by luck.

3. Bias–variance tradeoff done right
   - Compared to linear models, XGBoost reduces bias by capturing nonlinear interactions.
   - Compared to Random Forests, it reduces variance by sequential error correction and regularization.

### 17) Limitations of XGBoost
- feature engineering choices 
- remaining noise in housing prices
- diminishing returns from further hyperparameter tuning

---

## Appendix — Full code for GridSearch in Xgboost 

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

base_params = dict(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    max_depth=3,
    subsample=0.85,
    colsample_bytree=0.65,
    min_child_weight=1,
    reg_lambda=1,
)

xgb = XGBRegressor(**base_params)

grid_A = {
    "learning_rate": [0.04, 0.05, 0.06],
    "n_estimators": [800, 1000, 1200, 1400],
}

gsA = GridSearchCV(xgb, grid_A, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1, verbose=1)
gsA.fit(X, y_log)
print("A best:", gsA.best_params_, "RMSE:", -gsA.best_score_)
bestA = gsA.best_params_

xgbB = XGBRegressor(**base_params, **bestA)

grid_B = {
    "subsample": [0.80, 0.85, 0.90],
    "colsample_bytree": [0.60, 0.65, 0.70],
}

gsB = GridSearchCV(xgbB, grid_B, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1, verbose=1)
gsB.fit(X, y_log)
print("B best:", gsB.best_params_, "RMSE:", -gsB.best_score_)
bestB = gsB.best_params_

fixed_params = base_params.copy()
fixed_params.update(bestA)   # lr, n_estimators
fixed_params.update(bestB)   # subsample, colsample_bytree

xgbC = XGBRegressor(**fixed_params)

grid_C = {
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1, 3, 5, 10],
}

gsC = GridSearchCV(
    xgbC,
    grid_C,
    scoring="neg_root_mean_squared_error",
    cv=kf,           
    n_jobs=-1,
    verbose=1
)

gsC.fit(X, y_log)

print("C best:", gsC.best_params_, "RMSE:", -gsC.best_score_)
bestC = gsC.best_params_

fixed = base_params.copy()
fixed.update(bestA)
fixed.update(bestB)
fixed.update(bestC)

xgbD = XGBRegressor(**fixed)

grid_D = {
    "max_depth": [2, 3, 4],
    "gamma": [0, 0.03, 0.06, 0.10],
    "reg_alpha": [0, 0.05, 0.10],
}

gsD = GridSearchCV(
    xgbD,
    grid_D,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    n_jobs=-1,
    verbose=1
)

gsD.fit(X, y_log)
print("D best:", gsD.best_params_, "RMSE:", -gsD.best_score_)
bestD = gsD.best_params_
