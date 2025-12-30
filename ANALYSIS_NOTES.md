### 80/20 train-test baseline + residuals (OLS)
Validation RMSE (log scale): 0.13222726287064454
That’s not terrible for a dumb baseline linear model with lots of dummies.

Interpretation :
- Confirmed there exist relationship/ correlation of the features with house prices.
- 

Residual table with mean near 0 is good (no big bias)
- Linear regression assumption fulfiled: iid errors mean zero.
- std around 0.13 matches RMSE scale.

Here above focus on predictive power rather than RMSE. Therefore, ignoring coefficient's bias.
For plain LinearRegression, it can cause unstable coefficients, but prediction can still work.


###  OLS KFold Interpretation: 
We trained the same OLS model 10 times. Each time, a different 10% of the data was held out. 
We measured prediction error on log(SalePrice).

Mean RMSE: 0.148, which is larger than the 80/20 split: 0.1322.
Std RMSE: 0.045, reflect model is unstable. 	


* After get_dummies, there is ~300 regressors, which many of them are highly correlated, redundant, and mutually exclusive (dummy variables). This violates the “well-conditioned X′X” assumption.If X'X is nearly singular (non-invertible):	•	coefficients explode	•	small data changes → big prediction swings	•	CV variance increases--> Therefore, OLS is unbiased but OLS is high-variance. This is textbook bias–variance tradeoff.

More Reasons:
1. Different data split difficulty
The single 80/20 split might have been “easy” (validation set similar to training).
K-fold averages across many splits, including harder ones → usually more honest about the true model performance.
2.	High variance model
OLS with tons of dummies is unstable. One lucky split (80/20 one) gives a good score, but some folds blow up (you saw 0.25-ish folds). K-fold exposes the true color.
3.	Evaluation noise
80/20 uses only 20% for validation once → noisy estimate for validation.
K-fold uses all points as validation across folds → less luck-driven.

Therefore, while higher, K-fold is the better estimate of true performance of the OLS model.


### Ridge regression interpretation
It is slightly worse than OLS 80/20, due to 
- CV is harsher
- Ridge trades bias ↑ for variance ↓
- Goal is stability, not best lucky split

$\alpha$ ≈ 429 mean?
- Strong multicollinearity (expected with 288 dummies) -> OLS coefficients were unstable
- Ridge shrinks them aggressively
- Prediction improves out-of-sample error
  
- OLS is BLUE only under low collinearity.
Ridge sacrifices unbiasedness to reduce variance.

### Interpretation: OLS vs Ridge Coefficient
➡ Variable's coefficient shrink a lot as it is highly correlated with other regressors

#### Why Ridge helps 
- Many housing characteristics describe the same underlying concepts (size, quality, garage presence, amenities) in this model.
- OLS tries to assign a separate marginal effect to each of them.
- When regressors overlap heavily, OLS coefficients become inflated and unstable.
##### However, 
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
- Our table can then be regarded as a diagnostic for collinearity, not just a comparison of models.

#### Top 5 most-shrunk features — interpretation
##### 1.GarageYrBlt
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
