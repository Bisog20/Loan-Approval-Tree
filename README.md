# Loan Approval Prediction - Decision Tree & Random Forest

Comparing a single Decision Tree against a Random Forest for predicting loan approval, with explicit overfitting checks and feature importance analysis.

## Dataset

491 loan applications with applicant demographics, income, credit history, and property area. Target: `Loan_Status` (Approved/Rejected, ~70/30 split).

## Data Cleaning

Six columns had missing values, the largest being `Credit_History` (43 missing - notably, on what turns out to be the most predictive feature in the model). Categorical gaps filled with mode, `LoanAmount` filled with median (robust to skew), and the `"3+"` string in `Dependents` converted to a clean integer.

## Approach

1. **Two models, same depth cap** - Decision Tree and Random Forest, both `max_depth=5`, to keep the comparison fair and guard against overfitting on a small (491-row) dataset
2. **Overfitting check** - compared train vs. test accuracy for both models before trusting any result
3. **Feature importance** - used the Random Forest's importance scores to identify what actually drives approval decisions

## Results

| Metric | Decision Tree | Random Forest |
|---|---|---|
| Train Accuracy | 0.8444 | 0.8138 |
| Test Accuracy | 0.8384 | **0.8485** |
| Train/Test Gap | 0.0060 | –0.0347 |
| Precision | 0.8442 | 0.8462 |
| Recall | 0.9420 | **0.9565** |
| F1 | 0.8904 | **0.8980** |

Neither model shows meaningful overfitting, Random Forest performs slightly better across the board.

## Known Limitation

Both models share the same costly blind spot: of 30 actual rejections in the test set, **12 were misclassified as approved**. For a lender, that's the expensive error (approving someone who should've been denied - real credit risk), while both models are excellent at correctly approving applicants who should be approved. This is named explicitly rather than glossed over a high accuracy score alone would hide this weak spot.

## Feature Importance

`Credit_History` dominates, accounting for **~48%** of the Random Forest's decision-making nearly 5x the next most important feature (`ApplicantIncome`, ~10%). This matches real underwriting practice (past repayment behavior is the strongest predictor of future default risk) and is also a reassuring fairness signal: demographic factors like gender and marital status rank near the bottom of importance.

## Business Framing

Credit history is the single strongest signal in loan approval but the model still misses a meaningful share of applicants who should be denied, so it functions as a decision-support tool for underwriters, not a replacement for their judgment.

## Files

- `loan_approval_tree.ipynb` - full notebook, staged and commented
- `loan_data.csv` - dataset
- `feature_importance.png` - Random Forest feature importance chart
- `requirements.txt` - dependencies

## Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook loan_approval_tree.ipynb
```
# Loan-Approval-Tree
