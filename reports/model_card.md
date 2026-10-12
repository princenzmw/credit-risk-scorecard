# Model card: credit-risk scorecard

**Task:** predict serious delinquency (90+ days past due) within two years.
**Data:** Give Me Some Credit training file, 150,000 borrowers, 6.68% default rate. Competition data from 2011 whose population is not documented; results are not evidence about any current lender.

## Method
- Split 60/20/20 (train/validation/test), stratified, seed 42. The test set is used once, after all choices were made.
- Features built inside the pipeline (medians and caps learned on training data only). Special past-due codes 96/98 are treated as flags, not counts; missing income and dependents get explicit missing-value flags.
- Candidates: logistic regression (scorecard) and gradient boosting. Champion rule: keep the simpler model unless the challenger beats it by more than 0.005 validation ROC-AUC.

## Model comparison (train CV, validation)
| Model | 5-fold CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val KS | Val Brier |
|---|---|---|---|---|---|
| Logistic regression (scorecard) | 0.8228 +/- 0.0072 | 0.8232 | 0.3542 | 0.514 | 0.0516 |
| Gradient boosting (challenger) | 0.8642 +/- 0.0035 | 0.8651 | 0.3936 | 0.572 | 0.0493 |

**Champion:** Gradient boosting (challenger).

## Final test-set result (single evaluation)
ROC-AUC **0.8688**, PR-AUC 0.4072, KS 0.584, Brier 0.0486 (a model with no skill has PR-AUC about 0.067).

## Decision threshold
Chosen on the validation set by minimising cost, assuming approving a defaulter costs 5x as much as declining a good borrower. This ratio is an **assumption**; change `COST_FN` in `creditrisk/config.py` for a different business. Threshold: **0.16**. On the test set this declines 10.7% of applicants and catches 57.2% of defaulters at 35.8% precision.

## Risk bands (test set)
| Band | Loans | Observed default rate |
|---|---|---|
| Low | 17,403 | 1.14% |
| Moderate | 6,432 | 4.96% |
| High | 3,497 | 12.81% |
| Very high | 2,668 | 38.98% |

## Limitations
- Old competition data of unknown origin; no out-of-time validation is possible.
- Age is a model input. Many jurisdictions restrict using age in lending decisions; this is a modelling exercise, not a compliant credit policy.
- No fairness analysis has been run.
- Rejected applicants are absent from the data (selection bias).
- Probabilities were checked for calibration on test data but not recalibrated.
