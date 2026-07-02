# Random Forest Classification — Airline Customer Satisfaction Analysis

## Project Summary
This project builds an optimized Random Forest ensemble classifier to predict airline passenger satisfaction using the Invistico Airline dataset (129,880 passengers, 22 features). A three-way split strategy (Train 60% / Validation 20% / Test 20%) with GridSearchCV + PredefinedSplit ensures zero data leakage during hyperparameter tuning. The final model significantly outperforms a single Decision Tree in accuracy, F1-score, and overfitting resistance.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook random_forest_airline.ipynb
```

## Hyperparameter Tuning Approach
- **Method**: GridSearchCV with `PredefinedSplit` (validation set never used for fitting)
- **Grid**: n_estimators=[100,200], max_depth=[15,None], min_samples_leaf=[1,5], max_features=['sqrt'] — 8 combinations
- **Scoring**: F1-score on the validation set (not accuracy, to balance precision/recall)

## Best Hyperparameters Found
| Parameter | Best Value |
|-----------|-----------|
| n_estimators | 200 |
| max_depth | None (unlimited) |
| min_samples_leaf | 1 |
| max_features | sqrt |
| **Best Validation F1** | **0.9590** |

## Side-by-Side Comparison (held-out Test Set)

| Metric | Random Forest | Decision Tree | RF Advantage |
|--------|:---:|:---:|:---:|
| Accuracy | **0.9553** | 0.9396 | +0.0157 |
| Precision | **0.9611** | 0.9491 | +0.0120 |
| Recall | **0.9566** | 0.9392 | +0.0174 |
| **F1-Score** | **0.9588** | 0.9441 | **+0.0147** |
| ROC-AUC | **0.9926** | 0.9753 | +0.0173 |
| Overfit Gap | **0.0447** | 0.0604 | RF less overfit ✅ |

## Top 5 Feature Importance (Random Forest)
| Rank | Feature | RF Importance |
|------|---------|:---:|
| 1 | Inflight entertainment | 0.2672 |
| 2 | Seat comfort | 0.1307 |
| 3 | Online boarding | 0.0848 |
| 4 | Ease of Online booking | 0.0608 |
| 5 | On-board service | 0.0504 |

## Why Random Forest Outperforms Decision Tree
- **Bagging**: 200 trees each trained on different bootstrap samples — noise doesn't affect all trees equally
- **Feature randomness**: Only sqrt(n_features) features considered per split — decorrelates trees so errors cancel when aggregated
- **Lower overfit gap**: RF train-test gap (0.04) < DT gap (0.06) — RF generalizes better to new passengers

## Files
- `random_forest_airline.ipynb` — fully executed notebook (GridSearchCV output, confusion matrix, ROC curves, feature importance, comparison table — all with embedded visualizations)
- `Invistico_Airline.csv` — source dataset
- `README.md` — this file
