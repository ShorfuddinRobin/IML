# Interpretable Machine Learning for Stroke Prediction

**Logistic Regression, Random Forest, SHAP, LIME and Partial Dependence Plots**

Author: Syed Md Shorfuddin, M.Sc. Data Science, Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)

## Overview

Stroke is one of the leading causes of death and disability worldwide, and early risk detection matters. In healthcare, accuracy alone is not enough: clinicians also need to understand *why* a model makes a prediction before they can trust it.

This project compares two models for stroke prediction:

- **Logistic Regression (LR):** a simple, directly interpretable baseline.
- **Random Forest (RF):** a more complex ensemble that can capture nonlinear patterns.

Beyond predictive performance, the project focuses on how well each model's decisions can be explained, using **SHAP**, **LIME** and **Partial Dependence Plots (PDPs)** at both global and local levels.

## Dataset

The study uses the public [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset), with about 5,000 patient records covering demographic, medical and lifestyle information. The target is binary: whether the patient had a stroke.

| Feature | Type | Notes |
|---|---|---|
| Age | Numeric | Strong correlation with stroke |
| Gender | Categorical | Encoded |
| BMI | Numeric | Missing values imputed with the mean |
| Avg Glucose Level | Numeric | Key predictor |
| Hypertension | Binary | Known risk factor |
| Heart Disease | Binary | Known risk factor |
| Smoking Status | Categorical | Encoded |
| Work Type | Categorical | Encoded |
| Residence Type | Categorical | Encoded (urban/rural) |

## Methodology

1. **Preprocessing:** mean imputation for missing BMI values; label encoding for categorical variables (gender, smoking status, work type, residence type).
2. **Split:** 80/20 stratified train-test split, keeping class proportions.
3. **Models:** Logistic Regression and Random Forest, trained on the same data.
4. **Evaluation:** Accuracy, AUC, Precision, Recall and F1.
5. **Interpretability:** SHAP (summary and waterfall plots), LIME (local explanations) and PDPs (age and glucose).

## Results

Performance on the test set:

| Model | Accuracy | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.95 | 0.84 | 0.62 | 0.41 | 0.49 |
| Random Forest | 0.93 | 0.80 | 0.59 | 0.37 | 0.45 |

Key findings:

- Logistic Regression slightly outperformed Random Forest in accuracy and AUC, while also being easier to interpret.
- Age, average glucose level, BMI and hypertension were the most influential features across all methods, which agrees with medical knowledge.
- SHAP gave consistent global and local explanations. LIME was easier to read for individual patients but less stable between runs.
- Random Forest PDPs showed nonlinear effects (for example, a sharp increase in risk above a glucose threshold), while LR gave smooth, monotonic curves that are easier to explain to clinicians.

## Limitations and Future Work

- **Class imbalance:** there are far fewer stroke cases than non-stroke cases, which lowered recall. Class weighting, oversampling, SMOTE or cost-sensitive learning could help.
- Test on larger and more diverse datasets.
- Compare with gradient boosting and neural networks combined with interpretability tools.
- Involve clinicians in evaluating the explanations.

## Tech Stack

Python, Pandas, scikit-learn, SHAP, LIME

## References

1. Lundberg, S.M., Lee, S.I.: A Unified Approach to Interpreting Model Predictions. NeurIPS (2017)
2. Ribeiro, M.T., Singh, S., Guestrin, C.: "Why Should I Trust You?" Explaining the Predictions of Any Classifier. KDD (2016)
3. Friedman, J.H.: Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics (2001)
4. Kaggle: Stroke Prediction Dataset
5. Benjamin, E.J. et al.: Heart Disease and Stroke Statistics – 2019 Update. Circulation (2019)
6. Doshi-Velez, F., Kim, B.: Towards A Rigorous Science of Interpretable Machine Learning. arXiv:1702.08608 (2017)
