# Indian Liver Patient Dataset — ML Classification

A hands-on supervised learning project comparing five classifiers for predicting liver disease from routine clinical and biochemical data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greymatter1005/indian-liver-patient-ml/blob/main/Indian_Liver_Patient_ML_Assignment.ipynb)

## Dataset

**Indian Liver Patient Dataset (ILPD)**, UCI Machine Learning Repository: 583 patients from north-east Andhra Pradesh, India.

- **Features (10):** age, gender, total and direct bilirubin, alkaline phosphatase, ALT, AST, total proteins, albumin, albumin/globulin ratio
- **Target:** `Dataset` (1 = liver patient, 2 = not a liver patient)
- **Class balance:** 416 liver patients (71.4%) and 167 non-patients (28.6%)

The CSV is not included here. Download `indian_liver_patient.csv` from the UCI repository or Kaggle and upload it when the notebook prompts.

## Workflow

1. Exploratory data analysis: summary statistics, missing values, class balance, gender split, correlation heatmap
2. Preprocessing: label-encode gender, mean-impute the 4 missing albumin/globulin ratio values, 80/20 train-test split (`random_state=42`), standard scaling fitted on the training set only
3. Train and compare Decision Tree, Random Forest, KNN (k=5), Gaussian Naive Bayes and SVM (RBF kernel)
4. Classification report and confusion matrix for the best model

## Results (held-out test set, n = 117)

| Algorithm | Accuracy |
|-----------|----------|
| **SVM (RBF kernel)** | **74.36%** |
| Random Forest | 72.65% |
| Decision Tree | 71.79% |
| KNN | 66.67% |
| Naive Bayes | 51.28% |

## Limitations and next steps

- **Accuracy is a weak metric here.** With 71% of patients in one class, predicting "liver patient" for everyone already scores about 71%, so the best model is only a few points above that baseline. Balanced accuracy, sensitivity/specificity, F1 and ROC-AUC would give a fairer picture.
- **Single split.** Results come from one 80/20 split; stratified k-fold cross-validation with confidence intervals would show how stable the ranking is.
- **No tuning or imbalance handling yet.** Next steps are hyperparameter search, class weighting or resampling, and interpretability (e.g. SHAP) to check which biochemical markers drive predictions.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab.

## Author

Dr Charan Kumar Pottem, psychiatrist (MD, NIMHANS) building skills in machine learning and AI evaluation for healthcare.
