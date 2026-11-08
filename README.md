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
4. Classification report and confusion matrix for the top-scoring model

## Results (held-out test set, n = 117)

| Algorithm | Accuracy |
|-----------|----------|
| **SVM (RBF kernel)** | **74.36%** |
| Random Forest | 72.65% |
| Decision Tree | 71.79% |
| KNN | 66.67% |
| Naive Bayes | 51.28% |

## Key finding: the top-scoring model learned nothing

The SVM ranks first on accuracy, but its classification report shows it predicted **"liver patient" for all 117 test cases**:

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Liver patient | 0.74 | 1.00 | 0.85 | 87 |
| Not a patient | 0.00 | 0.00 | 0.00 | 30 |

Its 74.36% accuracy is exactly 87/117, the share of liver patients in the test set. A model that always gives the same answer would score the same. Balanced accuracy is 50%, which is chance.

Class imbalance (71% vs 29%) plus accuracy as the only metric hid this. It is a small example of a wider evaluation problem: a headline metric can look fine while the model fails completely on the minority class, which in a clinical setting means missing every healthy person (or, flipped, every sick one).

## Limitations and next steps

- **Use metrics that respect imbalance.** Report balanced accuracy, per-class sensitivity and specificity, F1 and ROC-AUC alongside accuracy, and compare every model with a majority-class baseline (`DummyClassifier`).
- **Handle the imbalance.** Retrain with `class_weight='balanced'` (SVM, Random Forest, Decision Tree) or resampling, and stratify the train-test split.
- **Check the other models per class.** Random Forest and Decision Tree score just below the SVM; their confusion matrices will show whether they genuinely separate the two classes.
- **Single split.** Results come from one 80/20 split; stratified k-fold cross-validation with confidence intervals would show how stable the ranking is.
- **Then tune and interpret.** Hyperparameter search, and SHAP to see which biochemical markers drive predictions.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Google Colab.

## Author

Dr Charan Kumar Pottem, psychiatrist (MD, NIMHANS) building skills in machine learning and AI evaluation for healthcare.
