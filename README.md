# COM7003
Artificial Intelligence

This project is part of the COM7003 Artificial Intelligence module. 
The goal is to develop a credit risk prediction model that classifies loan applicants as either 
"Fully Paid" (0) or "Defaulted" (1) using machine learning.

# Problem to solve

When financial organisations lend money, they take on risks.  The work estimates loan default risk using machine learning models.  The purpose is to evaluate models and determine the most accurate model to assist banks in making better lending decisions.

# About the dataset & preprocessing

- Dataset source: Kaggle
- Target variable: `loan_status` (0 = Fully Paid, 1 = Defaulted)
- Preprocessing steps:
  - Handled missing values using median imputation & KNN imputer
  - Standardised numerical features (`income`, `loan_amount`, etc.)
  - Applied SMOTE to fix class imbalance
  - One-hot encoded categorical variables (e.g., home ownership, loan purpose)

  # Machine learning models used

1.Logistic Regression- Basic classification model.
2.Random Forest Classifier- Improved accuracy using an ensemble of decision trees.
3.XGBoost- The best model, tuned using GridSearchCV.

# Model evaluetion

-Logistic Regression

Model Accuracy: 0.7259
Classification Report:
               precision    recall  f1-score   support

           0       0.91      0.72      0.80      5066
           1       0.43      0.75      0.54      1418

    accuracy                           0.73      6484
   macro avg       0.67      0.73      0.67      6484
weighted avg       0.80      0.73      0.75      6484

-Random Forest

Model Accuracy: 0.9001
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.95      0.94      5066
           1       0.80      0.72      0.76      1418

    accuracy                           0.90      6484
   macro avg       0.86      0.83      0.85      6484
weighted avg       0.90      0.90      0.90      6484

-XGBoost

Classification Report:
               precision    recall  f1-score   support

           0       0.93      0.98      0.95      5066
           1       0.91      0.74      0.81      1418

    accuracy                           0.93      6484
   macro avg       0.92      0.86      0.88      6484
weighted avg       0.92      0.93      0.92      6484

Key Findings
- XGBoost outperformed other models, making it the best choice.
- SHAP analysis showed income, loan amount, and credit history were the most important factors.#

# Bias & fairness considerations

To ensure ethical AI use, bias analysis was performed:
- Checked the impact of income group & home ownership on predictions.
- Found some bias in predictions, requiring fairness adjustments.
- Applied demographic parity checks to detect unfair disparities.

# How to run the project

-Clone the Repository
-git clone https://github.com/yourusername/Credit_Risk_Analysis.git
cd Credit_Risk_Analysis
-Run data preprocessing: python dataset.py
-Train Models: python XGBoost.py, python logistic _regression.py, python random_forest.py
-Generate predictions & evaluate: python model_comparison.py

# Acknowledgments

Included in the report



