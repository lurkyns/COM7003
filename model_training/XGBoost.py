import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Define parameter grid for tuning
param_grid = {
    "n_estimators": [100, 200, 300],  # Number of trees
    "max_depth": [3, 5, 7],  # Tree depth
    "learning_rate": [0.01, 0.1, 0.2],  # Learning rate
    "subsample": [0.7, 0.8, 1.0],  # How much data each tree sees
}

# Set the model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Perform GridSearch to find best parameters
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train.values.ravel())

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Tuned XGBoost Model Accuracy: {accuracy:.4f}")
print("Best Parameters:", grid_search.best_params_)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt
import numpy as np

# Extract feature importance
feature_importances = best_model.feature_importances_
features = X_train.columns

# Sort by importance
sorted_idx = np.argsort(feature_importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(features)), feature_importances[sorted_idx], align="center")
plt.xticks(range(len(features)), features[sorted_idx], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC Score: {roc_auc:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost")
plt.legend()
plt.show()

#Suppress all user warnings in your terminal
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
