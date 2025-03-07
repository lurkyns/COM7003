import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Load the trained XGBoost model
import xgboost as xgb
best_model = xgb.XGBClassifier()
best_model.load_model("best_xgboost_model.json")  # Make sure your model is saved and available

# Compute ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC Score: {roc_auc:.4f}", color="blue")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost")
plt.legend()
plt.show()
