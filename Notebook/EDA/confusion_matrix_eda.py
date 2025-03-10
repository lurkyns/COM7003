import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load test data
y_test = pd.read_csv("y_test.csv")["loan_status"]

# Load predictions from models
y_pred_logistic = pd.read_csv("y_pred_logistic.csv")["loan_status"]
y_pred_rf = pd.read_csv("y_pred_rf.csv")["loan_status"]
y_pred_xgb = pd.read_csv("y_pred_xgb.csv")["loan_status"]

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fully Paid (0)", "Defaulted (1)"],
                yticklabels=["Fully Paid (0)", "Defaulted (1)"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(y_test, y_pred_logistic, "Logistic Regression")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
