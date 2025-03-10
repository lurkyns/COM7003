import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#to load Data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

#to load Trained XGBoost Model
best_model = xgb.XGBClassifier()
best_model.load_model("best_xgboost_model.json")  # Load the saved model

#SHAP implementation
explainer = shap.Explainer(best_model, X_train)  # Create SHAP explainer
shap_values = explainer(X_test)  # Get SHAP values for test data

#to generate SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)


plt.savefig("shap_summary_plot.png")

print("SHAP analysis completed and saved as 'shap_summary_plot.png'")