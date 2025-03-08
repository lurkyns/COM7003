import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import shap

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

best_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
best_model.load_model("best_xgboost_model.json")

feature_importance = best_model.feature_importances_
features = np.array(X_train.columns)

sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.scatter(feature_importance[sorted_idx], features[sorted_idx], color="blue", alpha=0.7)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance (Dot Plot)")
plt.grid(axis="x", linestyle="--", alpha=0.5)  # Add grid for better readability
plt.show()

# SHAP implementation
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

# Generate SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

plt.savefig("shap_summary_plot.png")
print("SHAP analysis completed and saved as 'shap_summary_plot.png'")