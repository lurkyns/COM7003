import pandas as pd
import numpy as np

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
y_pred_xgb = pd.read_csv("y_pred_xgb.csv")  # Predictions from XGBoost

# To combine predictions with actual values
df_results = pd.DataFrame({"Actual": y_test.values.ravel(), "Predicted": y_pred_xgb.values.ravel()})
df_results["Income Group"] = X_test["person_income"].apply(lambda x: "Low" if x < X_test["person_income"].median() else "High")

# To calculate default rate per income group
bias_check = df_results.groupby("Income Group")["Predicted"].value_counts(normalize=True).unstack()
print("\nDefault Prediction Rates by Income Group:\n", bias_check)

# To Calculate disparity ratio
disparity_ratio = bias_check.loc["Low", 1] / bias_check.loc["High", 1]
print(f"\nDisparity Ratio (Default Rate Low-Income vs High-Income): {disparity_ratio:.2f}")

bias_check.to_csv("bias_analysis_results.csv")
