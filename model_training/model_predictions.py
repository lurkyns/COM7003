import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train.values.ravel())
y_pred_logistic = logistic_model.predict(X_test)
pd.DataFrame(y_pred_logistic, columns=["loan_status"]).to_csv("y_pred_logistic.csv", index=False)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=88)
rf_model.fit(X_train, y_train.values.ravel())
y_pred_rf = rf_model.predict(X_test)
pd.DataFrame(y_pred_rf, columns=["loan_status"]).to_csv("y_pred_rf.csv", index=False)

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train.values.ravel())
y_pred_xgb = xgb_model.predict(X_test)
pd.DataFrame(y_pred_xgb, columns=["loan_status"]).to_csv("y_pred_xgb.csv", index=False)

print("Predictions for all models saved successfully!")
