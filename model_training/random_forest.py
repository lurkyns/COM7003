import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Set up and train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=88)
model.fit(X_train, y_train.values.ravel())

# Make Predictions
y_pred = model.predict(X_test)

# To evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Random Forest Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)