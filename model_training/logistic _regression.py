import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Set up and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

# To make predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)  # This must be AFTER y_pred is defined

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

