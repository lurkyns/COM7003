import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Loads the processed dataset
df = pd.read_csv("processed_credit_risk_dataset.csv")

# Defines features (X) and target variable (y)
X = df.drop(columns=["loan_status"])  # Features
y = df["loan_status"]  # Target (0 = Fully Paid, 1 = Defaulted)

# Splits into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Input SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Saves the training and testing data
X_train_resampled.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train_resampled.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Training and testing datasets saved successfully with SMOTE applied.")
print(f"Original Training Data Size: {X_train.shape[0]}, Resampled Training Data Size: {X_train_resampled.shape[0]}")