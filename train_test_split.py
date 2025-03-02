import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Loads the processed dataset
df = pd.read_csv("processed_credit_risk_dataset.csv")

# Defines features (X) and target variable (y)
X = df.drop(columns=["loan_status"])  # Features
y = df["loan_status"]  # Target (0 = Fully Paid, 1 = Defaulted)

# Splits into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88, stratify=y)

# Fixing missing columns between train and test sets
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=88)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Save the training and testing data
X_train_resampled.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train_resampled.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Print success message and data sizes
print("Training and testing datasets saved successfully with SMOTE applied.")
print(f"Original Training Data Size: {X_train.shape[0]}, Resampled Training Data Size: {X_train_resampled.shape[0]}")
print(f"Test data size: {X_test.shape[0]}")
print("First 5 rows of X_train:\n", X_train.head())
