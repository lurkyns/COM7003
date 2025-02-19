import pandas as pd

# To load the dataset
df = pd.read_csv("credit_risk_dataset.csv")

# To display the first five rows
print(df.head())

# This is to display dataset information
df.info()

# Check for missing values code
print("\nMissing Values:\n", df.isnull().sum())


# Fill missing values
df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
df["loan_int_rate"].fillna(df["loan_int_rate"].mean(), inplace=True)

# Confirm missing values are handled
print("\nMissing Values After Handling:\n", df.isnull().sum())
