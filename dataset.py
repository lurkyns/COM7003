import pandas as pd

# To load the dataset
df = pd.read_csv("credit_risk_dataset.csv")

# To display the first five rows
print(df.head())

# This is to display dataset information
df.info()

# Check for missing values code
print("\nMissing Values:\n", df.isnull().sum())


# To fix the FutureWarning by assigning the changes back to the DataFrame correctly
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())
df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].mean())

# Confirm missing values are handled
print("\nMissing Values After Handling:\n", df.isnull().sum())

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")

# Remove duplicate rows
df = df.drop_duplicates()

# Confirm duplicates are removed
duplicate_rows_after = df.duplicated().sum()
print(f"\nNumber of duplicate rows after removal: {duplicate_rows_after}")

# Save the cleaned dataset
df.to_csv("cleaned_credit_risk_dataset.csv", index=False)

