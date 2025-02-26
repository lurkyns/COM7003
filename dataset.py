import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# To set up the KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Load the dataset
df = pd.read_csv("cleaned_credit_risk_dataset.csv") 

# Features to scale
scale_cols = ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

# Apply StandardScaler
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

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

# Imputation to 'loan_int_rate' using KNN
df[["loan_int_rate"]] = imputer.fit_transform(df[["loan_int_rate"]])

# Confirm missing values are handled
print("\nMissing Values After KNN Imputation:\n", df.isnull().sum())

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

# Drop columns that are irrelevant or highly correlated
df = df.drop(columns=["loan_grade"])  # Loan grade is often redundant with loan amount & interest rate

# Turn category data into numbers using one-hot encoding
df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent", "cb_person_default_on_file"], drop_first=True)

# Save the processed dataset
df.to_csv("processed_credit_risk_dataset.csv", index=False)
print("Processed dataset saved successfully.")

# Developing new, significant variables using feature engineering

# How much of a person's income is taken up by the loan?
df["debt_to_income"] = df["loan_int_rate"] / df["loan_amnt2]"]

#Measures affordability
df["loan_to_income_ratio"] = df["loan_amnt"] / df["person_income"]

#Bucket employment length into categories
df["emp_length_category"] = pd.cut(df["person_emp_length"], bins=[0, 2, 5, 10, 20], 
 labels=["0-2 years", "3-5 years", "6-10 years", "10+ years"], 
 include_lowest=True)

print("/New features added successfully!")
print(df.head())
