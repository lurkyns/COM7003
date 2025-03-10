import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# raw dataset FIRST
df = pd.read_csv("credit_risk_dataset.csv")

# handle missing values before applying transformations
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

# KNN Imputer for 'loan_int_rate'
imputer = KNNImputer(n_neighbors=5)
df[["loan_int_rate"]] = imputer.fit_transform(df[["loan_int_rate"]])

# remove duplicate rows
df = df.drop_duplicates()

# save the cleaned dataset
df.to_csv("cleaned_credit_risk_dataset.csv", index=False)
print("Cleaned dataset saved successfully.")

# load the cleaned dataset for further processing
df_cleaned = pd.read_csv("cleaned_credit_risk_dataset.csv")

# features to scale
scale_cols = ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

# apply StandardScaler
scaler = StandardScaler()
df_cleaned[scale_cols] = scaler.fit_transform(df_cleaned[scale_cols])

# to drop irrelevant columns
df_cleaned = df_cleaned.drop(columns=["loan_grade"])

# one-hot encode categorical variables
df_cleaned = pd.get_dummies(df_cleaned, columns=["person_home_ownership", "loan_intent", "cb_person_default_on_file"], drop_first=True)

# save the processed dataset
df_cleaned.to_csv("processed_credit_risk_dataset.csv", index=False)
print("Processed dataset saved successfully.")

# feature engineering
df_cleaned["debt_to_income"] = df_cleaned["loan_amnt"] / (df_cleaned["person_income"] + 1)
df_cleaned["loan_to_income_ratio"] = df_cleaned["loan_amnt"] / df_cleaned["person_income"]
df_cleaned["int_rate_ratio"] = df_cleaned["loan_int_rate"] / df_cleaned["loan_amnt"]

# categorise employment length
df_cleaned["emp_length_category"] = pd.cut(df_cleaned["person_emp_length"], bins=[0, 2, 5, 10, 20], 
 labels=["0-2 years", "3-5 years", "6-10 years", "10+ years"], include_lowest=True)

# saving the final dataset
df_cleaned.to_csv("final_credit_risk_dataset.csv", index=False)
print("Final dataset saved successfully.")