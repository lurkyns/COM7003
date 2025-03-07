import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned dataset

df_cleaned = pd.read_csv("cleaned_credit_risk_dataset.csv")
missing_values_after = df_cleaned.isnull().sum()
print(missing_values_after)

#To load original dataset

df_raw = pd.read_csv("credit_risk_dataset.csv")
missing_values_before = df_raw.isnull().sum()
print(missing_values_before)

# Filter only features with missing values
missing_values_before = missing_values_before[missing_values_before > 0]
missing_values_after = missing_values_after[missing_values_before.index]  # Keep same order

# Prepare data for plotting
categories = missing_values_before.index
missing_before = missing_values_before.values
missing_after = missing_values_after.values

x = np.arange(len(categories))  # X-axis positions

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, missing_before, width=0.4, label="Before Cleaning", color="red")
plt.bar(x + 0.2, missing_after, width=0.4, label="After Cleaning", color="green")

# Formatting
plt.xticks(x, categories, rotation=90)
plt.ylabel("Number of Missing Values")
plt.title("Missing Values Before & After Cleaning")
plt.legend()
plt.show()

# Set a professional-style theme

sns.set_style("whitegrid")
custom_palette = ["#4C72B0", "#55A868"]  # Muted blue and green

#Loan Default Rate by Home Ownership (Stacked Bar Chart)

home_ownership_status = df.groupby(["person_home_ownership", "loan_status"]).size().unstack()
home_ownership_status.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Greens")

plt.title("Loan Default Rate by Home Ownership", fontsize=14)
plt.xlabel("Home Ownership Status", fontsize=10)
plt.ylabel("Number of Loans", fontsize=10)
plt.legend(["Fully Paid (0)", "Defaulted (1)"], title="Loan Status")
plt.xticks(rotation=45)
plt.show()


#Default Rate by Income Group (Binned Bar Chart)

df["income_group"] = pd.cut(df["person_income"], bins=[0, 25000, 50000, 75000, 100000, 200000], 
labels=["0-25K", "25K-50K", "50K-75K", "75K-100K", "100K+"])

income_default_rates = df.groupby("income_group")["loan_status"].mean()  # Default rate per income bin
plt.figure(figsize=(8, 5))
sns.barplot(x=income_default_rates.index, y=income_default_rates.values, palette=custom_palette)
plt.title("Default Rate by Income Group", fontsize=14)
plt.xlabel("Income Group", fontsize=12)
plt.ylabel("Default Rate (Proportion of Loans Defaulted)", fontsize=12)
plt.show()

#Correlation Heatmap (Feature Relationships)

# Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=["int64", "float64"])

# Create correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_cols.corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Key Features", fontsize=14)
plt.show()

# Load the processed dataset before SMOTE
df = pd.read_csv("processed_credit_risk_dataset.csv")

class_counts_before = df["loan_status"].value_counts()
print(class_counts_before)

plt.figure(figsize=(6, 4))
class_counts_before.plot(kind="bar", color=["blue", "red"])
plt.xticks(ticks=[0, 1], labels=["Fully Paid (0)", "Defaulted (1)"], rotation=0)
plt.ylabel("Count")
plt.title("Class Distribution Before SMOTE")
plt.show()
