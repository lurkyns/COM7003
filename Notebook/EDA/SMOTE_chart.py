import pandas as pd
import matplotlib.pyplot as plt

df_before = pd.read_csv("processed_credit_risk_dataset.csv")
class_counts_before = df_before["loan_status"].value_counts()

df_after = pd.read_csv("y_train.csv")  # y_train contains SMOTE-applied labels
class_counts_after = df_after["loan_status"].value_counts()


labels = ["Fully Paid (0)", "Defaulted (1)"]

# Plot class distribution before SMOTE 
plt.figure(figsize=(6, 6))
plt.pie(class_counts_before, labels=labels, autopct="%1.1f%%", colors=["blue", "red"], startangle=90)
plt.title("Class Distribution Before SMOTE")
plt.show()

# Plot class distribution after SMOTE
plt.figure(figsize=(6, 6))
plt.pie(class_counts_after, labels=labels, autopct="%1.1f%%", colors=["blue", "red"], startangle=90)
plt.title("Class Distribution After SMOTE")
plt.show()
