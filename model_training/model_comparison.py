import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy results from previous models
results = {
    "Logistic Regression": 0.7259,  # Replace with actual accuracy
    "Random Forest": 0.9021,  # Replace with actual accuracy
    "XGBoost": 0.9272  # Replace with actual accuracy
}

# Convert to DataFrame
df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.bar(df_results["Model"], df_results["Accuracy"], color=["red", "yellow", "green"])
plt.ylim(0.7, 1.0)  # Keep within the range of 70%-100%
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()
