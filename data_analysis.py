import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the Excel file
file_path = "quarterly_rates_aggregates.xlsx"
xls = pd.ExcelFile(file_path)

# Load the "Quarterly estimates" sheet
df = pd.read_excel(xls, sheet_name="Sheet1")

# Convert Date column to datetime format
df.rename(columns={"Date (Quarter)": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

# Basic summary statistics
summary = df.describe()
print("Summary Statistics:")
print(summary)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Time Series Visualization
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Policy rate_Avg"], label="Policy Rate (Avg)", marker='o')
plt.plot(df["Date"], df["Deposit rate_Avg"], label="Deposit Rate (Avg)", marker='x')
plt.plot(df["Date"], df["Lending rate_Avg"], label="Lending Rate (Avg)", marker='s')
plt.xlabel("Date")
plt.ylabel("Rate (%)")
plt.title("Interest Rates Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Financial Variables")
plt.show()

# Distribution Analysis
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.hist(figsize=(15, 12), bins=20)
plt.suptitle("Histograms of Financial Variables", fontsize=16)
plt.show()

# Seasonality and Trend Analysis for multiple interest rates
for rate in ["Policy rate_Avg", "Deposit rate_Avg", "Lending rate_Avg"]:
    print(f"Seasonal Decomposition of {rate}")
    decompose_result = seasonal_decompose(df.set_index("Date")[rate].dropna(), model="additive", period=4)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(decompose_result.observed, label="Observed")
    plt.legend()
    
    plt.subplot(412)
    plt.plot(decompose_result.trend, label="Trend")
    plt.legend()
    
    plt.subplot(413)
    plt.plot(decompose_result.seasonal, label="Seasonality")
    plt.legend()
    
    plt.subplot(414)
    plt.plot(decompose_result.resid, label="Residuals")
    plt.legend()
    
    plt.suptitle(f"Seasonal Decomposition of {rate}")
    plt.show()
