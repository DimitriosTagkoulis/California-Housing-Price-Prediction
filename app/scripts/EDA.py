# EDA Script for California Housing Prices Dataset
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load dataset
file_path = "..\..\Data\housing.csv"
df = pd.read_csv(file_path)

# Overview of the dataset
print("Shape of the dataset:", df.shape)
print("Columns in the dataset:", df.columns)
print("\nFirst 5 rows of the dataset:\n", df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per feature:\n", missing_values[missing_values > 0])

# Basic statistics summary
print("\nDescriptive statistics of the dataset:\n", df.describe())

# Data types of the columns
print("\nData types of each column:\n", df.dtypes)

# Use one-hot encoding for 'ocean_proximity'
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Histogram of all numeric features
df.hist(figsize=(15, 10), bins=20, color="skyblue", edgecolor="black")
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()

# Display descriptive statistics table
desc_stats = df.describe()
print("\nDescriptive Statistics Table:")
print(desc_stats)

# Correlation matrix heatmap (only for numeric columns)
corr = df.select_dtypes(include=[np.number]).corr()  # Select only numeric columns
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# Display correlation matrix as table
print("\nCorrelation Matrix:")
print(corr)

# Boxplot for distribution of 'median_house_value'
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="median_house_value", color="lightgreen")
plt.title("Boxplot of Median House Value", fontsize=16)
plt.show()

# Distribution of 'median_income' (or any feature of interest)
plt.figure(figsize=(8, 6))
sns.histplot(df["median_income"], kde=True, color="purple")
plt.title("Distribution of Median Income", fontsize=16)
plt.show()

# Scatter plot of 'households' vs 'median_house_value'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="households", y="median_house_value", color="red")
plt.title("Households vs Median House Value", fontsize=16)
plt.show()

# Pairplot for key features
sns.pairplot(
    df[["median_income", "households", "total_rooms", "median_house_value"]],
    diag_kind="kde",
)
plt.suptitle("Pairplot for Key Features", fontsize=16)
plt.show()

# Checking for outliers using z-score (optional)
z_scores = np.abs(
    df.select_dtypes(include=[np.number]).apply(zscore)
)  # Only apply to numeric columns
outliers = (z_scores > 3).all(axis=1)
print(f"\nNumber of rows with outliers: {outliers.sum()}")

# Display number of outliers
print("\nOutliers Calculation (Z-score > 3):")
outlier_table = pd.DataFrame(
    {
        "Row Index": np.where(outliers)[0],
        "Outlier Status": ["Yes" for _ in np.where(outliers)[0]],
    }
)
print(outlier_table)

# Distribution of a categorical feature, e.g., 'ocean_proximity'
# After one-hot encoding, this feature is now several columns (e.g., 'ocean_proximity_NEAR BAY', etc.)
# You can visualize the distribution of these columns instead of the original 'ocean_proximity'
df_ocean_proximity = df.filter(
    like="ocean_proximity"
)  # Select all 'ocean_proximity' columns
df_ocean_proximity_sum = (
    df_ocean_proximity.sum()
)  # Sum of each column to count occurrences

# Barplot for categorical feature distribution (one-hot encoded)
plt.figure(figsize=(8, 6))
sns.barplot(
    x=df_ocean_proximity_sum.index, y=df_ocean_proximity_sum.values, palette="Set2"
)
plt.title("Distribution of Ocean Proximity Categories", fontsize=16)
plt.ylabel("Count", fontsize=14)
plt.show()

# Display the ocean proximity distribution table
print("\nOcean Proximity Distribution Table:")
print(df_ocean_proximity_sum)

# Summary of one-hot encoded 'ocean_proximity' columns
df_ocean_proximity = df.filter(
    like="ocean_proximity"
)  # Select all one-hot encoded 'ocean_proximity' columns
ocean_proximity_summary = (
    df_ocean_proximity.mean()
)  # Calculate the mean of each one-hot encoded column (frequency of each category)

print("\nSummary of One-Hot Encoded Ocean Proximity Columns:")
print(ocean_proximity_summary)

print("Eda Complete")
