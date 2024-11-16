"""
Script Name: EDA.py
Description: Performs exploratory data analysis (EDA) on the California Housing Prices dataset, including visualizations, 
             statistical summaries, and outlier detection.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


def load_dataset(file_path):
    """
    Load the dataset from a specified file path.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def display_dataset_overview(df):
    """
    Display basic information about the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset to analyze.
    """
    print("Shape of the dataset:", df.shape)
    print("Columns in the dataset:", df.columns)
    print("\nFirst 5 rows of the dataset:\n", df.head())
    print("\nDescriptive statistics of the dataset:\n", df.describe())
    print("\nData types of each column:\n", df.dtypes)


def check_missing_values(df):
    """
    Check and display missing values in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset to analyze.
    """
    missing_values = df.isnull().sum()
    print("\nMissing values per feature:\n", missing_values[missing_values > 0])


def encode_categorical_features(df, column):
    """
    Perform one-hot encoding on a specified categorical column.

    Parameters:
    - df (pd.DataFrame): The dataset.
    - column (str): The name of the categorical column to encode.

    Returns:
    - pd.DataFrame: Dataset with one-hot encoding applied.
    """
    return pd.get_dummies(df, columns=[column], drop_first=True)


def plot_histograms(df):
    """
    Plot histograms for all numeric features in the dataset.

    Parameters:
    - df (pd.DataFrame): The dataset to visualize.
    """
    df.hist(figsize=(15, 10), bins=20, color="skyblue", edgecolor="black")
    plt.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot a heatmap for the correlation matrix of numeric columns.

    Parameters:
    - df (pd.DataFrame): The dataset to analyze.
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()
    print("\nCorrelation Matrix:\n", corr)


def detect_outliers(df):
    """
    Detect outliers using Z-scores.

    Parameters:
    - df (pd.DataFrame): The dataset to analyze.

    Returns:
    - int: Number of rows with outliers.
    """
    z_scores = np.abs(df.select_dtypes(include=[np.number]).apply(zscore))
    outliers = (z_scores > 3).all(axis=1)
    print(f"\nNumber of rows with outliers: {outliers.sum()}")
    return outliers


def visualize_feature_distributions(df):
    """
    Generate visualizations for key features.

    Parameters:
    - df (pd.DataFrame): The dataset to analyze.
    """
    # Boxplot for 'median_house_value'
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="median_house_value", color="lightgreen")
    plt.title("Boxplot of Median House Value", fontsize=16)
    plt.show()

    # Distribution of 'median_income'
    plt.figure(figsize=(8, 6))
    sns.histplot(df["median_income"], kde=True, color="purple")
    plt.title("Distribution of Median Income", fontsize=16)
    plt.show()

    # Scatter plot of 'households' vs 'median_house_value'
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="households", y="median_house_value", color="red")
    plt.title("Households vs Median House Value", fontsize=16)
    plt.show()


def visualize_categorical_feature(df, feature_prefix):
    """
    Visualize the distribution of a categorical feature that has been one-hot encoded.

    Parameters:
    - df (pd.DataFrame): The dataset.
    - feature_prefix (str): Prefix of the one-hot encoded features (e.g., 'ocean_proximity').
    """
    df_filtered = df.filter(like=feature_prefix)
    feature_sums = df_filtered.sum()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_sums.index, y=feature_sums.values, palette="Set2")
    plt.title(f"Distribution of {feature_prefix} Categories", fontsize=16)
    plt.ylabel("Count", fontsize=14)
    plt.show()


def main():
    """
    Main function to execute the EDA process.
    """
    file_path = "../../Data/housing.csv"
    df = load_dataset(file_path)

    display_dataset_overview(df)
    check_missing_values(df)

    df = encode_categorical_features(df, "ocean_proximity")
    plot_histograms(df)
    plot_correlation_matrix(df)

    outliers = detect_outliers(df)
    print(f"Outliers Detected: {outliers.sum()}")

    visualize_feature_distributions(df)
    visualize_categorical_feature(df, "ocean_proximity")

    print("EDA Complete")


if __name__ == "__main__":
    main()
