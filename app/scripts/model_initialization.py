"""
Script Name: model_initialization.py
Description: Handles data preprocessing, clustering, model training, evaluation, and saving the best-performing model. 
             Also includes functionality for visualizing model results.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

import sys
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from app.config.base import BASE_DIR
from app.scripts.modeling_pipeline import (
    train_and_evaluate,
    save_best_model,
    train_and_save_kmeans,
    apply_clustering,
)
from app.scripts.preprocessing import feature_engineering
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_data_path():
    """
    Returns the path to the housing dataset.

    Returns:
    - pathlib.Path: Path to the dataset file.
    """
    return BASE_DIR / "data" / "housing.csv"


def load_and_preprocess_data():
    """
    Load and preprocess the housing dataset, including feature engineering and clustering.

    Returns:
    - pd.DataFrame: Preprocessed dataset with clustering applied.
    """
    try:
        file_path = get_data_path()
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        logger.info("Applying feature engineering...")
        df = feature_engineering(df)
        logger.info("Feature engineering completed. Shape: %s", df.shape)

        logger.info("Training and applying KMeans clustering...")
        train_and_save_kmeans(df)
        df = apply_clustering(df, n_clusters=5)
        logger.info("Clustering applied successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {e}")
        raise


def split_data(df):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - df (pd.DataFrame): Preprocessed dataset.

    Returns:
    - tuple: Training and testing features (X_train, X_test) and targets (y_train, y_test).
    """
    try:
        logger.info("Splitting data into training and testing sets...")
        X = df.drop(columns=["median_house_value"])
        y = df["median_house_value"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(
            f"Data split complete. Training data: {X_train.shape}, Testing data: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates models using the provided data.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training targets.
    - y_test (pd.Series): Testing targets.

    Returns:
    - dict: Results of model evaluation.
    """
    try:
        logger.info("Training and evaluating models...")
        model_results = train_and_evaluate(X_train, X_test, y_train, y_test)
        logger.info("Model training and evaluation complete.")
        return model_results
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise


def save_and_validate_best_model(model_results):
    """
    Saves the best-performing model based on evaluation metrics.

    Parameters:
    - model_results (dict): Results of model evaluation.
    """
    try:
        logger.info("Saving the best model...")
        save_best_model(model_results)
        logger.info("Best model saved successfully.")
    except Exception as e:
        logger.error(f"Error during model saving: {e}")
        raise


def plot_model_results(model_results):
    """
    Visualizes model evaluation metrics (RMSE, MAPE, MASE) using bar plots.

    Parameters:
    - model_results (dict): Results of model evaluation.
    """
    try:
        logger.info("Plotting model results...")
        results_df = pd.DataFrame(model_results).T
        results_df.reset_index(inplace=True)
        results_df.columns = ["Model", "Algorithm", "RMSE", "MAPE", "MASE"]

        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ["RMSE", "MAPE", "MASE"]

        for idx, metric in enumerate(metrics):
            sns.barplot(x="Model", y=metric, data=results_df, ax=axes[idx])
            axes[idx].set_title(f"{metric} for each Model")
            axes[idx].set_xlabel("Model")
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
        logger.info("Model results plotted successfully.")
    except Exception as e:
        logger.error(f"Error during result plotting: {e}")
        raise


if __name__ == "__main__":
    try:
        df = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = split_data(df)
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        save_and_validate_best_model(model_results)
        plot_model_results(model_results)
    except Exception as e:
        logger.error(f"Fatal error in model initialization script: {e}")
