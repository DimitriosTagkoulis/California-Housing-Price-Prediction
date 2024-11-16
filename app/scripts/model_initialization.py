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


# Construct the path to the dataset
def get_data_path():
    """Returns the path to the housing dataset."""
    return BASE_DIR / "data" / "housing.csv"


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    try:
        # Load data
        file_path = get_data_path()
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Apply feature engineering
        logger.info("Applying feature engineering...")
        df = feature_engineering(df)
        logger.info(
            "Feature engineering completed. Data shape after processing: %s", df.shape
        )

        # Train and save KMeans clustering model
        logger.info("Training and applying KMeans clustering...")
        train_and_save_kmeans(df)
        df = apply_clustering(df, n_clusters=5)

        logger.info("Clustering applied successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during data loading and preprocessing: {e}")
        raise


def split_data(df):
    """Split the data into training and testing sets."""
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
    """Train and evaluate models."""
    try:
        logger.info("Training and evaluating models...")
        model_results = train_and_evaluate(X_train, X_test, y_train, y_test)
        logger.info("Model training and evaluation complete.")
        return model_results
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise


def save_and_validate_best_model(model_results):
    """Save the best-performing model and validate its correctness."""
    try:
        logger.info("Saving the best model...")
        save_best_model(model_results)
        logger.info("Best model saved successfully.")
    except Exception as e:
        logger.error(f"Error during model saving: {e}")
        raise


def plot_model_results(model_results):
    """Visualize the performance of the models."""
    try:
        logger.info("Plotting model results...")

        # Convert the results into a pandas DataFrame for easier plotting
        results_df = pd.DataFrame(model_results).T  # Transpose to have models as rows
        results_df.reset_index(inplace=True)

        # Dynamically set column names based on the actual number of metrics
        expected_columns = ["Model"] + list(
            model_results[next(iter(model_results))].keys()
        )
        if len(results_df.columns) != len(expected_columns):
            raise ValueError(
                f"Mismatch in column lengths. Expected: {len(expected_columns)}, "
                f"Got: {len(results_df.columns)}"
            )
        results_df.columns = expected_columns

        # Set up the plotting style
        sns.set(style="whitegrid")

        # Plot metrics for each model
        fig, axes = plt.subplots(
            1, len(expected_columns) - 1, figsize=(18, 6)
        )  # Adjust for number of metrics

        metrics = expected_columns[1:]  # Exclude the "Model" column
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
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data()

        # Step 2: Split data
        X_train, X_test, y_train, y_test = split_data(df)

        # Step 3: Train and evaluate models
        model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Step 4: Save the best model
        save_and_validate_best_model(model_results)

        # Step 5: Plot results
        plot_model_results(model_results)

    except Exception as e:
        logger.error(f"Fatal error in model initialization script: {e}")
