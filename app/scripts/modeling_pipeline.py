"""
Script Name: modeling_pipeline.py
Description: Handles training, evaluation, saving, and loading of regression and clustering models. Also includes 
             hyperparameter tuning and logging using MLflow.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb
import joblib
import json
from pathlib import Path
import os
import datetime
from app.config.base import BASE_DIR

# Ensure CatBoost log directory exists
catboost_log_dir = BASE_DIR / "logs" / "catboost_info"
catboost_log_dir.mkdir(parents=True, exist_ok=True)

# Define paths for models
kmeans_path = BASE_DIR / "models" / "clustering" / "kmeans_model.pkl"

# Define models and hyperparameters
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2),
    "XGBoost": xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50,
        random_state=42,
        tree_method="hist",
        max_bin=128,
        n_jobs=2,
    ),
    "CatBoost": cb.CatBoostRegressor(
        iterations=50,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        task_type="CPU",
        thread_count=2,
        train_dir=str(catboost_log_dir),
    ),
}

PARAM_GRID = {
    "Linear Regression": {"fit_intercept": [True]},
    "Random Forest": {"n_estimators": [50], "max_depth": [10, 20]},
    "XGBoost": {
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 6],
        "subsample": [0.7, 0.8],
    },
    "CatBoost": {"iterations": [50], "learning_rate": [0.05, 0.1], "depth": [6]},
}


def calculate_mase(y_true, y_pred, y_train):
    """
    Calculate the Mean Absolute Scaled Error (MASE).

    Parameters:
    - y_true (np.array): True values.
    - y_pred (np.array): Predicted values.
    - y_train (np.array): Training targets used for scaling.

    Returns:
    - float: MASE value.
    """
    naive_forecast = np.mean(y_train)
    scaled_errors = np.abs(y_true - y_pred) / np.mean(np.abs(y_train - naive_forecast))
    return np.mean(scaled_errors)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models with hyperparameter tuning using GridSearchCV.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training targets.
    - y_test (pd.Series): Testing targets.

    Returns:
    - dict: Results of model evaluation including RMSE, MAPE, and MASE metrics.
    """
    model_results = {}

    for model_name, model in MODELS.items():
        if mlflow.active_run():
            mlflow.end_run()

        try:
            print(f"Training {model_name}...")
            with mlflow.start_run(run_name=f"{model_name}_run"):
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=PARAM_GRID[model_name],
                    cv=2,
                    n_jobs=-1,
                    scoring="neg_root_mean_squared_error",
                    verbose=2,
                )
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                y_pred = best_model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                mase = calculate_mase(y_test, y_pred, y_train)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAPE", mape)
                mlflow.log_metric("MASE", mase)

                model_results[model_name] = {
                    "model": best_model,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "MASE": mase,
                }

                mlflow.sklearn.log_model(best_model, "model")
                print(
                    f"Metrics for {model_name} - RMSE: {rmse}, MAPE: {mape}, MASE: {mase}"
                )

        finally:
            mlflow.end_run()

    return model_results


def save_best_model(model_results, model_dir=BASE_DIR / "models" / "regression"):
    """
    Save the best-performing model based on RMSE.

    Parameters:
    - model_results (dict): Results of model evaluation.
    - model_dir (Path): Directory to save the model.
    """
    best_model_name = min(model_results, key=lambda x: model_results[x]["RMSE"])
    best_model = model_results[best_model_name]["model"]

    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{best_model_name}_{timestamp}.pkl"

    joblib.dump(best_model, model_path)
    print(f"Best model saved at: {model_path}")
    mlflow.log_artifact(str(model_path))


def train_and_save_kmeans(df, n_clusters=5, model_path=kmeans_path):
    """
    Train and save a KMeans clustering model.

    Parameters:
    - df (pd.DataFrame): Dataset for clustering.
    - n_clusters (int): Number of clusters.
    - model_path (Path): Path to save the KMeans model.

    Returns:
    - KMeans: Trained KMeans model.
    """
    numerical_columns = df.select_dtypes(include="number").columns
    df_clustering = df[numerical_columns].drop(
        columns=["median_house_value"], errors="ignore"
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_clustering)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, model_path)
    print(f"KMeans model saved at {model_path}")

    feature_names_path = model_path.parent / f"{model_path.stem}_features.json"
    with open(feature_names_path, "w") as f:
        json.dump(list(df_clustering.columns), f)

    return kmeans


def load_model(model_dir):
    """
    Load a saved model from a specified directory.

    Parameters:
    - model_dir (Path): Directory containing the saved model.

    Returns:
    - model: Loaded model.
    """
    model_dir = Path(model_dir)
    model_files = [f for f in model_dir.iterdir() if f.suffix in [".pkl", ".txt"]]

    if len(model_files) != 1:
        raise ValueError(
            f"Expected one model file in {model_dir}, but found {len(model_files)}."
        )

    model_file = model_files[0]
    if model_file.suffix == ".pkl":
        return joblib.load(model_file)
    elif model_file.suffix == ".txt":
        return xgb.Booster(model_file=str(model_file))
    else:
        raise ValueError(f"Unsupported model format: {model_file}")
