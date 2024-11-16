import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import datetime
from app.config.base import BASE_DIR
from pathlib import Path


# Define a valid directory for CatBoost logs
BASE_DIR = Path(__file__).resolve().parent.parent.parent
catboost_log_dir = BASE_DIR / "logs" / "catboost_info"

# Ensure the directory exists
if not catboost_log_dir.exists():
    catboost_log_dir.mkdir(parents=True, exist_ok=True)


# Define kmeans path

# kmeans_path = "../models/clustering/kmeans_model.pkl"
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
kmeans_path = str(BASE_DIR / "models" / "clustering" / "kmeans_model.pkl")


# Define Models and Hyperparameters (adjusted for lighter usage)
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
    # "LightGBM": lgb.LGBMRegressor(n_estimators=50,
    #                               random_state=42,
    #                               device_type='cpu',
    #                              num_threads=2),
    "CatBoost": cb.CatBoostRegressor(
        iterations=50,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        task_type="CPU",
        thread_count=2,
        train_dir="..\..\logs\catboost_info",
    ),
}

# Smaller hyperparameter grid to reduce the search space
PARAM_GRID = {
    "Linear Regression": {"fit_intercept": [True]},
    "Random Forest": {"n_estimators": [50], "max_depth": [10, 20]},
    "XGBoost": {
        "learning_rate": [0.01, 0.05],
        "max_depth": [3, 6],
        "subsample": [0.7, 0.8],
    },
    #    "LightGBM": {"num_leaves": [31], "learning_rate": [0.05, 0.1], "n_estimators": [50]},
    "CatBoost": {"iterations": [50], "learning_rate": [0.05, 0.1], "depth": [6]},
}


def calculate_mase(y_true, y_pred, y_train):
    naive_forecast = np.mean(y_train)
    scaled_errors = np.abs(y_true - y_pred) / np.mean(np.abs(y_train - naive_forecast))
    return np.mean(scaled_errors)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    model_results = {}

    for model_name, model in MODELS.items():
        # Ensure any active run is ended before starting a new one
        if mlflow.active_run() is not None:
            mlflow.end_run()

        try:
            print(f"Training {model_name}...")
            with mlflow.start_run(run_name=f"{model_name}_run"):
                mlflow.log_param("model_name", model_name)

                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=PARAM_GRID[model_name],
                    cv=2,
                    n_jobs=-1,
                    scoring="neg_root_mean_squared_error",
                    verbose=2,
                )
                grid_search.fit(X_train, y_train)

                # Use the best estimator
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_

                mlflow.log_params(best_params)

                print(f"Best hyperparameters for {model_name}: {best_params}")

                # Evaluate the best model
                y_pred = best_model.predict(X_test)

                # Calculate and log metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = mean_absolute_percentage_error(y_test, y_pred)
                mase = calculate_mase(y_test, y_pred, y_train)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAPE", mape)
                mlflow.log_metric("MASE", mase)

                model_results[model_name] = {
                    "model": best_model,  # Save the trained model
                    "RMSE": rmse,
                    "MAPE": mape,
                    "MASE": mase,
                }

                print(
                    f"Metrics for {model_name} - RMSE: {rmse}, MAPE: {mape}, MASE: {mase}"
                )

                # Log the best model
                mlflow.sklearn.log_model(best_model, "model")

                if hasattr(best_model, "feature_importances_"):
                    feature_importances = best_model.feature_importances_
                    for i, col in enumerate(X_train.columns):
                        mlflow.log_metric(f"importance_{col}", feature_importances[i])

        finally:
            # Ensure the MLflow run is ended
            mlflow.end_run()

    return model_results


import joblib
import os
import datetime


def save_best_model(model_results, model_dir=BASE_DIR / "models" / "regression"):
    # Find the model with the best performance (min RMSE)
    best_model_name = min(model_results, key=lambda x: model_results[x]["RMSE"])
    best_model = model_results[best_model_name]["model"]  # Use the trained model

    # Ensure the model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"{best_model_name}_{timestamp}.pkl"

    # Save the trained model
    joblib.dump(best_model, model_path)
    print(f"Best model saved at: {model_path}")

    # Log the model path to MLflow
    mlflow.log_artifact(str(model_path))


def apply_clustering(df, n_clusters=5, model_save_path=kmeans_path):
    numerical_columns = df.select_dtypes(include="number").columns

    # Load the pre-trained KMeans model
    if os.path.exists(model_save_path):
        kmeans = joblib.load(model_save_path)
        print(f"Loaded existing KMeans model from {model_save_path}")
    else:
        raise FileNotFoundError(
            f"No KMeans model found at {model_save_path}. Please train and save the model first."
        )

    # Prepare the data for clustering, excluding the target column from clustering
    df_clustering = df[numerical_columns].drop(
        columns=["median_house_value"], errors="ignore"
    )

    # Assign clusters
    df["cluster"] = kmeans.predict(df_clustering)

    return df


import json


import json


from pathlib import Path
import json


def train_and_save_kmeans(df, n_clusters=5, model_path=kmeans_path):
    # Ensure model_path is a Path object
    model_path = Path(model_path)

    numerical_columns = df.select_dtypes(include="number").columns

    # Ensure the directory exists before saving the model
    model_dir = model_path.parent
    if not model_dir.exists():
        print(f"Creating directory for KMeans model: {model_dir}")
        model_dir.mkdir(parents=True, exist_ok=True)

    # Exclude the target column (e.g., 'median_house_value') from training data
    df_clustering = df[numerical_columns].drop(
        columns=["median_house_value"], errors="ignore"
    )
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_clustering)

    # Save the model
    joblib.dump(kmeans, model_path)
    print(f"KMeans model saved at {model_path}")

    # Save feature names used during training
    feature_names_path = model_path.parent / f"{model_path.stem}_features.json"
    with open(feature_names_path, "w") as f:
        json.dump(list(df_clustering.columns), f)
    print(f"Feature names saved at {feature_names_path}")

    # Log the model with MLflow (optional)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(feature_names_path)

    return kmeans


from pathlib import Path


def load_model(model_dir):
    """
    Load the model from the specified directory.

    Parameters:
    - model_dir (str): Directory containing the saved model files.

    Returns:
    - model: Loaded model (could be LightGBM or other model types).
    """
    model_dir = Path(model_dir)  # Ensure it's a Path object
    if not model_dir.exists():
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")
    print(f"Directory {model_dir} exists.")

    # List all the model files in the directory (we expect only one)
    model_files = [f for f in model_dir.iterdir() if f.suffix in [".pkl", ".txt"]]
    print(f"Found model files: {model_files}")

    if len(model_files) != 1:
        raise ValueError(
            f"Expected exactly one model file in {model_dir}, but found {len(model_files)}."
        )

    # Get the full path of the model file
    model_file = model_files[0]
    print(f"Loading model from {model_file}")

    # Load the model depending on the file type
    try:
        if model_file.suffix == ".pkl":
            # Load non-LightGBM models using joblib
            model = joblib.load(model_file)
            print("Non-LightGBM model loaded successfully.")
        elif model_file.suffix == ".txt":
            # Load LightGBM models using LightGBM's Booster class
            model = lgb.Booster(model_file=str(model_file))
            print("LightGBM model loaded successfully.")
        else:
            raise ValueError(f"Unsupported model format: {model_file}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model from {model_file}. Error: {e}")

    return model
