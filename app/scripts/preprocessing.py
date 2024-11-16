"""
Script Name: preprocessing.py
Description: Handles data preprocessing and feature engineering for the California Housing dataset. Includes handling 
             missing values, feature scaling, interaction feature creation, one-hot encoding, and geospatial feature generation.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def handle_missing_values(df):
    """
    Remove rows with missing values.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with missing values removed.
    """
    try:
        logging.info("Handling missing values...")
        df = df.dropna()
        logging.info(f"Remaining rows after dropping missing values: {df.shape[0]}")
        return df
    except Exception as e:
        logging.error(f"Error in handle_missing_values: {e}")
        raise


def create_interaction_features(df):
    """
    Create interaction features like rooms per household and bedrooms per room.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with new interaction features added.
    """
    try:
        logging.info("Creating interaction features...")
        df["rooms_per_household"] = df["total_rooms"] / df["households"].replace(
            0, np.nan
        )
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"].replace(
            0, np.nan
        )
        df["population_per_household"] = df["population"] / df["households"].replace(
            0, np.nan
        )
        logging.info("Interaction features created.")
        return df
    except Exception as e:
        logging.error(f"Error in create_interaction_features: {e}")
        raise


def bin_income(df):
    """
    Bin the median income column into categories.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with a new income bin column added.
    """
    try:
        logging.info("Binning income column...")
        df["income_bin"] = pd.cut(
            df["median_income"],
            bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        ).astype(int)
        logging.info("Income column binned.")
        return df
    except Exception as e:
        logging.error(f"Error in bin_income: {e}")
        raise


def apply_log_transformation(df):
    """
    Apply log transformation to reduce skewness in specified columns.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with log-transformed columns.
    """
    try:
        logging.info("Applying log transformation...")
        df["log_total_rooms"] = np.log(df["total_rooms"] + 1)
        df["log_total_bedrooms"] = np.log(df["total_bedrooms"] + 1)
        logging.info("Log transformation applied.")
        return df
    except Exception as e:
        logging.error(f"Error in apply_log_transformation: {e}")
        raise


def one_hot_encode(df):
    """
    Perform one-hot encoding for the 'ocean_proximity' categorical feature.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with one-hot encoded categorical features.
    """
    try:
        logging.info("One-hot encoding 'ocean_proximity'...")
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
        logging.info("One-hot encoding done.")
        return df
    except Exception as e:
        logging.error(f"Error in one_hot_encode: {e}")
        raise


def scale_features(df, numerical_columns):
    """
    Scale numerical features using StandardScaler.

    Parameters:
    - df (pd.DataFrame): Input dataset.
    - numerical_columns (list): List of numerical column names to scale.

    Returns:
    - pd.DataFrame: Dataset with scaled numerical features.
    """
    try:
        logging.info("Scaling features...")
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        logging.info("Features scaled.")
        return df
    except Exception as e:
        logging.error(f"Error in scale_features: {e}")
        raise


def create_geospatial_features(df):
    """
    Create geospatial features like distance from the ocean.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Dataset with new geospatial features added.
    """
    try:
        logging.info("Creating geospatial features...")
        df["distance_from_ocean"] = np.sqrt(df["latitude"] ** 2 + df["longitude"] ** 2)
        logging.info("Geospatial features created.")
        return df
    except Exception as e:
        logging.error(f"Error in create_geospatial_features: {e}")
        raise


def feature_engineering(df):
    """
    Perform all feature engineering steps sequentially.

    Parameters:
    - df (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Processed dataset with all feature engineering steps applied.
    """
    logging.info("Starting feature engineering...")
    try:
        df = handle_missing_values(df)
        df = create_interaction_features(df)
        df = bin_income(df)
        df = apply_log_transformation(df)

        numerical_columns = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        df = scale_features(df, numerical_columns)
        df = create_geospatial_features(df)
        df = one_hot_encode(df)

        logging.info(f"Feature engineering completed. Data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise
