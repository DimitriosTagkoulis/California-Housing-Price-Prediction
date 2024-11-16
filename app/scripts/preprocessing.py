import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define all feature engineering functions as discussed earlier
def handle_missing_values(df):
    try:
        # Handle missing values
        logging.info("Handling missing values...")
        df = df.dropna()
        logging.info(f"Remaining rows after dropping missing values: {df.shape[0]}")
        return df
    except Exception as e:
        logging.error(f"Error in handle_missing_values: {e}")
        raise

def create_interaction_features(df):
    try:
        # Create interaction features
        logging.info("Creating interaction features...")
        df.loc[:, 'rooms_per_household'] = df['total_rooms'] / df['households'].replace(0, np.nan)
        df.loc[:, 'bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms'].replace(0, np.nan)
        df.loc[:, 'population_per_household'] = df['population'] / df['households'].replace(0, np.nan)
        logging.info("Interaction features created.")
        return df
    except Exception as e:
        logging.error(f"Error in create_interaction_features: {e}")
        raise

def bin_income(df):
    try:
        # Bin the income column into categories
        logging.info("Binning income column...")
        df.loc[:, 'income_bin'] = pd.cut(df['median_income'], 
                                         bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                         labels=[1, 2, 3, 4, 5]).astype(int)
        logging.info("Income column binned.")
        return df
    except Exception as e:
        logging.error(f"Error in bin_income: {e}")
        raise

def apply_log_transformation(df):
    try:
        # Apply log transformation to highly skewed columns
        logging.info("Applying log transformation...")
        df.loc[:, 'log_total_rooms'] = np.log(df['total_rooms'] + 1)
        df.loc[:, 'log_total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
        logging.info("Log transformation applied.")
        return df
    except Exception as e:
        logging.error(f"Error in apply_log_transformation: {e}")
        raise

def one_hot_encode(df):
    try:
        # One-hot encode categorical features
        logging.info("One-hot encoding 'ocean_proximity'...")
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
        logging.info("One-hot encoding done.")
        return df
    except Exception as e:
        logging.error(f"Error in one_hot_encode: {e}")
        raise

def scale_features(df, numerical_columns):
    try:
        # Scale numerical features using StandardScaler
        logging.info("Scaling features...")
        scaler = StandardScaler()
        df.loc[:, numerical_columns] = scaler.fit_transform(df[numerical_columns])
        logging.info("Features scaled.")
        return df
    except Exception as e:
        logging.error(f"Error in scale_features: {e}")
        raise

def create_geospatial_features(df):
    try:
        # Create geospatial features (distance from some location, for example)
        logging.info("Creating geospatial features...")
        df.loc[:, 'distance_from_ocean'] = np.sqrt(df['latitude'] ** 2 + df['longitude'] ** 2)
        logging.info("Geospatial features created.")
        return df
    except Exception as e:
        logging.error(f"Error in create_geospatial_features: {e}")
        raise

def feature_engineering(df):
    logging.info("Starting feature engineering...")
    
    # Combine all feature engineering steps
    try:
        df = handle_missing_values(df)
        df = create_interaction_features(df)
        df = bin_income(df)
        df = apply_log_transformation(df)

        numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                             'population', 'households', 'median_income']
        df = scale_features(df, numerical_columns)
        df = create_geospatial_features(df)
        df = one_hot_encode(df)
        
        logging.info(f"Feature engineering completed. Data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise
