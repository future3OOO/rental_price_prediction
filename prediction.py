# prediction.py

import logging
import os
import sys
import traceback
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from config import MODEL_DIR, CLEANED_DATA_PATH
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)

def load_full_pipeline_and_metadata():
    try:
        # Paths to the saved files
        full_pipeline_path = os.path.join(MODEL_DIR, 'full_pipeline.joblib')
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')

        # Check if files exist
        if not os.path.exists(full_pipeline_path):
            logging.error(f"Full pipeline file not found at: {full_pipeline_path}")
            sys.exit(1)
        if not os.path.exists(metadata_path):
            logging.error(f"Metadata file not found at: {metadata_path}")
            sys.exit(1)

        # Load the full pipeline
        full_pipeline = joblib.load(full_pipeline_path)
        logging.info(f"Full pipeline loaded successfully from '{full_pipeline_path}'.")

        # Load the metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Metadata loaded successfully from '{metadata_path}'.")

        return full_pipeline, metadata

    except Exception as e:
        logging.error(f"Error loading full pipeline and metadata: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def calculate_vwap(data, bed_type, last_rental_date):
    try:
        # Ensure 'Bed' is of string type
        data['Bed'] = data['Bed'].astype(str)

        # Filter data for the given bed_type and before last_rental_date
        bed_data = data[
            (data['Bed'] == bed_type) &
            (data['Last Rental Date'] < last_rental_date)
        ].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        if bed_data.empty:
            logging.warning(f"No historical data available for VWAP calculation for Bed='{bed_type}' before {last_rental_date.date()}.")
            return None, None

        # Log the number of records
        logging.info(f"Number of historical records for Bed='{bed_type}': {len(bed_data)}")

        # Calculate VWAP_3M
        start_date_3M = last_rental_date - pd.DateOffset(months=3)
        vw_data_3M = bed_data[bed_data['Last Rental Date'] >= start_date_3M]
        if len(vw_data_3M) >= 5:
            VWAP_3M = vw_data_3M['Last Rental Price'].mean()
        else:
            VWAP_3M = bed_data['Last Rental Price'].mean()  # Fallback to all data

        # Calculate VWAP_12M
        start_date_12M = last_rental_date - pd.DateOffset(months=12)
        vw_data_12M = bed_data[bed_data['Last Rental Date'] >= start_date_12M]
        if len(vw_data_12M) >= 20:
            VWAP_12M = vw_data_12M['Last Rental Price'].mean()
        else:
            VWAP_12M = bed_data['Last Rental Price'].mean()  # Fallback to all data

        return VWAP_3M, VWAP_12M

    except Exception as e:
        logging.error(f"Error calculating VWAP: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def predict_rental_price(features: pd.DataFrame, full_pipeline, metadata):
    try:
        # Log initial features
        logging.info(f"Initial features:\n{features}\n")

        # Ensure 'Last Rental Date' is in datetime format
        if 'Last Rental Date' in features.columns and not pd.api.types.is_datetime64_any_dtype(features['Last Rental Date']):
            features['Last Rental Date'] = pd.to_datetime(features['Last Rental Date'])

        # Ensure that categorical features are strings
        categorical_features = metadata['cat_features']
        for col in categorical_features:
            if col in features.columns:
                features[col] = features[col].astype(str)
            else:
                features[col] = 'Unknown'  # Handle missing categorical features

        # Make prediction using the full pipeline
        predicted_price = full_pipeline.predict(features)
        return predicted_price[0]

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def display_feature_importances(full_pipeline):
    try:
        # Extract the model from the pipeline
        model = full_pipeline.named_steps['model']

        # Get feature importances from the model (prettified)
        feature_importances = model.get_feature_importance(prettified=True)

        # Log the feature importances
        logging.info("\nModel Feature Importances:")
        logging.info(feature_importances.to_string(index=False))

    except Exception as e:
        logging.error(f"Error displaying feature importances: {str(e)}")
        logging.error(traceback.format_exc())

def main():
    try:
        # Load the full pipeline and metadata
        full_pipeline, metadata = load_full_pipeline_and_metadata()

        # Optionally display feature importances
        display_feature_importances(full_pipeline)

        # Load historical data for VWAP calculation
        historical_data = pd.read_csv(CLEANED_DATA_PATH)
        historical_data['Last Rental Date'] = pd.to_datetime(historical_data['Last Rental Date'])
        historical_data['Last Rental Price'] = pd.to_numeric(historical_data['Last Rental Price'], errors='coerce')
        # Ensure 'Bed' is of string type
        historical_data['Bed'] = historical_data['Bed'].astype(str)

        # Log the date range and unique 'Bed' values
        logging.info(f"Historical data date range: {historical_data['Last Rental Date'].min()} to {historical_data['Last Rental Date'].max()}")
        logging.info(f"Unique 'Bed' values in historical data: {historical_data['Bed'].unique()}")

        # Define base features
        base_features = {
            'Bed': '4',  # Ensure it's a string
            'Bath': 1,
            'Car': 1,
            'Land Size (sqm)': 760,
            'Floor Size (sqm)': 120,
            'Year Built': 1950,
            'Days on Market': 6,
            'Suburb': 'Papanui, CHRISTCHURCH',
            'Active Listing': 'No',
            'Capital Value': None,  # Will be set in loop
            'Last Rental Date': pd.to_datetime('2024-05-01'),
            'VWAP_3M': None,       # Will be calculated
            'VWAP_12M': None,      # Will be calculated
            'Percentage_Diff': None  # Will be calculated
        }

        # Get dynamic VWAP values
        VWAP_3M, VWAP_12M = calculate_vwap(
            historical_data,
            bed_type=base_features['Bed'],
            last_rental_date=base_features['Last Rental Date']
        )

        if VWAP_3M is None or VWAP_12M is None:
            logging.error("Insufficient historical data to calculate VWAP features.")
            sys.exit(1)

        base_features['VWAP_3M'] = VWAP_3M
        base_features['VWAP_12M'] = VWAP_12M

        # Calculate Percentage_Diff
        base_features['Percentage_Diff'] = ((VWAP_3M - VWAP_12M) / VWAP_12M) * 100

        # Define different Capital Values to test
        capital_values = [
            550000,   # -20%
            645000,   # Original
            1000000   # +20%
        ]

        logging.info("\nTesting different Capital Values:")
        for cv in capital_values:
            logging.info(f"\n{'='*50}")
            logging.info(f"Testing Capital Value: ${cv:,.2f}")
            logging.info(f"{'='*50}")

            # Create a copy of the base features and update 'Capital Value'
            features = base_features.copy()
            features['Capital Value'] = cv

            # Create DataFrame for prediction
            example_features = pd.DataFrame([features])

            # Predict rental price
            predicted_price = predict_rental_price(example_features, full_pipeline, metadata)
            logging.info(f"Final predicted rental price: ${predicted_price:.2f}")
            logging.info(f"{'='*50}\n")

    except Exception as e:
        logging.error(f"An error occurred during the prediction process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
