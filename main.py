import logging
import traceback
import sys
import sklearn
import pandas as pd
import os

from config import (
    RAW_DATA_PATH, PLOT_DIR, MODEL_DIR
)
from data_cleaning import data_cleaning
from outlier_removal import sophisticated_outlier_removal  # Updated function signature
from data_preparation import prepare_data
from eda import perform_eda
from model_training import train_model, make_prediction_with_range
from reproducibility import reproducibility_guidelines
from plotting import plot_average_price_per_month, plot_cumulative_change

def run_full_analysis():
    try:
        logging.info("Starting full analysis workflow...")

        # 1) Load data
        logging.info("Loading data...")
        data = pd.read_csv(RAW_DATA_PATH)
        if data is None or data.empty:
            raise ValueError("Data is empty.")
        logging.info(f"Data shape after loading: {data.shape}")

        # 2) Clean data
        logging.info("Cleaning data...")
        data = data_cleaning(data)
        logging.info(f"Data shape after cleaning: {data.shape}")

        # 3) Remove outliers
        logging.info("Removing outliers...")
        numeric_features_for_outlier_removal = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Capital Value', 'Last Rental Price'
        ]
        data, num_outliers = sophisticated_outlier_removal(
            data=data,
            property_type_column='Bed',  # Matches revised function signature
            numeric_cols=numeric_features_for_outlier_removal,
            output_dir=PLOT_DIR
        )
        logging.info(f"Removed {num_outliers} outliers. Data shape: {data.shape}")

        # 4) Prepare data
        logging.info("Preparing data...")
        data = prepare_data(data, PLOT_DIR)
        logging.info(f"Data shape after preparation: {data.shape}")

        # Check if we can plot monthly averages/cumulative changes
        if 'Last Rental Date' in data.columns and 'Last Rental Price' in data.columns:
            date_column = 'Last Rental Date'
            price_column = 'Last Rental Price'
            if not data[date_column].isnull().any() and not data[price_column].isnull().any():
                plot_average_price_per_month(data, date_column, price_column, PLOT_DIR)
                plot_cumulative_change(data, date_column, price_column, PLOT_DIR)

        # 5) Validate features
        numeric_features = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Land Value', 'Capital Value',
            'Percentage_Diff', 'VWAP_3M'
        ]
        categorical_features = ['Active Listing', 'Suburb', 'Bed']

        required_columns = numeric_features + categorical_features + ['Last Rental Date', 'Last Rental Price']
        missing_columns = [c for c in required_columns if c not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # 6) Exploratory Data Analysis
        logging.info("Performing EDA...")
        perform_eda(data, PLOT_DIR)

        # 7) Train the model
        logging.info("Training the model...")
        result = train_model(data)
        full_pipeline = result['full_pipeline']
        model = result['model']
        metadata = result['metadata']
        date_column = result['date_column']

        if full_pipeline is None or model is None:
            raise ValueError("Model training failed.")
        logging.info("Model trained successfully.")

        # 8) Prepare a test instance to get a sample prediction
        test_features = numeric_features + categorical_features + ['Last Rental Date']
        missing_test_features = [f for f in test_features if f not in data.columns]
        if missing_test_features:
            raise ValueError(f"Missing test features: {missing_test_features}")

        X_test = data[test_features].iloc[-1:]
        y_test_actual = data['Last Rental Price'].iloc[-1]

        mae = metadata['MAE']
        mape = metadata['MAPE']
        logging.info(f"MAE: {mae}, MAPE: {mape}")

        y_pred, lower, upper = make_prediction_with_range(full_pipeline, X_test, mae)
        logging.info("Prediction made successfully.")

        print(f"Predicted price: ${y_pred:.2f}")
        print(f"Expected range: ${lower:.2f} to ${upper:.2f}")
        print(f"Actual price: ${y_test_actual:.2f}")

        # 9) Reproducibility guidelines
        logging.info("Generating reproducibility guidelines...")
        reproducibility_guidelines(data, PLOT_DIR)
        logging.info("Reproducibility guidelines generated successfully.")

        logging.info("Full analysis complete.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    finally:
        logging.info("Analysis process finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("main.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Using Python executable: {sys.executable}")
    logging.info(f"scikit-learn version: {sklearn.__version__}")

    run_full_analysis()
