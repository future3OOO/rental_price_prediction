# view_model.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def find_latest_model_directory(prefix='analysis_plots_'):
    """Find the most recent analysis directory based on creation time."""
    analysis_dirs = [d for d in os.listdir() if d.startswith(prefix) and os.path.isdir(d)]
    if not analysis_dirs:
        print(f"No directories found with prefix '{prefix}'. Ensure that analysis has been run.")
        exit()
    latest_dir = max(analysis_dirs, key=lambda x: os.path.getmtime(x))
    print(f"Using model from the most recent analysis directory: {latest_dir}")
    return latest_dir

def load_trained_model(model_path):
    """Load the trained machine learning model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            trained_model = pickle.load(file)
        print(f"Model loaded successfully from '{model_path}'")
        return trained_model
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

def main():
    # Find the latest model directory
    latest_dir = find_latest_model_directory()
    model_path = os.path.join(latest_dir, 'rental_price_model.pkl')
    trained_model = load_trained_model(model_path)

    print("\nModel Pipeline Steps:")
    if isinstance(trained_model, Pipeline):
        for name, step in trained_model.named_steps.items():
            print(f"\n{name}:\n{step}")

        # Access the regressor
        if 'regressor' in trained_model.named_steps:
            xgb_model = trained_model.named_steps['regressor']
            print("\nFeature Importances:")

            # Create a sample DataFrame to extract feature names
            numeric_features = ['Living Space (m²)', 'Bath', 'Bed', 'Car', 'Land Size (m²)', 'Capital Value']
            categorical_features = ['Suburb']
            date_features = ['Last Rental Date']
            sample_data = pd.DataFrame(columns=numeric_features + categorical_features + date_features)

            # Preprocess and engineer features
            X_processed = trained_model.named_steps['preprocessor'].transform(sample_data)
            X_engineered = trained_model.named_steps['feature_engineering'].transform(X_processed)
            X_encoded = trained_model.named_steps['encoder'].transform(X_engineered)
            feature_names = X_encoded.columns.tolist()

            # Get feature importances
            importances = xgb_model.feature_importances_

            if importances is not None and len(importances) == len(feature_names):
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                print("\nTop Feature Importances:")
                print(feature_importance_df.to_string(index=False))
            else:
                print("Feature importances are not available or do not match the number of features.")
                print(f"Number of features: {len(feature_names)}")
                print(f"Number of importance values: {len(importances) if importances is not None else 'None'}")
        else:
            print("Regressor not found in the pipeline.")
    else:
        print("The loaded model is not a scikit-learn Pipeline.")
        print(f"Type of loaded model: {type(trained_model)}")
        exit()

    # Print additional model information
    print("\nAdditional Model Information:")
    if hasattr(xgb_model, 'get_params'):
        params = xgb_model.get_params()
        for param, value in params.items():
            print(f"{param}: {value}")

if __name__ == "__main__":
    main()
