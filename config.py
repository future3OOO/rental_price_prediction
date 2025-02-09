# config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Data file paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'rental_data.csv')  # Ensure this matches your actual filename
CLEANED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'cleaned_rental_data.csv')

# Plot directory
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Cache directory
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# Model saving directory and path
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rental_price_model.joblib')

# Metadata saving path
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.joblib')

# Optuna study saving path
STUDIES_DIR = os.path.join(BASE_DIR, 'studies')
OPTUNA_STUDY_PATH = os.path.join(STUDIES_DIR, 'optuna_study.pkl')

# Extra directories for models, metadata, and studies
EXTRA_DIRS = [
    MODEL_DIR,
    PLOT_DIR,
    CACHE_DIR,
    METADATA_DIR,
    STUDIES_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR
]

print(f"MODEL_PATH: {MODEL_PATH}")

# Ensure MODEL_DIR exists
os.makedirs(MODEL_DIR, exist_ok=True)
