# data_loading.py

import pandas as pd
import logging

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig')  # Use 'utf-8-sig' for UTF-8 with BOM
        logging.info(f"Data loaded successfully from {file_path}")
        logging.info(f"Columns in loaded data: {data.columns.tolist()}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found at path: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None
