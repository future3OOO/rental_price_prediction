import pandas as pd
import numpy as np
import logging
import traceback
from config import CLEANED_DATA_PATH  # ensure your config.py has the correct path

def data_cleaning(data):
    """
    Cleans the dataset by:
      - Normalizing suburb strings.
      - Renaming columns to ensure consistent naming of 'sqm'.
      - Parsing numeric columns, including Land/Capital Value columns that have
        $ signs or commas.
      - Converting 'Last Rental Date' to datetime with a chosen format.
      - Excluding properties with bed count > 5.
      - (Optionally) dropping rows missing 'Bed' or leaving them as NaN.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logging.info("Starting data cleaning...")

    try:
        #######################
        # 1. Normalize Suburb
        #######################
        if 'Suburb' in data.columns:
            data['Suburb'] = data['Suburb'].str.strip().str.upper()
            logging.info("Normalized 'Suburb' column.")
        else:
            logging.warning("'Suburb' column not found; skipping normalization.")

        #######################
        # 2. Rename columns
        #######################
        # Ensures consistent naming of 'sqm'
        rename_map = {
            'Land Size (m²)': 'Land Size (sqm)',
            'Floor Size (m²)': 'Floor Size (sqm)'
        }
        data.rename(columns=rename_map, inplace=True)
        logging.info(f"Columns after renaming: {list(data.columns)}")

        ###############################
        # 3. Handle 'Capital Value' & 'Land Value'
        ###############################
        # If they contain $ or commas, remove them:
        def clean_currency_column(df, colname):
            if colname in df.columns:
                df[colname] = (
                    df[colname]
                    .astype(str)
                    .replace(r'[^0-9\.\-]+', '', regex=True)  # remove $, commas, etc.
                )
                df[colname] = pd.to_numeric(df[colname], errors='coerce')
                logging.info(f"Cleaned currency column '{colname}'.")
            else:
                logging.warning(f"Column '{colname}' not found; skipping currency cleanup.")

        clean_currency_column(data, 'Capital Value')
        clean_currency_column(data, 'Land Value')

        #######################
        # 4. Numeric columns
        #######################
        # Adjust list to match your data layout
        numeric_cols = [
            'Bed', 'Bath', 'Car',
            'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Last Rental Price',
            'Days on Market', 'Capital Value', 'Land Value'
        ]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                logging.info(f"Converted '{col}' to numeric. NaN count: {data[col].isnull().sum()}")
            else:
                logging.warning(f"Numeric column '{col}' not found in the data.")

        #######################
        # 5. Fill Missing Numeric
        #######################
        # For numeric columns other than 'Bed', fill with median
        fill_with_median = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Last Rental Price', 'Days on Market',
            'Capital Value', 'Land Value'
        ]
        for col in fill_with_median:
            if col in data.columns:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                logging.info(f"Filled missing '{col}' with median = {median_val:.2f}.")
            else:
                logging.warning(f"Column '{col}' not found for missing-value fill.")

        #######################
        # 6. DO NOT Fill 'Bed' with Median
        #######################
        #  Option A: drop rows that have Bed = NaN
        # data = data.dropna(subset=['Bed'])
        # logging.info("Dropped rows where 'Bed' was missing.")

        #  Option B: leave them as NaN to handle further in pipeline
        # logging.info("Leaving missing 'Bed' as NaN (not filling with median).")

        # For now let's choose Option B, leaving them as NaN:
        # If you'd prefer to drop them, just uncomment Option A above.
        if data['Bed'].isnull().any():
            logging.warning(f"There are {data['Bed'].isnull().sum()} rows with missing 'Bed' values. Not filling them.")

        #######################
        # 7. Parse 'Last Rental Date'
        #######################
        if 'Last Rental Date' in data.columns:
            sample_dates = data['Last Rental Date'].dropna().head(5).tolist()
            logging.info(f"Sample 'Last Rental Date' before parsing: {sample_dates}")

            # Choose correct date format for your data
            date_format = '%d-%b-%y'  # Example: "8/11/2023" might not match this
            # Adjust to match your raw data if needed
            data['Last Rental Date'] = pd.to_datetime(
                data['Last Rental Date'],
                format=date_format,
                errors='coerce'
            )

            valid_dates = data['Last Rental Date'].notnull().sum()
            missing_dates = data['Last Rental Date'].isnull().sum()
            logging.info(f"After parsing, valid 'Last Rental Date': {valid_dates}, missing: {missing_dates}")

            # If missing still exist, fill with median date or drop rows
            if missing_dates > 0:
                date_median = data['Last Rental Date'].median()
                if pd.isnull(date_median):
                    logging.warning("Cannot fill missing 'Last Rental Date'; median is NaT.")
                else:
                    data['Last Rental Date'].fillna(date_median, inplace=True)
                    logging.info(f"Filled missing 'Last Rental Date' with median = {date_median}")
        else:
            logging.warning("'Last Rental Date' column not found in data. Skipping date parsing.")

        #######################
        # 8. Remove bed > 5
        #######################
        if 'Bed' in data.columns:
            prev_shape = data.shape
            data = data[data['Bed'] <= 5]
            new_shape = data.shape
            logging.info(f"Removed rows with Bed > 5. Shape changed from {prev_shape} to {new_shape}.")

        #######################
        # 9. Final Logging
        #######################
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {list(data.columns)}")

        # Save
        data.to_csv(CLEANED_DATA_PATH, index=False)
        logging.info(f"Cleaned data saved to '{CLEANED_DATA_PATH}'.")

        return data

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        logging.error(traceback.format_exc())
        raise

#
# Additional Note on Predicting 'Bed' for missing rows
#
# If you'd like to *predict* Bed for missing rows (rather than drop them or leave them as NaN),
# you could train a small classifier/regressor using other columns (e.g. Land Size, Floor Size, Year Built, etc.)
# This would be done *after* or as part of data cleaning. The approach might be:
#  1) Separate rows that have a known 'Bed'.
#  2) Train a model to predict 'Bed' from your numeric/categorical features.
#  3) Apply that model to rows missing 'Bed', fill them in, and proceed.
#
# This is more advanced but can sometimes boost overall data coverage and performance.
# It's recommended to validate carefully to ensure you don't introduce too much noise.
