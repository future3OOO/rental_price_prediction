# transformers.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
import traceback

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts cyclical date features from a date column.
    """
    def __init__(self, date_column='Last Rental Date', drop_original=True, date_format=None):
        """
        Initializes the DateTransformer.

        Parameters:
            date_column (str): The name of the date column to transform.
            drop_original (bool): Whether to drop the original date column after transformation.
            date_format (str): The format of the date strings. Example: '%Y-%m-%d'
        """
        self.date_column = date_column
        self.drop_original = drop_original
        self.earliest_date = None  # For calculating Time_Index
        self.date_format = date_format
        self.new_feature_names_ = None  # To store the names of the new features

    def fit(self, X, y=None):
        try:
            # Ensure date column exists
            if self.date_column not in X.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during fit.")

            # Convert to datetime and handle errors
            X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')

            # Store the earliest date for Time_Index calculation
            self.earliest_date = X[self.date_column].min()

            if pd.isnull(self.earliest_date):
                raise ValueError(f"All dates in column '{self.date_column}' are NaT or invalid.")

            return self
        except Exception as e:
            logging.error(f"Error in DateTransformer.fit: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X):
        try:
            X = X.copy()

            # Ensure the date column exists
            if self.date_column not in X.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during transform.")

            # Convert to datetime and handle errors
            X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')

            # Extract cyclical month features
            X['Month'] = X[self.date_column].dt.month.fillna(0).astype(int)
            X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
            X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)

            # Create Time_Index (e.g., months since earliest date)
            X['Time_Index'] = ((X[self.date_column] - self.earliest_date).dt.days // 30).fillna(0).astype(int)

            # Store the new feature names for get_feature_names_out
            self.new_feature_names_ = ['Month_sin', 'Month_cos', 'Time_Index']

            # Drop the original date column and 'Month' if configured to do so
            columns_to_drop = [self.date_column, 'Month'] if self.drop_original else ['Month']
            X = X.drop(columns=columns_to_drop, errors='ignore')

            return X
        except Exception as e:
            logging.error(f"Error in DateTransformer.transform: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the features produced by the transformer.

        Parameters:
            input_features (array-like of str): Input feature names. If None, defaults to the date column.

        Returns:
            List[str]: Output feature names.
        """
        return self.new_feature_names_
