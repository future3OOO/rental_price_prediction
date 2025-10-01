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
        self.date_column = date_column
        self.drop_original = drop_original
        self.earliest_date = None
        self.date_format = date_format
        self.new_feature_names_ = None

    def fit(self, X, y=None):
        try:
            if self.date_column not in X.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during fit.")

            X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')
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
            X_ = X.copy()
            if self.date_column not in X_.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during transform.")

            X_[self.date_column] = pd.to_datetime(X_[self.date_column], format=self.date_format, errors='coerce')

            X_['Month'] = X_[self.date_column].dt.month.fillna(0).astype(int)
            X_['Month_sin'] = np.sin(2 * np.pi * X_['Month'] / 12)
            X_['Month_cos'] = np.cos(2 * np.pi * X_['Month'] / 12)

            X_['Time_Index'] = ((X_[self.date_column] - self.earliest_date).dt.days // 30).fillna(0).astype(int)

            self.new_feature_names_ = ['Month_sin', 'Month_cos', 'Time_Index']

            columns_to_drop = [self.date_column, 'Month'] if self.drop_original else ['Month']
            X_.drop(columns=columns_to_drop, errors='ignore', inplace=True)

            return X_
        except Exception as e:
            logging.error(f"Error in DateTransformer.transform: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        return self.new_feature_names_


# ------------------------------------------------------------------
# NEW: Leave-One-Out encoder for high-cardinality 'Suburb'.
# Fits within scikit-learn Pipeline so encoding is applied consistently
# during both training and inference.
# ------------------------------------------------------------------

try:
    from category_encoders import LeaveOneOutEncoder
except ImportError:  # Fallback – allow rest of the codebase to run without hard dep.
    LeaveOneOutEncoder = None  # type: ignore


class SuburbLOOEncoder(BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around category_encoders.LeaveOneOutEncoder.

    – Accepts a single input column ('Suburb') and appends/overwrites
      a numeric column 'Suburb_LOO'.  Original 'Suburb' is retained so
      downstream CatBoost can still treat it as categorical **and** the
      numeric LOO version can live in ``numeric_features``.
    – When category_encoders is missing the transformer degrades to a
      pass-through that creates a constant 0 column, but logs a warning
      so training can continue without hard failure.
    """

    def __init__(self, col: str = "Suburb", sigma: float = 0.1, random_state: int = 42):
        self.col = col
        self.sigma = sigma
        self.random_state = random_state
        self._encoder = None  # Will hold actual LeaveOneOutEncoder instance

    def fit(self, X: pd.DataFrame, y=None):
        if LeaveOneOutEncoder is None:
            logging.warning("category_encoders not installed; SuburbLOOEncoder will generate zeros.")
            return self

        try:
            if self.col not in X.columns:
                raise ValueError(f"Column '{self.col}' not found in input data while fitting SuburbLOOEncoder.")

            self._encoder = LeaveOneOutEncoder(cols=[self.col], sigma=self.sigma, random_state=self.random_state)
            # Encoder expects y to compute target mean – raise if missing
            if y is None:
                raise ValueError("Target 'y' must be provided to fit SuburbLOOEncoder.")
            self._encoder.fit(X[[self.col]], y)
            return self
        except Exception as e:
            logging.error(f"Error fitting SuburbLOOEncoder: {e}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        if LeaveOneOutEncoder is None or self._encoder is None:
            # Graceful degradation: create constant zeros column if encoder not available
            X_out[f"{self.col}_LOO"] = 0.0
            return X_out

        try:
            encoded = self._encoder.transform(X_out[[self.col]])[self.col]
            X_out[f"{self.col}_LOO"] = encoded
            return X_out
        except Exception as e:
            logging.error(f"Error in SuburbLOOEncoder.transform: {e}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        # The transformer adds exactly one new numeric column
        return [f"{self.col}_LOO"]


# ------------------------------------------------------------------
# NEW: SuburbMonthLOOEncoder – captures seasonality within suburbs.
# ------------------------------------------------------------------

class SuburbMonthLOOEncoder(BaseEstimator, TransformerMixin):
    """Target encoding for (Suburb, Month) groups using Leave-One-Out strategy.

    Creates a new numeric column ``SuburbMonth_LOO``.
    Must run BEFORE DateTransformer removes the month information.
    """

    def __init__(self, suburb_col: str = 'Suburb', date_col: str = 'Last Rental Date', sigma: float = 0.1, random_state: int = 42):
        self.suburb_col = suburb_col
        self.date_col = date_col
        self.sigma = sigma
        self.random_state = random_state
        self._encoder = None

    def _make_group(self, X: pd.DataFrame):
        month_series = pd.to_datetime(X[self.date_col], errors='coerce').dt.month.fillna(0).astype(int)
        return X[self.suburb_col].astype(str) + '_' + month_series.astype(str)

    def fit(self, X: pd.DataFrame, y=None):
        if LeaveOneOutEncoder is None:
            logging.warning("category_encoders not installed; SuburbMonthLOOEncoder will output zeros.")
            return self

        if y is None:
            raise ValueError("Target y must be provided for SuburbMonthLOOEncoder.fit().")

        grp = self._make_group(X)
        df_tmp = grp.to_frame(name='SuburbMonth')
        self._encoder = LeaveOneOutEncoder(cols=['SuburbMonth'], sigma=self.sigma, random_state=self.random_state)
        self._encoder.fit(df_tmp, y)
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        if LeaveOneOutEncoder is None or self._encoder is None:
            X_out['SuburbMonth_LOO'] = 0.0
            return X_out

        grp = self._make_group(X_out)
        df_tmp = grp.to_frame(name='SuburbMonth')
        vals = self._encoder.transform(df_tmp)['SuburbMonth']
        X_out['SuburbMonth_LOO'] = vals
        return X_out

    def get_feature_names_out(self, input_features=None):
        return ['SuburbMonth_LOO']
