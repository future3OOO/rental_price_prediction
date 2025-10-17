# transformers.py
# Harden public API: rely solely on native encoders to avoid category-encoders tag drift.
# All estimators exposed here are sklearn-compatible and free of external CE dependencies.
import pandas as pd
import numpy as np
import logging
import traceback
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "DateTransformer",
    "SuburbLOOEncoder",
    "SuburbMonthLOOEncoder",
]

# NOTE: This module must remain category-encoders free; downstream imports rely on these native classes.

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
# Native, version‑agnostic Leave‑One‑Out encoders
# ------------------------------------------------------------------

class _LOOBase(BaseEstimator, TransformerMixin):
    """Utility base for native LOO with additive smoothing."""
    def __init__(self, k: float = 10.0):
        # k is the smoothing weight toward the global mean
        self.k = float(k)
        self.global_mean_: float | None = None
        self.sum_: dict[str, float] | None = None
        self.count_: dict[str, int] | None = None

    @staticmethod
    def _to_key(x) -> str:
        return "nan" if pd.isna(x) else str(x)

    def _fit_stats(self, keys: np.ndarray, y: np.ndarray):
        sums: dict[str, float] = {}
        cnts: dict[str, int] = {}
        for key, val in zip(keys, y):
            k = self._to_key(key)
            if not np.isfinite(val):
                continue
            sums[k] = sums.get(k, 0.0) + float(val)
            cnts[k] = cnts.get(k, 0) + 1
        self.sum_, self.count_ = sums, cnts
        gm = np.nanmean(y.astype(float)) if y.size else np.nan
        self.global_mean_ = float(gm) if np.isfinite(gm) else 0.0

    def _infer(self, key: str) -> float:
        # Smoothed mean for inference (non‑TRAIN rows)
        s = self.sum_.get(key, None) if self.sum_ else None
        c = self.count_.get(key, 0) if self.count_ else 0
        if s is None or c <= 0:
            return float(self.global_mean_ or 0.0)
        return float((s + self.k * (self.global_mean_ or 0.0)) / (c + self.k))

    def _train_loo(self, key: str, y_i: float) -> float:
        # Leave‑one‑out value for TRAIN rows
        s = self.sum_.get(key, None) if self.sum_ else None
        c = self.count_.get(key, 0) if self.count_ else 0
        if s is None or c <= 1:
            return float(self.global_mean_ or 0.0)
        s_loo = float(s - y_i)
        c_loo = int(c - 1)
        return float((s_loo + self.k * (self.global_mean_ or 0.0)) / (c_loo + self.k))


class SuburbLOOEncoder(_LOOBase):
    """Native LOO for a single categorical column (default: 'Suburb').
    Adds/returns a float column named '<col>_LOO'.
    """
    def __init__(self, col: str = "Suburb", k: float = 10.0, sigma: float | None = None, random_state: int | None = None):
        # Accept legacy args (sigma, random_state) for backward compatibility.
        super().__init__(k=(float(sigma) if sigma is not None else float(k)))
        self.col = col

    def fit(self, X: pd.DataFrame, y=None):
        try:
            if y is None:
                raise ValueError("Target 'y' must be provided to fit SuburbLOOEncoder.")
            if self.col not in X.columns:
                raise ValueError(f"Column '{self.col}' not found while fitting SuburbLOOEncoder.")
            keys = X[self.col].astype("string").to_numpy()
            yv = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
            self._fit_stats(keys, yv)
            return self
        except Exception as e:
            logging.error(f"SuburbLOOEncoder.fit failed: {e}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X: pd.DataFrame):
        # Inference‑only transform (no TRAIN mask knowledge)
        try:
            X_out = X.copy()
            vals: list[float] = []
            for key in X_out[self.col].astype("string").to_numpy():
                vals.append(self._infer(self._to_key(key)))
            X_out[f"{self.col}_LOO"] = np.asarray(vals, dtype=np.float32)
            return X_out
        except Exception as e:
            logging.error(f"SuburbLOOEncoder.transform failed: {e}")
            logging.error(traceback.format_exc())
            raise

    def transform_all(self, X_all: pd.DataFrame, y_all: pd.Series, train_mask: np.ndarray):
        """Leak‑safe generation for ALL rows (TRAIN gets true LOO; others get smoothed mean)."""
        try:
            keys = X_all[self.col].astype("string").to_numpy()
            yv = pd.to_numeric(pd.Series(y_all), errors="coerce").to_numpy()
            out = np.empty(keys.shape[0], dtype=np.float32)
            for i, key in enumerate(keys):
                k = self._to_key(key)
                if bool(train_mask[i]):
                    yi = float(yv[i]) if np.isfinite(yv[i]) else float(self.global_mean_ or 0.0)
                    out[i] = self._train_loo(k, yi)
                else:
                    out[i] = self._infer(k)
            return out
        except Exception as e:
            logging.error(f"SuburbLOOEncoder.transform_all failed: {e}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        return [f"{self.col}_LOO"]


class SuburbMonthLOOEncoder(_LOOBase):
    """Native LOO for (Suburb × Month) with column name 'SuburbMonth_LOO'."""
    def __init__(self, suburb_col: str = "Suburb", date_col: str = "Last Rental Date", k: float = 10.0, sigma: float | None = None, random_state: int | None = None):
        super().__init__(k=(float(sigma) if sigma is not None else float(k)))
        self.suburb_col = suburb_col
        self.date_col = date_col

    def _make_key(self, X: pd.DataFrame) -> np.ndarray:
        mon = pd.to_datetime(X[self.date_col], errors="coerce").dt.month.fillna(0).astype(int).astype("string")
        return (X[self.suburb_col].astype("string") + "_" + mon).to_numpy()

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("Target 'y' must be provided to fit SuburbMonthLOOEncoder.")
        if self.suburb_col not in X.columns or self.date_col not in X.columns:
            raise ValueError("Required columns missing for SuburbMonthLOOEncoder.")
        keys = self._make_key(X)
        yv = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy()
        self._fit_stats(keys, yv)
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        keys = self._make_key(X_out)
        vals = [self._infer(self._to_key(k)) for k in keys]
        X_out["SuburbMonth_LOO"] = np.asarray(vals, dtype=np.float32)
        return X_out

    def transform_all(self, X_all: pd.DataFrame, y_all: pd.Series, train_mask: np.ndarray):
        keys = self._make_key(X_all)
        yv = pd.to_numeric(pd.Series(y_all), errors="coerce").to_numpy()
        out = np.empty(keys.shape[0], dtype=np.float32)
        for i, k in enumerate(keys):
            key = self._to_key(k)
            if bool(train_mask[i]):
                yi = float(yv[i]) if np.isfinite(yv[i]) else float(self.global_mean_ or 0.0)
                out[i] = self._train_loo(key, yi)
            else:
                out[i] = self._infer(key)
        return out

    def get_feature_names_out(self, input_features=None):
        return ["SuburbMonth_LOO"]
