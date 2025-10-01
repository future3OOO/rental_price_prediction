#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_training.py — CatBoost rental predictor • v7.2 (Kaggle-ready)

Major upgrades vs v6.x
----------------------
• Fix: import-time stability & DEFAULT_TRIALS/DEFAULT_TIMEOUT exported
• Time-aware features: leak-safe 90d suburb median + LOO encoders
• NEW: OutlierScorer (fit on TRAIN once, frozen for EARLY/TEST) and used
       in CV folds as well; down-weights (not deletes) flagged rows
• Block CV (Monte-Carlo / walk-forward) aligned to final split
• Robust weighting: inverse-sqrt with TRAIN cap; overflow-safe expm1
• Full artifacts: pipeline, meta, importance, split CSVs, outlier flags
• GPU/CPU compatible; Optuna SH pruning; SHAP optional

This file is a drop-in replacement for earlier versions and exports:
  - train_model
  - cli
  - DEFAULT_TRIALS
  - DEFAULT_TIMEOUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import DataConversionWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
try:
    from sklearn.feature_selection import mutual_info_regression as _mi_reg
except Exception:
    _mi_reg = None  # type: ignore

import warnings
import re
import math

# ────────────────────── Global settings & exports ────────────────────── #

RANDOM_STATE = 42
N_BLOCK_FOLDS = 12
BLOCK_VAL_PCT = float(os.getenv("BLOCK_VAL_PCT", "0.20"))
GAP_DAYS = int(os.getenv("GAP_DAYS", "0"))  # day-level embargo; if >0, overrides GAP_MONTHS
TIME_DECAY_HALFLIFE_DAYS = int(os.getenv("TIME_DECAY_HALFLIFE_DAYS", "0"))
CLAMP_PRED_TO_TARGET_RANGE = os.getenv("CLAMP_PRED_TO_TARGET_RANGE", "0").lower() in {"1","true","yes"}
DRIFT_METRIC = os.getenv("DRIFT_METRIC", "wmae").lower()  # one of {wmae, rmse, mae}
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "1"))
DROP_REGION_IDS_IN_CV = os.getenv("DROP_REGION_IDS_IN_CV", "0").lower() in {"1","true","yes"}
# NEW toggles
TEST_HOLDOUT_MONTHS = int(os.getenv("TEST_HOLDOUT_MONTHS", "12"))  # 12-month TEST by default
EARLY_HOLDOUT_MONTHS = int(os.getenv("EARLY_HOLDOUT_MONTHS", "2"))
GAP_MONTHS_OVERRIDE = int(os.getenv("GAP_MONTHS", "1"))
SEARCH_HALFLIFE = os.getenv("SEARCH_HALFLIFE", "1").lower() in {"1","true","yes"}
SAVE_QUANTILE_MODELS = os.getenv("SAVE_QUANTILE_MODELS", "1").lower() in {"1","true","yes"}
QUANTILE_ALPHAS = [float(a) for a in os.getenv("QUANTILE_ALPHAS", "0.1,0.9").split(",") if a.strip()]
ENABLE_CQR = os.getenv("ENABLE_CQR", "1").lower() in {"1","true","yes"}
PI_ALPHA = float(os.getenv("PI_ALPHA", "0.20"))  # 80% coverage
ENABLE_ADVERSARIAL_VALID = os.getenv("ENABLE_ADVERSARIAL_VALID", "1").lower() in {"1","true","yes"}
ADV_LOG_SOLVER = os.getenv("ADV_LOG_SOLVER", "saga")
ADV_LOG_C = float(os.getenv("ADV_LOG_C", "0.25"))
ADV_LOG_MAX_ITER = int(os.getenv("ADV_LOG_MAX_ITER", "1000"))
ADV_LOG_TOL = float(os.getenv("ADV_LOG_TOL", "1e-3"))
ADV_WEIGHT_CAP = float(os.getenv("ADV_WEIGHT_CAP", "3.0"))
ADV_WEIGHT_FLOOR = float(os.getenv("ADV_WEIGHT_FLOOR", "0.33"))
AUTO_DRIFT_SHRINK = os.getenv("AUTO_DRIFT_SHRINK", "0").lower() in {"1","true","yes"}
ADV_AUC_STRONG = float(os.getenv("ADV_AUC_STRONG", "0.90"))
EXCLUDE_CAT_FEATURES = tuple(
    x.strip().lower() for x in os.getenv("EXCLUDE_CAT_FEATURES", "street_address,__geo_query__,Valuation Date").split(",") if x.strip()
)
EXCLUDE_GEO_BEARING = os.getenv("EXCLUDE_GEO_BEARING", "0").lower() in {"1","true","yes"}
ALWAYS_KEEP_GEO = tuple(
    s.strip().lower() for s in os.getenv(
        "ALWAYS_KEEP_GEO",
        "dist_university_km,acc_university,dist_school_primary_km,dist_school_intermediate_km,dist_school_high_km,acc_school_primary,acc_school_intermediate,acc_school_high"
    ).split(",") if s.strip()
)
ENABLE_MEDIAN_BLEND = os.getenv("ENABLE_MEDIAN_BLEND", "1").lower() in {"1","true","yes"}
MEDIAN_BLEND_LAMBDA_MAX = float(os.getenv("MEDIAN_BLEND_LAMBDA_MAX", "0.35"))
ENABLE_RESIDUAL_BOOSTER = os.getenv("ENABLE_RESIDUAL_BOOSTER", "0").lower() in {"1","true","yes"}
TEACHER_PIPELINE = os.getenv("TEACHER_PIPELINE", "").strip()
GEO_POI_CSV = os.getenv("GEO_POI_CSV", "").strip()
ADV_AUC_WARN = float(os.getenv("ADV_AUC_WARN", "0.75"))
# NEW: domain-importance weighting (covariate-shift mitigation)
ENABLE_IMPORTANCE_WEIGHTED_FIT = os.getenv("ENABLE_IMPORTANCE_WEIGHTED_FIT", "1").lower() in {"1","true","yes"}
IW_ENABLE_ON_ADV_AUC = float(os.getenv("IW_ENABLE_ON_ADV_AUC", "0.85"))
try:
    _iw_clip_vals = [float(x) for x in os.getenv("IW_CLIP", "0.2,5.0").split(",")]
    IW_CLIP_LOW, IW_CLIP_HIGH = _iw_clip_vals[0], _iw_clip_vals[1]
except Exception:
    IW_CLIP_LOW, IW_CLIP_HIGH = 0.2, 5.0
IW_TEMP = float(os.getenv("IW_TEMP", "1.0"))

# Performance guards for diagnostics (train-time only)
ADV_MAX_ROWS = int(os.getenv("ADV_MAX_ROWS", "6000"))   # cap rows for adversarial LR probe
IW_MAX_ROWS = int(os.getenv("IW_MAX_ROWS", "8000"))     # cap rows for SGD density-ratio

# Optuna budget (can be overridden by env/CLI)
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "20"))
OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT_SEC", "0"))  # 0 = no time limit
# Parallel trials (default to 8 parallel jobs)
OPTUNA_N_JOBS = int(os.getenv("OPTUNA_N_JOBS", "8"))

# CPU threading for CatBoost (0 = CatBoost default)
CB_THREAD_COUNT = int(os.getenv("CB_THREAD_COUNT", "20"))

# Small-data safeguards (env overridable)
MIN_TEST_ROWS = int(os.getenv("MIN_TEST_ROWS", "120"))
MIN_EARLY_ROWS = int(os.getenv("MIN_EARLY_ROWS", "80"))
MAX_FINAL_ITER = int(os.getenv("MAX_FINAL_ITER", "1500"))
FINAL_ES_ROUNDS = int(os.getenv("FINAL_ES_ROUNDS", "200"))

# Backward-compat constants used by main.py
DEFAULT_TRIALS = OPTUNA_TRIALS
DEFAULT_TIMEOUT = OPTUNA_TIMEOUT

# Outlier handling knobs (env-overridable)
ENABLE_OUTLIER_SCORER = os.getenv("ENABLE_OUTLIER_SCORER", "1").lower() in {"1", "true", "yes"}
OUTLIER_WEIGHT_MULT = float(os.getenv("OUTLIER_WEIGHT_MULT", "0.35"))  # down-weight factor
OUTLIER_DROP_TRAIN = os.getenv("OUTLIER_DROP_TRAIN", "0").lower() in {"1", "true", "yes"}

# Monotonic constraints toggle (off by default due to past regressions)
ENFORCE_MONOTONE = os.getenv("ENFORCE_MONOTONE", "0").lower() in {"1", "true", "yes"}

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

# Headless plotting backend (guard against Tkinter crashes during training threads)
import os as _os_for_mpl
_os_for_mpl.environ.setdefault("MPLBACKEND", "Agg")
_os_for_mpl.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:  # pragma: no cover
    import matplotlib as _mpl  # type: ignore
    if str(getattr(_mpl, "get_backend", lambda: "")()).lower().endswith("agg") is False:
        _mpl.use("Agg", force=True)
except Exception:
    pass

# Production baseline parameters for final fit (used for point + quantile models)
BEST_PARAMS_PRODUCTION = {
    # geometry / capacity
    "iterations": 3200,
    "depth": 8,
    "min_data_in_leaf": 128,
    # optimization & regularization
    "learning_rate": 0.028,
    "l2_leaf_reg": 8.0,
    "random_strength": 1.2,
    # stochastic subsampling
    "rsm": 0.62,
    "subsample": 0.82,
    # invariants / house rules
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "od_type": "Iter",
    "od_wait": 200,
    "one_hot_max_size": 12,
    "allow_writing_files": False,
    "random_seed": RANDOM_STATE,
}

# Helper: coerce arbitrary arrays/frames to numeric for light linear probes
def _coerce_for_adv_lr(arr) -> np.ndarray:
    """Coerce arbitrary (dense/sparse, mixed-type) arrays to dense float32.

    - If sparse and numeric → toarray().astype(float32)
    - If contains non-numerics → factorize per-column via pandas then to numpy
    """
    if sp.issparse(arr):
        try:
            return arr.astype(np.float32).toarray()
        except Exception:
            dense = arr.toarray()
            df_ = pd.DataFrame(dense)
    else:
        df_ = pd.DataFrame(arr) if not isinstance(arr, pd.DataFrame) else arr.copy()
    # If df_ not yet defined (numeric sparse case), create from dense array above
    if 'df_' not in locals():  # pragma: no cover (defensive)
        df_ = pd.DataFrame(np.asarray(arr))
    for c in df_.columns:
        s = df_[c]
        if not pd.api.types.is_numeric_dtype(s):
            df_[c] = pd.Categorical(s).codes.astype(np.float32)
        else:
            df_[c] = pd.to_numeric(s, errors="coerce").astype(np.float32)
    return df_.to_numpy(dtype=np.float32, copy=False)

# Feature toggles
USE_SUBURB_365D_MEDIAN = os.getenv("USE_SUBURB_365D_MEDIAN", "1").lower() in {"1", "true", "yes"}
USE_SUBURB_FEATURES = os.getenv("USE_SUBURB_FEATURES", "1").lower() in {"1", "true", "yes"}
USE_SUBURB_LOO = os.getenv("USE_SUBURB_LOO", "0").lower() in {"1", "true", "yes"}
USE_SUBURB_MEDIANS = os.getenv("USE_SUBURB_MEDIANS", "1").lower() in {"1", "true", "yes"}
USE_SUBURB_RANK = os.getenv("USE_SUBURB_RANK", "0").lower() in {"1", "true", "yes"}

# Median computation controls (for stability when USE_SUBURB_MEDIANS=1)
USE_LOG_MEDIANS = os.getenv("USE_LOG_MEDIANS", "1").lower() in {"1", "true", "yes"}
SUBURB_MED_WINDOW_DAYS = int(os.getenv("SUBURB_MED_WINDOW_DAYS", "90"))
SUBURB_MED_MIN_COUNT = int(os.getenv("SUBURB_MED_MIN_COUNT", "8"))
SUBURB_MED365_MIN_COUNT = int(os.getenv("SUBURB_MED365_MIN_COUNT", "16"))
SUBURB_BED_MIN_COUNT = int(os.getenv("SUBURB_BED_MIN_COUNT", "6"))
INCLUDE_RAW_SUBURB_MEDIANS = os.getenv("INCLUDE_RAW_SUBURB_MEDIANS", "0").lower() in {"1", "true", "yes"}

# Silence benign warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)

__all__ = ["train_model", "cli", "DEFAULT_TRIALS", "DEFAULT_TIMEOUT"]
_cv_objective_for_test = None  # populated during tune_catboost for unit tests

# Optional transformers (graceful degrade)
try:
    from transformers import SuburbLOOEncoder, SuburbMonthLOOEncoder  # type: ignore
except Exception:
    SuburbLOOEncoder = SuburbMonthLOOEncoder = None  # type: ignore

try:
    from featurealgo.monthly_rank import MonthlyRankingTransformer  # type: ignore
except Exception:
    MonthlyRankingTransformer = None  # type: ignore


# ────────────────────────── Logging helper ────────────────────────── #

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


# ───────────────────────── GPU detection helper ───────────────────────── #
def _gpu_available() -> bool:
    try:
        from catboost.utils import get_gpu_device_count  # type: ignore
        return int(get_gpu_device_count()) > 0
    except Exception:
        return False

# ───────────────────── Gap & weighting helpers ───────────────────── #
def _auto_gap_days() -> int:
    """
    Auto compute a safe day-level embargo between TRAIN and VAL:
      - takes the longest temporal window used in engineered medians
      - adds a small buffer (7d)
      - returns days (0 if no median features enabled)
    """
    days = 0
    if USE_SUBURB_MEDIANS:
        days = max(days, SUBURB_MED_WINDOW_DAYS)
        if USE_SUBURB_365D_MEDIAN:
            days = max(days, 365)
    return max(0, int(days + 7))

def _recency_weights(dates: pd.Series, ref_date: pd.Timestamp, halflife_days: int) -> np.ndarray:
    """Exponential time-decay weights; larger for recent rows."""
    if halflife_days <= 0:
        return np.ones(len(dates), dtype=float)
    # Robust to Series, Index, array-like
    td = pd.to_datetime(ref_date) - pd.to_datetime(dates)
    try:
        # Convert timedeltas to integer days without relying on .dt for Index
        age = (td / np.timedelta64(1, 'D')).astype(int)
    except Exception:
        age = pd.to_timedelta(td).astype('timedelta64[D]').astype(int)
    age = np.clip(np.asarray(age, dtype=int), 0, None)
    lam = math.log(2.0) / max(halflife_days, 1)
    w = np.exp(-lam * age).astype(float)
    # Normalize so the most recent observation has weight 1.0
    m = float(w.max()) if w.size else 1.0
    return (w / m) if m > 0 else w


# ─────────────────────── Custom transformers ─────────────────────── #

class RareCategoryBinner(BaseEstimator, TransformerMixin):
    """Bins infrequent labels to 'OTHER' to stabilise categorical signal."""
    def __init__(self, min_count: int = 20, replacement: str = "OTHER"):
        self.min_count = min_count
        self.replacement = replacement
        self.frequent_: dict[str, set] = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.frequent_.clear()
        for col in X_df.columns:
            vc = X_df[col].astype("string").value_counts(dropna=False)
            self.frequent_[col] = set(vc[vc >= self.min_count].index)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            keep = self.frequent_.get(col, set())
            ser = X_df[col].astype("string")
            ser = ser.where(ser.isin(keep), other=self.replacement).fillna(self.replacement)
            X_df[col] = ser.astype("string")
        return X_df


class MissingLevelImputer(BaseEstimator, TransformerMixin):
    """Impute missing categoricals to a dedicated 'MISSING' level (train=serve)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            ser = X_df[col]
            # Convert to string but preserve 'MISSING' explicitly
            ser = ser.astype("string")
            ser = ser.fillna("MISSING")
            X_df[col] = ser
        return X_df

class ToStringTransformer(BaseEstimator, TransformerMixin):
    """Ensure string dtype for CatBoost native-categorical ingestion."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].astype("string")
        return X_df


class PreInputAdapter(BaseEstimator, TransformerMixin):
    """Apply string cleaning and date parts so train=serve inside the pipeline.

    This mirrors _clean_strings/_apply_ordinals/add_date_parts used earlier so that
    the persisted preprocessor can reproduce those columns at serve-time.
    """
    def __init__(self, *, date_col: str):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        # String cleaning on key categoricals
        df = _apply_ordinals(_clean_strings(df))
        # Date parts (if present)
        if self.date_col in df.columns:
            df = add_date_parts(df, self.date_col)
        # Safe log features for skewed numerics commonly used downstream (exclude Car)
        for base_col in ("Land Value", "Capital Value", "Days on Market"):
            if base_col in df.columns and f"log_{base_col}" not in df.columns:
                df[f"log_{base_col}"] = np.log1p(pd.to_numeric(df[base_col], errors="coerce"))
        return df


# ------------------- New schema alias + normalization helpers ------------------- #

# Source-of-truth aliases for new CSV schemas
_ALIASES = {
    "target": ["last_rent_price", "last_rental_price", "rent", "weekly_rent", "Last Rental Price", "Rent"],
    "date_col": ["last_rent_date", "last_rental_date", "date", "Last Rental Date", "Date"],
    "suburb_col": ["suburb", "locality", "Suburb"],
    "postcode": ["postcode", "post_code", "postal_code", "Postcode"],
    "carparks_total": ["Carparks", "car_parks_total", "carparks", "car"],
    "garage_parks": ["garage_parks", "Garage Parks"],
    "offstreet_parks": ["offstreet_parks", "off_street_parks", "Offstreet Parks"],
    "furnishings": ["furnishingsF", "furnishings", "Furnishings"],
    "valuation_date": ["valuation", "valuation_date", "Valuation Date"],
    # NEW: geo coordinate aliases to ensure Latitude/Longitude are canonicalized early
    "latitude": ["latitude", "lat", "gps_lat", "Latitude", "LAT"],
    "longitude": ["longitude", "lon", "lng", "gps_lon", "Longitude", "LON"],
}

_ALIAS_CANONICAL_MAP = {
    "last_rent_price": "Last Rental Price",
    "last_rental_price": "Last Rental Price",
    "weekly_rent": "Last Rental Price",
    "rent": "Last Rental Price",
    "last_rent_date": "Last Rental Date",
    "last_rental_date": "Last Rental Date",
    "date": "Last Rental Date",
    "suburb": "Suburb",
    "locality": "Suburb",
    "postcode": "Postcode",
    "post_code": "Postcode",
    "postal_code": "Postcode",
    "land_size (m²)": "Land Size (sqm)",
    "land_size (m2)": "Land Size (sqm)",
    "floor_size (m²)": "Floor Size (sqm)",
    "floor_size (m2)": "Floor Size (sqm)",
    "year_built": "Year Built",
    "bedrooms": "Bed",
    "bathrooms": "Bath",
    "baths": "Bath",
    "property_type": "Property Type",
    "days_on_market": "Days on Market",
    "capital_value": "Capital Value",
    "land_value": "Land Value",
    "improvement_value": "Improvement Value",
    "valuation": "Valuation Date",
    "valuation_date": "Valuation Date",
    "furnishingsf": "Furnishings",
    "furnishings": "Furnishings",
    "carparks": "Car",
    "car": "Car",
    "latitude": "Latitude",
    "lat": "Latitude",
    "gps_lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "gps_lon": "Longitude",
}


def _rename_alias_columns(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    work = df.copy()
    for alias_lower, canonical in alias_map.items():
        matches = [col for col in work.columns if col.lower() == alias_lower]
        for col in matches:
            if col == canonical:
                continue
            if canonical in work.columns:
                work[canonical] = work[canonical].combine_first(work[col])
                work.drop(columns=[col], inplace=True)
            else:
                work.rename(columns={col: canonical}, inplace=True)
    return work


# ----------------------- Low-entropy categorical guard ------------------------ #
LOW_ENTROPY_CAT_TOPRATIO = float(os.getenv("LOW_ENTROPY_CAT_TOPRATIO", "0.85"))
LOW_ENTROPY_MIN_LEVELS = int(os.getenv("LOW_ENTROPY_MIN_LEVELS", "2"))

def _drop_low_entropy_cats_df(df: pd.DataFrame, cats: list[str]) -> tuple[list[str], list[str]]:
    """Drop categorical columns where the most frequent level dominates (e.g., >85%).

    Returns (kept_cats, dropped_cats). Safe for small data; ignores columns not in df.
    """
    kept, dropped = [], []
    for c in cats:
        if c not in df.columns:
            continue
        ser = df[c].astype("string")
        vc = ser.value_counts(dropna=True)
        if vc.empty:
            dropped.append(c)
            continue
        top_ratio = float(vc.iloc[0]) / max(1, int(vc.sum()))
        n_levels = int(vc.shape[0])
        if (n_levels >= LOW_ENTROPY_MIN_LEVELS) and (top_ratio >= LOW_ENTROPY_CAT_TOPRATIO):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


def _parse_geo_radii(env_value: str | None, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if not env_value:
        return default
    try:
        vals = tuple(float(x.strip()) for x in env_value.split(" ") if x.strip())
        if vals:
            return vals
    except Exception:
        pass
    try:
        vals = tuple(float(x.strip()) for x in env_value.split(",") if x.strip())
        if vals:
            return vals
    except Exception:
        pass
    return default


def _parse_geo_categories(env_value: str | None) -> Tuple[str, ...] | None:
    if not env_value:
        return None
    cats = [c.strip() for part in env_value.split(",") for c in part.split(" ") if c.strip()]
    cats = [c.upper().replace(" ", "_") for c in cats]
    return tuple(dict.fromkeys(cats)) if cats else None


def _geo_env_config() -> dict[str, Any] | None:
    """Assemble geo configuration from ENV with config.py fallbacks.

    Prefers environment variables (CLI/CI). If not set, reads defaults from
    config.py (GEO_POI_CSV, GEO_LAT_COL, GEO_LON_COL, GEO_RADII_KM, GEO_DECAY_KM,
    GEO_MAX_DECAY_KM, GEO_CATEGORIES). Returns None if no POI CSV is available.
    """
    poi_csv = GEO_POI_CSV
    lat_col = os.getenv("LAT_COL")
    lon_col = os.getenv("LON_COL")
    radii_env = os.getenv("GEO_RADII_KM")
    decay_env = os.getenv("GEO_DECAY_KM")
    maxdecay_env = os.getenv("GEO_MAX_DECAY_KM")
    cats_env = os.getenv("GEO_CATEGORIES")

    # Fallback to config defaults when ENV is empty
    if not poi_csv or not lat_col or not lon_col or not radii_env or not decay_env or not maxdecay_env or not cats_env:
        try:
            import config as _cfg_geo  # type: ignore
            if not poi_csv:
                _p = getattr(_cfg_geo, "GEO_POI_CSV", None)
                if _p and Path(_p).exists():
                    poi_csv = str(_p)
            if not lat_col:
                lat_col = getattr(_cfg_geo, "GEO_LAT_COL", "Latitude")
            if not lon_col:
                lon_col = getattr(_cfg_geo, "GEO_LON_COL", "Longitude")
            if not radii_env:
                _r = getattr(_cfg_geo, "GEO_RADII_KM", (0.5, 1.0, 2.0))
                try:
                    radii_env = ",".join(str(float(v)) for v in _r)
                except Exception:
                    radii_env = None
            if not decay_env:
                decay_env = str(getattr(_cfg_geo, "GEO_DECAY_KM", 1.5))
            if not maxdecay_env:
                maxdecay_env = str(getattr(_cfg_geo, "GEO_MAX_DECAY_KM", 3.0))
            if not cats_env:
                _c = getattr(_cfg_geo, "GEO_CATEGORIES", None)
                if _c:
                    try:
                        cats_env = ",".join(_c)
                    except Exception:
                        cats_env = None
        except Exception:
            pass

    if not poi_csv:
        return None

    lat_col = lat_col or "Latitude"
    lon_col = lon_col or "Longitude"
    radii = _parse_geo_radii(radii_env, (0.5, 1.0, 2.0))
    try:
        decay_km = float(decay_env) if decay_env is not None else 1.5
    except Exception:
        decay_km = 1.5
    try:
        max_decay_km = float(maxdecay_env) if maxdecay_env is not None else 3.0
    except Exception:
        max_decay_km = 3.0
    categories = _parse_geo_categories(cats_env)
    return {
        "poi_csv": poi_csv,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "radii_km": radii,
        "decay_km": decay_km,
        "max_decay_km": max_decay_km,
        "categories": categories,
    }


def _record_removed_rows(tracker: Any, step: str, reference_df: pd.DataFrame, removed_ids: List[int]) -> None:
    if tracker is None or not removed_ids:
        return
    try:
        tracker.track_ids(removed_ids, reference_df, step)
    except AttributeError:
        try:
            remaining = reference_df[~reference_df["__row_id"].isin(removed_ids)]
            tracker.track(reference_df, remaining, step)
        except Exception:
            logging.debug("RemovedRowsTracker fallback failed for step '%s'", step)
    except Exception as exc:
        logging.debug("RemovedRowsTracker logging failed for step '%s': %s", step, exc)

def _pick_first(df_cols: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df_cols:
            return c
        # case-insensitive match
        for dc in df_cols:
            if str(dc).lower() == str(c).lower():
                return dc
    return None

def _normalize_new_schema(
    df_in: pd.DataFrame,
    *,
    target_preference: list[str] | None = None,
    date_preference: list[str] | None = None,
    suburb_preference: list[str] | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """
    Map flexible input schemas (snake_case or legacy Title Case) to canonical
    training columns used across the pipeline:
      - Last Rental Price (target)
      - Last Rental Date (date)
      - Suburb (location key)

    Also derives common structural features from the provided schema:
      - Bed (int), Bath (float), Car (int)
      - Land Size (sqm), Floor Size (sqm), Year Built
      - Postcode (zero-padded string)
    and applies conservative clamps per spec.
    """
    df = _rename_alias_columns(df_in, _ALIAS_CANONICAL_MAP)
    cols = set(df.columns)
    tgt_name = _pick_first(cols, target_preference or _ALIASES["target"]) or "Last Rental Price"
    dt_name = _pick_first(cols, date_preference or _ALIASES["date_col"]) or "Last Rental Date"
    sb_name = _pick_first(cols, suburb_preference or _ALIASES["suburb_col"]) or "Suburb"

    # Create canonical columns if not already present
    if "Last Rental Price" not in df.columns and tgt_name in df.columns:
        df["Last Rental Price"] = pd.to_numeric(df[tgt_name], errors="coerce")
    if "Last Rental Date" not in df.columns and dt_name in df.columns:
        # dayfirst=True for new CSVs
        df["Last Rental Date"] = pd.to_datetime(df[dt_name], dayfirst=True, errors="coerce")
    if "Suburb" not in df.columns and sb_name in df.columns:
        df["Suburb"] = df[sb_name].astype("string").str.strip().str.upper()

    # Bed/Bath from schema if available
    if "Bed" not in df.columns:
        for cand in ("bedrooms", "beds", "Bed"):
            if cand in df.columns:
                df["Bed"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "Bath" not in df.columns:
        for cand in ("bathrooms", "baths", "Bath"):
            if cand in df.columns:
                df["Bath"] = pd.to_numeric(df[cand], errors="coerce")
                break

    # Carparks: prefer explicit Car else derive from components
    if "Car" not in df.columns:
        car_total_col = _pick_first(cols, _ALIASES["carparks_total"])  # may be 'Carparks' or 'car'
        g_col = _pick_first(cols, _ALIASES["garage_parks"]) or None
        o_col = _pick_first(cols, _ALIASES["offstreet_parks"]) or None
        car = None
        if car_total_col and car_total_col in df.columns:
            car = pd.to_numeric(df[car_total_col], errors="coerce")
        if (g_col and g_col in df.columns) or (o_col and o_col in df.columns):
            g = pd.to_numeric(df[g_col], errors="coerce") if (g_col and g_col in df.columns) else 0
            o = pd.to_numeric(df[o_col], errors="coerce") if (o_col and o_col in df.columns) else 0
            comp = g + o
            car = comp if car is None else np.fmax(car, comp)
        if car is not None:
            df["Car"] = pd.to_numeric(car, errors="coerce").fillna(0)

    # Sizes (m² -> sqm)
    if "Land Size (sqm)" not in df.columns:
        for cand in ("land_size (m²)", "land_size (m2)", "Land Size (m�)", "Land Size (sqm)"):
            if cand in df.columns:
                df["Land Size (sqm)"] = pd.to_numeric(df[cand], errors="coerce")
                break
    if "Floor Size (sqm)" not in df.columns:
        for cand in ("floor_size (m²)", "floor_size (m2)", "Floor Size (m�)", "Floor Size (sqm)"):
            if cand in df.columns:
                df["Floor Size (sqm)"] = pd.to_numeric(df[cand], errors="coerce")
                break

    # Latitude/Longitude
    if "Latitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns:
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Year Built
    if "Year Built" not in df.columns:
        for cand in ("year_built", "YearBuilt", "Year Built"):
            if cand in df.columns:
                df["Year Built"] = pd.to_numeric(df[cand], errors="coerce")
                break

    # Postcode as zero-padded string
    pc_col = _pick_first(set(df.columns), _ALIASES["postcode"]) or None
    if pc_col:
        pc = df[pc_col].astype("Int64", copy=False)
        df["Postcode"] = pc.astype("string").str.zfill(4)

    # Property Type + basic bucketing
    if "Property Type" not in df.columns:
        for cand in ("property_type", "Property Type"):
            if cand in df.columns:
                df["Property Type"] = (
                    df[cand]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .replace({
                        r".*apartment.*": "apartment",
                        r".*unit.*": "unit",
                        r".*town.*house.*|.*townhouse.*": "townhouse",
                        r".*house.*": "house",
                    }, regex=True)
                )
                break

    # Furnishings normalization
    furn_col = _pick_first(set(df.columns), _ALIASES["furnishings"]) or None
    if furn_col:
        df["Furnishings"] = (
            df[furn_col]
            .astype("string")
            .str.lower()
            .replace({"nill": "none", "nil": "none", "none": "none", "partial": "partial", "fully furnished": "full", "full": "full"})
        )

    # Days on Market
    if "Days on Market" not in df.columns:
        for cand in ("days_on_market", "dom", "Days on Market"):
            if cand in df.columns:
                ser = pd.to_numeric(df[cand], errors="coerce")
                # Winsorize at P99 as a light clamp
                try:
                    p99 = float(np.nanquantile(ser, 0.99))
                except Exception:
                    p99 = np.nan
                df["Days on Market"] = ser.clip(lower=0, upper=p99 if np.isfinite(p99) else None)
                break

    # Clamp numeric ranges per spec
    if "Bed" in df.columns:
        df["Bed"] = pd.to_numeric(df["Bed"], errors="coerce").clip(lower=0, upper=8)
    if "Bath" in df.columns:
        df["Bath"] = pd.to_numeric(df["Bath"], errors="coerce").clip(lower=0, upper=6)
    if "Year Built" in df.columns:
        df["Year Built"] = pd.to_numeric(df["Year Built"], errors="coerce").clip(lower=1900, upper=2025)
    if "Last Rental Price" in df.columns:
        df["Last Rental Price"] = pd.to_numeric(df["Last Rental Price"], errors="coerce").clip(lower=150, upper=2500)

    # Standardize string columns used downstream
    for c in ("Suburb", "Agency", "Agent", "Land Use", "Development Zone", "Category"):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()

    # Return canonical names for downstream
    return df, "Last Rental Price", "Last Rental Date", "Suburb"


# ─────────────────────── Numeric robust cleaner ─────────────────────── #

class RobustWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features to robust quantile bounds learned on TRAIN only.

    This is leakage-safe in CV since the transformer is fit only on the fold's
    training partition and applied to validation via the pipeline.
    """

    def __init__(self, lower_q: float = 0.005, upper_q: float = 0.995):
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.bounds_.clear()
        for col in X_df.columns:
            s = pd.to_numeric(X_df[col], errors="coerce").astype("Float64")
            if s.notna().any():
                lo = float(np.nanquantile(s, self.lower_q))
                hi = float(np.nanquantile(s, self.upper_q))
            else:
                lo, hi = (np.nan, np.nan)
            self.bounds_[str(col)] = (lo, hi)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        out = X_df.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in out.columns:
                # Ensure float dtype to safely apply float quantile bounds across datasets
                s = pd.to_numeric(out[col], errors="coerce").astype("Float64")
                if np.isfinite(lo):
                    s = s.where(s >= lo, lo)
                if np.isfinite(hi):
                    s = s.where(s <= hi, hi)
                out[col] = s
        return out


# ──────────────────────── Outlier Scorer ─────────────────────────── #

class OutlierScorer:
    """
    Leak-safe outlier scorer:
      • Feature 1: zMAD of log(target) residual vs Suburb90dMedian
      • Feature 2: zMAD of log(target per floor_sqm) (where floor>0)
    Fit on TRAIN only, store per-suburb robust med/scale, apply unchanged to
    EARLY/TEST. Use for down-weighting (not deletion) outside TRAIN by default.
    """

    def __init__(
        self,
        *,
        target_col: str,
        suburb_col: str,
        date_col: str,
        floor_col: str | None = "Floor Size (sqm)",
        k_resid: float = 4.5,
        k_pps: float = 4.0,
        min_group: int = 10,
    ):
        self.tgt = target_col
        self.suburb = suburb_col
        self.date = date_col
        self.floor = floor_col
        self.k_resid = k_resid
        self.k_pps = k_pps
        self.min_group = min_group
        self._stats_resid: dict[str, tuple[float, float]] = {}
        self._stats_pps: dict[str, tuple[float, float]] = {}
        self._global_resid: tuple[float, float] | None = None
        self._global_pps: tuple[float, float] | None = None

    @staticmethod
    def _zmad(x: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
        # Guard against all-NaN slices
        x_fin = x[np.isfinite(x)]
        if x_fin.size == 0:
            return 0.0, 1.0
        med = float(np.nanmedian(x_fin))
        mad = float(np.nanmedian(np.abs(x_fin - med)))
        if not np.isfinite(mad) or mad == 0:
            return med, 1.0
        scale = 1.4826 * mad
        return med, float(max(scale, eps))

    def fit(self, df: pd.DataFrame) -> "OutlierScorer":
        xlog = np.log1p(pd.to_numeric(df[self.tgt], errors="coerce"))
        # Residual vs 90d median (computed beforehand in make_time_features)
        if "Suburb90dMedian" in df.columns:
            med90 = np.log1p(pd.to_numeric(df["Suburb90dMedian"], errors="coerce"))
            resid = xlog - med90
        else:
            # Fallback: plain log(target)
            resid = xlog.copy()

        # Rent-per-floor sqm (where floor > 0)
        pps = pd.Series(np.nan, index=df.index, dtype=float)
        if self.floor and self.floor in df.columns:
            floor = pd.to_numeric(df[self.floor], errors="coerce")
            # Keep as Series aligned to df.index for downstream .loc
            pps = (xlog - np.log1p(floor)).where(floor > 0, np.nan)

        # Per-suburb robust stats (only if enough rows)
        self._stats_resid.clear()
        self._stats_pps.clear()

        for sub, grp in df.groupby(self.suburb, dropna=False):
            if len(grp) >= self.min_group:
                m_r, s_r = self._zmad((resid.loc[grp.index]).to_numpy())
                self._stats_resid[str(sub)] = (m_r, s_r)
                if np.isfinite(pps.loc[grp.index]).sum() >= self.min_group:
                    m_p, s_p = self._zmad((pps.loc[grp.index]).to_numpy())
                    self._stats_pps[str(sub)] = (m_p, s_p)

        # Global fallback
        self._global_resid = self._zmad(resid.to_numpy())
        if np.isfinite(pps).sum():
            self._global_pps = self._zmad(pps.to_numpy())
        else:
            self._global_pps = None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        xlog = np.log1p(pd.to_numeric(df[self.tgt], errors="coerce"))
        if "Suburb90dMedian" in df.columns:
            med90 = np.log1p(pd.to_numeric(df["Suburb90dMedian"], errors="coerce"))
            resid = xlog - med90
        else:
            resid = xlog.copy()

        pps = pd.Series(np.nan, index=df.index, dtype=float)
        if self.floor and self.floor in df.columns:
            floor = pd.to_numeric(df[self.floor], errors="coerce")
            pps = (xlog - np.log1p(floor)).where(floor > 0, np.nan)

        z_resid = np.empty(len(df), dtype=float)
        z_pps = np.full(len(df), np.nan, dtype=float)

        for i, (sub, r, p) in enumerate(zip(df[self.suburb].astype("string"), resid, pps)):
            m_r, s_r = self._stats_resid.get(str(sub), self._global_resid)  # type: ignore
            z_resid[i] = 0.0 if s_r == 0 or not np.isfinite(r) else (r - m_r) / s_r
            if np.isfinite(p):
                if str(sub) in self._stats_pps:
                    m_p, s_p = self._stats_pps[str(sub)]
                else:
                    m_p, s_p = self._global_pps if self._global_pps else (0.0, np.nan)
                z_pps[i] = 0.0 if (not np.isfinite(s_p)) or s_p == 0 else (p - m_p) / s_p

        flag = (np.abs(z_resid) > self.k_resid) | (np.abs(z_pps) > self.k_pps)
        return pd.DataFrame(
            {"_z_resid": z_resid, "_z_pps": z_pps, "_outlier": flag.astype(bool)},
            index=df.index,
        )


# ─────────────────────────── Feature helpers ─────────────────────────── #

_STRING_COLS = [
    "Land Use",
    "Development Zone",
    "Property Type",
    "Suburb",
    "Agency",
    "Agent",
    "Owner Type",
    "Street Address",
    "Postcode",
]

# Ordinal maps (applied only if columns exist)
_ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "Condition Rating": {"POOR": 1, "FAIR": 2, "AVERAGE": 3, "GOOD": 4, "EXCELLENT": 5},
    "Build Quality": {"POOR": 1, "FAIR": 2, "AVERAGE": 3, "GOOD": 4, "EXCELLENT": 5},
    # Common MLS-like kitchen ratings; safe no-op if column absent:
    "Kitchen Qual": {"PO": 1, "FA": 2, "TA": 3, "GD": 4, "EX": 5},
}

def _apply_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in _ORDINAL_MAPS.items():
        if col in out.columns:
            out[col] = (
                out[col].astype("string", copy=False).str.upper().map(mapping).astype("Int8")
            )
    return out


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _STRING_COLS:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string", copy=False)
                .str.upper()
                .str.strip()
                .str.replace(r"[\\/]", "_", regex=True)
                .fillna("UNKNOWN")
            )
    return out


def add_date_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    out = df.copy()
    out["Year"] = dt.dt.year
    out["Month"] = dt.dt.month
    out["WeekOfYear"] = dt.dt.isocalendar().week.astype("Int16")
    out["DayOfWeek"] = dt.dt.dayofweek.astype("Int8")
    out["Month_sin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["Month_cos"] = np.cos(2 * np.pi * out["Month"] / 12.0)
    return out


# Unified rolling median helper (module-level) used by train and serve-bundle code
def _roll_median(s: pd.Series, window: str, min_count: int) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").sort_index()
    if USE_LOG_MEDIANS:
        v = np.log1p(v)
    m = v.shift().rolling(window, closed="left", min_periods=min_count).median()
    if USE_LOG_MEDIANS:
        m = np.expm1(m)
    return m


# Build 90D leak-safe rolling features and LOO encodings using TRAIN-only history
def make_time_features(
    df: pd.DataFrame,
    fit_idx: np.ndarray,
    *,
    date_col: str,
    target_col: str,
    suburb_col: str,
    bed_col: str = "Bed",
) -> pd.DataFrame:
    work = df.copy()
    if work.empty or fit_idx.size == 0:
        return work
    if not USE_SUBURB_FEATURES:
        return work
    # rolling 90-day median of target using TRAIN rows only
    train_mask = np.zeros(len(work), dtype=bool)
    valid_fit_idx = fit_idx[(fit_idx >= 0) & (fit_idx < len(work))]
    if valid_fit_idx.size == 0:
        return work
    train_mask[valid_fit_idx] = True
    t_train = work[target_col].where(pd.Series(train_mask, index=work.index), np.nan)

    def _roll_median(s: pd.Series, window: str, min_count: int) -> pd.Series:
        v = pd.to_numeric(s, errors="coerce").sort_index()
        # Optionally operate in log domain for scale stability
        if USE_LOG_MEDIANS:
            v = np.log1p(v)
        m = v.shift().rolling(window, closed="left", min_periods=min_count).median()
        if USE_LOG_MEDIANS:
            m = np.expm1(m)
        return m

    med90 = (
        work.assign(_t=t_train)
        .set_index(date_col)
        .groupby(suburb_col)["_t"]
        .apply(lambda x: _roll_median(x, f"{SUBURB_MED_WINDOW_DAYS}D", SUBURB_MED_MIN_COUNT))
        .rename("Suburb90dMedian")
        .reset_index()
    )
    med90 = med90.drop_duplicates(subset=[suburb_col, date_col], keep="last")
    if USE_SUBURB_MEDIANS:
        work = work.merge(med90, on=[suburb_col, date_col], how="left")
        # Optional: also expose raw target rolling median in parallel to log-domain one for diagnostics
        if INCLUDE_RAW_SUBURB_MEDIANS:
            med90_raw = (
                work.assign(_t=t_train)
                .set_index(date_col)
                .groupby(suburb_col)["_t"]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").sort_index().shift().rolling(f"{SUBURB_MED_WINDOW_DAYS}D", closed="left", min_periods=SUBURB_MED_MIN_COUNT).median())
                .rename("Suburb90dMedian_raw")
                .reset_index()
            )
            work = work.merge(med90_raw, on=[suburb_col, date_col], how="left")

    # 365D median (12-month) leak-safe – optional, controlled by flag
    if USE_SUBURB_MEDIANS and USE_SUBURB_365D_MEDIAN:
        med365 = (
            work.assign(_t=t_train)
            .set_index(date_col)
            .groupby(suburb_col)["_t"]
            .apply(lambda x: _roll_median(x, "365D", SUBURB_MED365_MIN_COUNT))
            .rename("Suburb365dMedian")
            .reset_index()
        )
        med365 = med365.drop_duplicates(subset=[suburb_col, date_col], keep="last")
        work = work.merge(med365, on=[suburb_col, date_col], how="left")
        # Bed-conditional 365D median per suburb (if Bed present)
        if bed_col in work.columns:
            work[bed_col] = work[bed_col].astype("string")
            med365_bed = (
                work.assign(_t=t_train)
                .set_index(date_col)
                .groupby([suburb_col, bed_col])["_t"]
                .apply(lambda x: _roll_median(x, "365D", SUBURB_MED365_MIN_COUNT))
                .rename("SuburbBed365dMedian")
                .reset_index()
            )
            med365_bed = med365_bed.drop_duplicates(subset=[suburb_col, bed_col, date_col], keep="last")
            work = work.merge(med365_bed, on=[suburb_col, bed_col, date_col], how="left")

    # Bed-conditional 90D median per suburb (string-normalized bed)
    if USE_SUBURB_MEDIANS and bed_col in work.columns:
        work[bed_col] = work[bed_col].astype("string")
        med90_bed = (
            work.assign(_t=t_train)
            .set_index(date_col)
            .groupby([suburb_col, bed_col])["_t"]
            .apply(lambda x: _roll_median(x, f"{SUBURB_MED_WINDOW_DAYS}D", SUBURB_BED_MIN_COUNT))
            .rename("SuburbBed90dMedian")
            .reset_index()
        )
        # Guard against duplicate keys causing many-to-many merge explosions
        med90_bed = med90_bed.drop_duplicates(subset=[suburb_col, bed_col, date_col], keep="last")
        work = work.merge(med90_bed, on=[suburb_col, bed_col, date_col], how="left")

    # Bed-normalized floor size ratio to bed-level median floor sqm (TRAIN-only)
    try:
        if bed_col in work.columns and "Floor Size (sqm)" in work.columns:
            tmp = work.loc[fit_idx, [bed_col, "Floor Size (sqm)"]].copy()
            tmp["Floor Size (sqm)"] = pd.to_numeric(tmp["Floor Size (sqm)"], errors="coerce")
            bed_med = tmp.groupby(bed_col, dropna=False)["Floor Size (sqm)"].median().to_dict()
            bed_series = work[bed_col].astype("string")
            med_series = bed_series.map(bed_med)
            flr = pd.to_numeric(work["Floor Size (sqm)"], errors="coerce")
            work["floor_to_bed_med"] = (flr / pd.to_numeric(med_series, errors="coerce").replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            work["floor_to_bed_med"] = work["floor_to_bed_med"].fillna(1.0)
    except Exception as e:
        logging.warning("floor_to_bed_med computation failed: %s", e)

    # LOO encoders trained on TRAIN only, applied to all rows (enabled via env)
    if USE_SUBURB_LOO and os.getenv("DISABLE_LOO", "0").lower() not in {"1", "true", "yes"} and SuburbLOOEncoder is not None:
        enc_sub = SuburbLOOEncoder(col=suburb_col, sigma=0.1, random_state=RANDOM_STATE)
        enc_sm = SuburbMonthLOOEncoder(
            suburb_col=suburb_col, date_col=date_col, sigma=0.1, random_state=RANDOM_STATE
        )
        enc_sub.fit(work.loc[fit_idx][[suburb_col]], work.loc[fit_idx][target_col])
        enc_sm.fit(work.loc[fit_idx][[suburb_col, date_col]], work.loc[fit_idx][target_col])
        work["Suburb_LOO"] = enc_sub.transform(work[[suburb_col]])[f"{suburb_col}_LOO"].values
        work["SuburbMonth_LOO"] = enc_sm.transform(work[[suburb_col, date_col]])["SuburbMonth_LOO"].values

    # Optional monthly ranking transformer
    if USE_SUBURB_RANK and MonthlyRankingTransformer is not None:
        try:
            ranker = MonthlyRankingTransformer(date_col=date_col, price_col=target_col)
            ranker.fit(work.loc[fit_idx])
            work = ranker.transform(work)
        except Exception as e:
            logging.warning("MonthlyRankingTransformer skipped: %s", e)
    # Ensure key count columns remain numeric for downstream feature typing
    for _c in {bed_col, "Bath", "Car"}:
        if _c and _c in work.columns:
            work[_c] = pd.to_numeric(work[_c], errors="coerce")
    return work


# ───────────────────────────── Preprocessor ───────────────────────────── #

def build_preprocessor(
    df_train: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale_numeric: bool,
    *,
    date_col: str,
) -> Tuple[ColumnTransformer, List[int]]:
    num_steps: list = [("winsor", RobustWinsorizer(lower_q=float(os.getenv("WINSOR_LO", "0.02")), upper_q=float(os.getenv("WINSOR_HI", "0.98")))), ("imp", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # Stronger binning for high-cardinality IDs (agency/agent/address)
    bin_min = int(os.getenv("RARE_BIN_MIN", "60"))
    cat_pipe = Pipeline(
        [
            ("missing", MissingLevelImputer()),
            # Bed-aware binning for 'Car' to reduce spurious penalties for rare 2-car on 2-bed
            ("bin", RareCategoryBinner(min_count=bin_min, replacement="OTHER")),
            ("to_str", ToStringTransformer()),
        ]
    )

    transformers: list = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    # Base column transformer
    base_ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    # Wrap with a pre-input adapter so serve-time reproduces train feature columns
    preproc = Pipeline([
        ("preinput", PreInputAdapter(date_col=date_col)),
        ("ct", base_ct),
    ])
    n_num = len(numeric_cols)
    cat_idx = list(range(n_num, n_num + len(categorical_cols)))
    return preproc, cat_idx


# ───────────────────────────── CV helpers ───────────────────────────── #

def make_block_labels(dates: pd.Series, freq: str = "M") -> np.ndarray:
    return (
        pd.to_datetime(dates)
        .dt.to_period(freq)
        .astype("int64", copy=False)
        .to_numpy()
    )

# --- NEW: metric and small utilities ---
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    err = np.abs((y_true - y_pred) / denom)
    try:
        return float(np.average(err, weights=w) if w is not None else np.mean(err))
    except Exception:
        return float(np.mean(err))

def _drop_raw_date_from_cats(cat_cols: list[str], date_col: str) -> list[str]:
    date_like = {str(date_col), "Valuation Date", "Last Rental Date"}
    return [c for c in cat_cols if c not in date_like]

# --- NEW: deterministic month-based holdout ---
def split_holdout_by_months(
    dates: pd.Series, *, test_months: int, early_months: int, gap_months: int = 0
):
    labels = make_block_labels(dates, "M")
    uniq = np.sort(np.unique(labels))
    if len(uniq) < (test_months + early_months + max(gap_months, 0) + 1):
        # Not enough months: let legacy fallback handle
        return (np.arange(len(dates)), np.array([], int), np.array([], int))
    test_lbls = uniq[-test_months:]
    early_end = -test_months
    early_start = early_end - early_months
    early_lbls = uniq[early_start:early_end] if early_months > 0 else np.array([], dtype=uniq.dtype)
    train_hi = early_start - max(gap_months, 0)
    # Support negative slicing semantics to drop last months
    if train_hi == 0:
        train_lbls = uniq[:0]
    else:
        train_lbls = uniq[:train_hi]
    idx = np.arange(len(dates))
    return (
        idx[np.isin(labels, train_lbls)],
        idx[np.isin(labels, early_lbls)],
        idx[np.isin(labels, test_lbls)],
    )

# --- NEW: optional geo enrichment hook ---
def _try_add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    cfg = _geo_env_config()
    if not cfg:
        return df
    try:
        from geo_features import compute_geo_features  # local module

        return compute_geo_features(
            df,
            poi_csv=cfg["poi_csv"],
            lat_col=cfg["lat_col"],
            lon_col=cfg["lon_col"],
            categories=cfg["categories"],
            radii_km=cfg["radii_km"],
            decay_km=cfg["decay_km"],
            max_decay_km=cfg["max_decay_km"],
        )
    except Exception as e:  # pragma: no cover - optional
        logging.warning("Geo features skipped: %s", e)
        return df

# --- NEW: conformalized quantile regression helper ---
def _cqr_qhat(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, alpha: float) -> float:
    s = np.maximum(q_lo - y_true, y_true - q_hi)
    s = np.maximum(s, 0.0)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    return float(np.quantile(s, 1.0 - alpha))

# --- NEW: median-blend helper (EARLY-tuned lambda only) ---
def _apply_median_blend(
    y_early: np.ndarray,
    y_pred_early: np.ndarray,
    w_early: np.ndarray,
    early_median: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    w_test: np.ndarray,
    lambda_max: float = MEDIAN_BLEND_LAMBDA_MAX,
) -> tuple[np.ndarray, float, float]:
    """Return (y_pred_test_unmodified, best_lambda, baseline_TEST_WMAE).

    This helper estimates the best lambda on EARLY only; caller can apply it to TEST
    using their preferred test-side median vector (keeps function pure and testable).
    """
    if early_median is None or len(early_median) != len(y_early):
        base_w = float(np.sum(w_test * np.abs(y_test - y_pred_test)) / np.sum(w_test))
        return y_pred_test, 0.0, base_w
    lambdas = np.linspace(0.0, max(0.0, float(lambda_max)), num=8)
    best_lam = 0.0
    best_w = float(np.sum(w_early * np.abs(y_early - y_pred_early)) / np.sum(w_early))
    for lam in lambdas:
        pe = (1 - lam) * y_pred_early + lam * early_median
        w = float(np.sum(w_early * np.abs(y_early - pe)) / np.sum(w_early))
        if w < best_w:
            best_w, best_lam = w, float(lam)
    base_w = float(np.sum(w_test * np.abs(y_test - y_pred_test)) / np.sum(w_test))
    return y_pred_test, best_lam, base_w

_adv_rs = np.random.RandomState(42)
def _adversarial_auc(X_train, X_early, X_test) -> float:
    """Fast LR AUC probe with optional row cap to keep runtime bounded."""
    def _to_numeric_dense(block):
        if sp.issparse(block):
            try:
                return block.astype(np.float32).toarray()
            except Exception:
                block = block.toarray()
        arr = np.asarray(block)
        if arr.dtype.kind in {"O", "U", "S"}:
            df = pd.DataFrame(arr)
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
            arr = df.to_numpy(dtype=np.float32)
        else:
            arr = arr.astype(np.float32, copy=False)
        return arr

    Xt = _to_numeric_dense(X_train)
    Xe = _to_numeric_dense(X_early)
    Xs = _to_numeric_dense(X_test)
    # Build ADV matrix
    X_adv = sp.vstack([Xt, Xe, Xs]) if sp.issparse(Xt) else np.vstack([Xt, Xe, Xs])
    y_adv = np.concatenate([np.zeros(Xt.shape[0] + Xe.shape[0], dtype=int), np.ones(Xs.shape[0], dtype=int)])
    # Optional subsample
    n = X_adv.shape[0]
    if n > ADV_MAX_ROWS > 0:
        n_tr, n_ea, n_te = Xt.shape[0], Xe.shape[0], Xs.shape[0]
        def take(k, want):
            idx = np.arange(k)
            return idx if k <= want else _adv_rs.choice(idx, size=want, replace=False)
        want_tr = max(1, int(ADV_MAX_ROWS * (n_tr / n)))
        want_ea = max(0, int(ADV_MAX_ROWS * (n_ea / n)))
        want_te = max(1, int(ADV_MAX_ROWS - want_tr - want_ea))
        tr_idx = take(n_tr, want_tr)
        ea_idx = take(n_ea, want_ea)
        te_idx = take(n_te, want_te)
        X_adv = (sp.vstack([Xt[tr_idx], Xe[ea_idx], Xs[te_idx]]) if sp.issparse(Xt)
                 else np.vstack([Xt[tr_idx], Xe[ea_idx], Xs[te_idx]]))
        y_adv = np.concatenate([np.zeros(len(tr_idx) + len(ea_idx), dtype=int), np.ones(len(te_idx), dtype=int)])
    # Faster convergence settings
    clf = LogisticRegression(max_iter=5000, tol=1e-3, solver="lbfgs")
    clf.fit(X_adv, y_adv)
    p = clf.predict_proba(X_adv)[:, 1]
    return float(roc_auc_score(y_adv, p))

def _adversarial_auc_and_weights(X_train, X_early, X_test):
    """Robust adversarial classifier on preprocessed features.

    Returns (auc, w_train, w_early) where weights are clipped importance weights.
    """
    X_adv = sp.vstack([X_train, X_early, X_test]) if sp.issparse(X_train) else np.vstack([X_train, X_early, X_test])
    n_tr, n_ea, n_te = len(X_train), len(X_early), len(X_test)
    y_adv = np.concatenate([np.zeros(n_tr + n_ea, dtype=int), np.ones(n_te, dtype=int)])
    clf = LogisticRegression(
        solver=ADV_LOG_SOLVER, penalty="l2", C=ADV_LOG_C,
        max_iter=ADV_LOG_MAX_ITER, tol=ADV_LOG_TOL,
        class_weight="balanced", n_jobs=-1, fit_intercept=True,
    )
    clf.fit(X_adv, y_adv)
    p_all = clf.predict_proba(X_adv)[:, 1]
    auc = float(roc_auc_score(y_adv, p_all))
    p_tr_ea = np.clip(p_all[: n_tr + n_ea], 1e-6, 1 - 1e-6)
    iw = p_tr_ea / (1.0 - p_tr_ea)
    iw = np.clip(iw.astype(np.float32), ADV_WEIGHT_FLOOR, ADV_WEIGHT_CAP)
    w_tr = iw[:n_tr]
    w_ea = iw[n_tr:] if n_ea else np.empty((0,), dtype=np.float32)
    return auc, w_tr, w_ea

# --- NEW: density-ratio via fast domain classifier (for importance weighting) ---
def _domain_importance_weights(
    X_train, X_early, X_test, *, clip: tuple[float, float] = (0.2, 5.0), temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a quick domain classifier (TRAIN+EARLY vs TEST) and return per-row weights for TRAIN and EARLY:
      w = (p_test / (1 - p_test)) ** (1/temperature), clipped to [clip_low, clip_high]
    """
    # Ensure numeric arrays (dense or sparse) to avoid string conversion errors
    Xt = _coerce_for_adv_lr(X_train)
    Xe = _coerce_for_adv_lr(X_early)
    Xs = _coerce_for_adv_lr(X_test)
    X_teach = sp.vstack([Xt, Xe]) if sp.issparse(Xt) else np.vstack([Xt, Xe])
    X_all = sp.vstack([X_teach, Xs]) if sp.issparse(Xt) else np.vstack([X_teach, Xs])
    y_all = np.concatenate([np.zeros(X_teach.shape[0], dtype=int), np.ones(X_test.shape[0], dtype=int)])
    # Optional subsample for IW to keep runtime bounded
    n_all = X_all.shape[0]
    if n_all > IW_MAX_ROWS > 0:
        n_teach = X_teach.shape[0]
        n_test = X_test.shape[0]
        want_teach = max(2, int(IW_MAX_ROWS * (n_teach / n_all)))
        want_test = max(2, int(IW_MAX_ROWS - want_teach))
        rs = np.random.RandomState(42)
        idx_teach = rs.choice(np.arange(n_teach), size=want_teach, replace=False) if n_teach > want_teach else np.arange(n_teach)
        idx_test = rs.choice(np.arange(n_test), size=want_test, replace=False) if n_test > want_test else np.arange(n_test)
        X_all = (sp.vstack([X_teach[idx_teach], X_test[idx_test]]) if sp.issparse(X_all)
                 else np.vstack([X_teach[idx_teach], X_test[idx_test]]))
        y_all = np.concatenate([np.zeros(len(idx_teach), dtype=int), np.ones(len(idx_test), dtype=int)])
    clf = SGDClassifier(
        loss="log_loss", alpha=1e-4, penalty="l2",
        max_iter=800, early_stopping=True, n_iter_no_change=3, random_state=RANDOM_STATE
    )
    clf.fit(X_all, y_all)
    p = clf.predict_proba(X_teach)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if temperature and temperature != 1.0:
        logit = np.log(p / (1 - p)) / float(temperature)
        p = 1.0 / (1.0 + np.exp(-logit))
    w = p / (1.0 - p)
    w = np.clip(w.astype(np.float64), float(clip[0]), float(clip[1]))
    return w[: X_train.shape[0]], w[X_train.shape[0] :]


def monte_carlo_block_splits(labels: np.ndarray) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_STATE)
    uniq = np.unique(labels)
    n_val_blocks = max(1, int(len(uniq) * BLOCK_VAL_PCT))
    for _ in range(N_BLOCK_FOLDS):
        rng.shuffle(uniq)
        val_blocks = uniq[:n_val_blocks]
        val_mask = np.isin(labels, val_blocks)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        yield train_idx, val_idx


def make_mc_folds(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = list(monte_carlo_block_splits(labels))
    folds.sort(key=lambda tv: tv[1].min())
    return folds


def make_walk_folds(
    labels: np.ndarray,
    val_blocks: int | None = None,
    gap_months: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.sort(np.unique(labels))
    if val_blocks is None:
        val_blocks = max(1, int(len(uniq) * BLOCK_VAL_PCT))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for end in range(val_blocks, len(uniq)):
        val_lbls = uniq[end - val_blocks + 1 : end + 1]
        train_end = end - val_blocks + 1 - gap_months
        train_lbls = uniq[: max(train_end, 0)]
        val_idx = np.where(np.isin(labels, val_lbls))[0]
        train_idx = np.where(np.isin(labels, train_lbls))[0]
        if train_idx.size and val_idx.size:
            folds.append((train_idx, val_idx))
    return folds[-N_BLOCK_FOLDS:] if len(folds) >= N_BLOCK_FOLDS else folds


def apply_gap_days(
    folds: list[tuple[np.ndarray, np.ndarray]],
    dates: pd.Series,
    gap_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Enforce an absolute embargo of `gap_days` between TRAIN and VAL by date."""
    if gap_days <= 0:
        return folds
    out: list[tuple[np.ndarray, np.ndarray]] = []
    dts = pd.to_datetime(dates, errors="coerce")
    for tr_idx, va_idx in folds:
        if len(va_idx) == 0:
            continue
        val_start = dts.iloc[va_idx].min()
        cutoff = val_start - pd.Timedelta(days=gap_days)
        keep = dts.iloc[tr_idx] <= cutoff
        tr_clean = tr_idx[np.where(keep.to_numpy())[0]]
        if tr_clean.size and va_idx.size:
            out.append((tr_clean, va_idx))
    return out


def purge_groups(
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Optional: remove TRAIN rows whose group appears in VAL (prevents near-duplicate leakage)."""
    if groups is None:
        return folds
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in folds:
        gval = set(groups[va_idx])
        keep = ~np.isin(groups[tr_idx], list(gval))
        tr_clean = tr_idx[np.where(keep)[0]]
        if tr_clean.size and va_idx.size:
            out.append((tr_clean, va_idx))
    return out


# ─────────────────────────── Leak helpers ─────────────────────────── #

_HELPER_LEAKS = {"_log_rent", "_log_cap", "rent_shift", "rent_copy"}

def make_leak_checker(target_col: str) -> Callable[[str], bool]:
    tgt_re = re.compile(rf"\b{re.escape(target_col.lower())}\b", flags=re.I)
    def _leaks(col: str) -> bool:
        name = col.lower()
        return name in _HELPER_LEAKS or tgt_re.search(name) is not None
    return _leaks


# ─────────────────────────── Utilities ─────────────────────────── #

def _safe_expm1(x: np.ndarray | float) -> np.ndarray | float:
    return np.expm1(np.clip(x, None, 40.0))  # e^40 ~ 2.35e17
# Helper: Top-K numeric selection via mutual information with log target, with correlation fallback
def _select_topk_numeric(df: pd.DataFrame, y: pd.Series, candidates: list[str], k: int) -> list[str]:
    try:
        X = df[candidates].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
        y_log = np.log1p(pd.to_numeric(y, errors="coerce").fillna(y.median()))
        if _mi_reg is not None:
            mi = _mi_reg(X.values, y_log.values)
            s = pd.Series(mi, index=candidates)
            return list(s.sort_values(ascending=False).head(k).index)
    except Exception:
        pass
    # Fallback to absolute correlation with log target
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs = df[candidates].corrwith(np.log1p(y)).abs().fillna(0)
        return list(corrs.sort_values(ascending=False).head(k).index)
    except Exception:
        return list(candidates[:k])


# Top-level wrapper to allow pickling of log-target models
class LogTargetWrapper:
    def __init__(self, base_model):
        self.base = base_model

    def predict(self, X):
        return _safe_expm1(self.base.predict(X))

    def __getattr__(self, k):
        # Avoid recursion during unpickling by using object.__getattribute__
        base = object.__getattribute__(self, "base")
        return getattr(base, k)

    def __getstate__(self):
        return {"base": self.base}

    def __setstate__(self, state):
        self.base = state["base"]

# Backward-compat alias for older pickled artifacts
class LogExpWrapper(LogTargetWrapper):
    pass

# Simple averaging ensemble on original scale
class AverageEnsemble:
    def __init__(self, models: list[LogTargetWrapper]):
        self.models = models

    def predict(self, X):
        if not self.models:
            return np.zeros((len(X) if hasattr(X, "__len__") else 0,), dtype=float)
        preds = [m.predict(X) for m in self.models]
        # Ensure array conversion and equal-weight averaging
        stack = np.column_stack([np.asarray(p, dtype=float) for p in preds])
        return stack.mean(axis=1)

    # for sklearn Pipeline compatibility; avoid recursion during unpickling
    def __getattr__(self, k):
        try:
            models = object.__getattribute__(self, "models")
        except Exception:
            raise AttributeError(k)
        return getattr(models[0], k)

    def __getstate__(self):
        return {"models": self.models}

    def __setstate__(self, state):
        self.models = state.get("models", [])

def make_weights(y: np.ndarray, cap: float, *, scheme: str = "inv_sqrt") -> np.ndarray:
    """Compute base sample weights.

    scheme:
      - "inv_sqrt": 1/sqrt(clipped y)
      - "uniform": all ones
    """
    if scheme == "uniform":
        return np.ones_like(y, dtype=float)
    w = 1.0 / np.sqrt(np.maximum(np.minimum(y, cap), 1))
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    lo, hi = 0.4, 2.5
    if os.getenv("CLIP_WEIGHTS_RANGE"):
        try:
            lo, hi = [float(v) for v in os.getenv("CLIP_WEIGHTS_RANGE").split(",", 1)]
        except Exception:
            pass
    w = np.clip(w, lo, hi)
    w *= len(w) / max(w.sum(), 1e-12)
    return w


# ───────────────────────── Optuna objective ───────────────────────── #

def tune_catboost(
    df_full: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: str,
    suburb_col: str,
    cv_folds: list[tuple[np.ndarray, np.ndarray]] | None,
    task_type: str,
    scale_numeric: bool,
    *,
    n_trials: int,
    timeout_sec: int,
    leak_checker: Callable[[str], bool],
    use_outlier_scorer: bool,
    outlier_weight_mult: float,
    opt_metric: str,
    cap_value: float | None = None,
) -> Tuple[dict, float]:
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE, multivariate=True, group=True, n_startup_trials=10
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # Enforce time-aware folds aligned with final geometry (walk-forward if provided)
    labels = make_block_labels(df_full[date_col])
    if cv_folds is not None and len(cv_folds) >= 2:
        folds = cv_folds
    else:
        folds = list(TimeSeriesSplit(n_splits=5).split(np.arange(len(df_full))))

    def objective(trial: optuna.trial.Trial) -> float:
        # Common params
        # We always train on log-target with RMSE; unify early-stopping metric to RMSE to be hardware-invariant
        _loss_eval = "RMSE"

        # --- halflife search knob (learn how much to trust older data) ---
        halflife_days = TIME_DECAY_HALFLIFE_DAYS
        if SEARCH_HALFLIFE and hasattr(trial, "suggest_categorical"):
            halflife_days = trial.suggest_categorical(
                "halflife_days", [0, 45, 90, 120, 150, 210]
            )

        _depth = trial.suggest_int("depth", 7, 9)
        _leaf_low = 64 if _depth <= 7 else 96
        params = {
            "iterations": trial.suggest_int("iterations", 2400, 4500),
            "depth": _depth,
            "learning_rate": trial.suggest_float("learning_rate", 0.020, 0.040, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 4.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.6, 1.8),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", _leaf_low, 224),
            "rsm": trial.suggest_float("rsm", 0.55, 0.80),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),

            "loss_function": "RMSE",
            "eval_metric": _loss_eval,
            "od_type": "Iter",
            "od_wait": 200,

            "task_type": task_type,
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False,
            "one_hot_max_size": 12,
        }
        # rsm can be CPU-specific; on GPU prefer improving border_count resolution
        if task_type != "CPU":
            params.pop("rsm", None)
            params["border_count"] = trial.suggest_int("border_count", 128, 254)
        # CPU-safe bootstrap: Bernoulli (supports subsample)
        if task_type == "CPU":
            params["bootstrap_type"] = "Bernoulli"
            params["leaf_estimation_method"] = "Gradient"
            # For very small datasets, disable subsampling to avoid CatBoost errors
            try:
                if int(df_full.shape[0]) < 300:
                    params["subsample"] = max(params.get("subsample", 0.8), 0.9)
            except Exception:
                pass
        else:
            # GPU path: use Bernoulli+subsample to reduce divergence
            params["bootstrap_type"] = "Bernoulli"

        fold_scores: list[float] = []
        for fold_no, (tr_idx, val_idx) in enumerate(folds, 1):
            # time-aware features leak-free
            # Compute time features once using TRAIN history only (avoid leakage across folds)
            df_fold = make_time_features(
                df_full, tr_idx, date_col=date_col, target_col=target_col, suburb_col=suburb_col
            )

            feat_fold = [c for c in df_fold.columns if c not in (target_col, date_col) and not leak_checker(c)]
            num_cols_fold = [c for c in feat_fold if pd.api.types.is_numeric_dtype(df_fold[c])]
            cat_cols_fold = [c for c in feat_fold if c not in num_cols_fold]

            # light Top-K numeric selection to steady CV
            TOP_K_CV = int(os.getenv("CV_TOPK_NUM", "24"))
            if len(num_cols_fold) > TOP_K_CV:
                try:
                    num_cols_fold = _select_topk_numeric(
                        df_fold.iloc[tr_idx], df_fold.iloc[tr_idx][target_col], num_cols_fold, TOP_K_CV
                    )
                except Exception as e:
                    logging.warning("CV Top-K selection (MI) skipped (fold %d): %s", fold_no, e)

            # Prefer log variants for key numerics if available
            try:
                _prefer_log_for = [s.strip() for s in os.getenv(
                    "PREFER_LOG_FOR",
                    "Capital Value,Land Value,Days on Market",
                ).split(",") if s.strip()]
                for _base in _prefer_log_for:
                    _raw = _base
                    _logn = f"log_{_base}"
                    if _raw in num_cols_fold and _logn in df_fold.columns:
                        _idx = num_cols_fold.index(_raw)
                        if _logn not in num_cols_fold:
                            num_cols_fold[_idx] = _logn
                        else:
                            num_cols_fold.pop(_idx)
            except Exception:
                pass
            # Reduce reliance on raw region IDs when robust suburb features are active
            if (DROP_REGION_IDS_IN_CV or (USE_SUBURB_FEATURES and (USE_SUBURB_MEDIANS or USE_SUBURB_LOO))):
                cat_cols_fold = [c for c in cat_cols_fold if c not in {"Suburb", "Postcode"}]

            # Fit preprocessor on fold TRAIN only
            preproc, cat_idx = build_preprocessor(
                df_fold.iloc[tr_idx], num_cols_fold, cat_cols_fold, scale_numeric, date_col=date_col
            )

            X_tr = preproc.fit_transform(df_fold.iloc[tr_idx][num_cols_fold + cat_cols_fold])
            X_va = preproc.transform(df_fold.iloc[val_idx][num_cols_fold + cat_cols_fold])
            X_tr = np.array(X_tr, copy=False)
            X_va = np.array(X_va, copy=False)

            y_tr = df_fold.iloc[tr_idx][target_col].to_numpy()
            y_va = df_fold.iloc[val_idx][target_col].to_numpy()

            # Use provided global cap if available; otherwise use fold TRAIN cap
            cap_tr = float(np.percentile(y_tr, 99.5))
            _cap = cap_value if (cap_value is not None and np.isfinite(cap_value) and cap_value > 0) else cap_tr
            w_tr = make_weights(y_tr, _cap)
            # Keep validation weighting geometry consistent with training/global cap
            w_va = make_weights(y_va, _cap)

            # fold-level outlier weighting (fit only on TRAIN indices of the fold)
            if use_outlier_scorer:
                scorer = OutlierScorer(
                    target_col=target_col, suburb_col=suburb_col, date_col=date_col
                ).fit(df_fold.iloc[tr_idx])
                flags_tr = scorer.transform(df_fold.iloc[tr_idx])["_outlier"].to_numpy()
                flags_va = scorer.transform(df_fold.iloc[val_idx])["_outlier"].to_numpy()
                w_tr = w_tr * np.where(flags_tr, outlier_weight_mult, 1.0)
                w_va = w_va * np.where(flags_va, outlier_weight_mult, 1.0)

            # Always train on log-target for stability (GPU & CPU)
            y_tr_fit = np.log1p(y_tr)
            y_va_fit = np.log1p(y_va)

            # Optional recency weighting (bias toward most recent TRAIN time)
            if halflife_days and halflife_days > 0:
                ref_dt = pd.to_datetime(df_fold.iloc[tr_idx][date_col]).max()
                w_tr = w_tr * _recency_weights(df_fold.iloc[tr_idx][date_col], ref_dt, halflife_days)
                w_va = w_va * _recency_weights(df_fold.iloc[val_idx][date_col], ref_dt, halflife_days)

            # Optional monotonic constraints (guarded behind env flag)
            params_fold = params
            if ENFORCE_MONOTONE:
                try:
                    _mono_pos = {"Bath", "Car", "has_car", "has_two_car", "has_three_plus_car"}
                    mono = [1 if (c in _mono_pos) else 0 for c in num_cols_fold]
                    params_fold = {**params, "monotone_constraints": mono}
                except Exception:
                    params_fold = params

            if CB_THREAD_COUNT > 0:
                params_fold = {**params_fold, "thread_count": CB_THREAD_COUNT}
                model = CatBoostRegressor(**params_fold)
                try:
                    eval_pool = Pool(X_va, y_va_fit, cat_features=cat_idx, weight=w_va)
                except Exception:
                    eval_pool = None
                try:
                    model.fit(
                        X_tr,
                        y_tr_fit,
                        sample_weight=w_tr,
                        cat_features=cat_idx,
                        eval_set=eval_pool,
                        early_stopping_rounds=200,
                        use_best_model=True,
                        verbose=False,
                    )
                except Exception:
                    # Fallback without eval_set for tests/stubs
                    model.fit(
                        X_tr,
                        y_tr_fit,
                        sample_weight=w_tr,
                        cat_features=cat_idx,
                        verbose=False,
                    )

            _pred = model.predict(X_va)
            y_pred = _safe_expm1(_pred)
            metric = opt_metric.lower()
            if metric == "wmae":
                score = float(np.sum(w_va * np.abs(y_va - y_pred)) / np.sum(w_va))
            elif metric == "mae":
                score = float(np.mean(np.abs(y_va - y_pred)))
            elif metric == "rmse":
                score = float(np.sqrt(np.mean((y_va - y_pred) ** 2)))
            elif metric == "mape":
                mask = y_va > 0
                score = float(np.mean(np.abs((y_va[mask] - y_pred[mask]) / y_va[mask]))) if mask.any() else float("inf")
            else:
                # default to weighted MAE for stability
                score = float(np.sum(w_va * np.abs(y_va - y_pred)) / np.sum(w_va))
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    # expose objective for unit tests to hook
    try:
        globals()["_cv_objective_for_test"] = objective
    except Exception:
        pass
    show_bar = os.getenv("OPTUNA_PROGRESS_BAR", "0").lower() in {"1", "true", "yes"}
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None if timeout_sec <= 0 else timeout_sec,
            show_progress_bar=show_bar,
            n_jobs=OPTUNA_N_JOBS,
        )
    except TypeError:
        # Some test doubles/stubs may not accept show_progress_bar kwarg
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=OPTUNA_N_JOBS,
            timeout=None if timeout_sec <= 0 else timeout_sec,
        )
    try:
        logging.info("Optuna best CV-%s %.3f after %d trials", opt_metric.upper(), study.best_value, len(study.trials))
    except Exception:
        logging.info("Optuna CV optimization completed (stubbed study).")
    best = study.best_params
    if "halflife_days" not in best:
        best["halflife_days"] = TIME_DECAY_HALFLIFE_DAYS
    try:
        val = float(study.best_value)
    except Exception:
        val = float("nan")
    return best, val


# ───────────────────────────── Train function ───────────────────────────── #

def train_model(
    df_raw: pd.DataFrame,
    target_col: str = "last_rent_price",
    date_col: str = "last_rent_date",
    suburb_col: str = "suburb",
    *,
    train_frac: float = 0.7,
    valid_frac: float = 0.1,
    max_ohe_card: int = 15,  # kept for API compatibility, unused (CatBoost native cats)
    use_gpu: bool = False,
    scale_numeric: bool = True,
    explain: bool = False,
    block_cv: bool = False,
    skip_featurealgo: bool = False,  # kept for API compatibility (monthly rank is auto-handled)
    trials: int = 8,
    timeout_sec: int = DEFAULT_TIMEOUT,
    dedup_by_address: bool = False,
    opt_loss: str = "wmae",  # reserved for future; internally optimize wMAE now
    final_fit_on_all: bool | None = None,
    removed_tracker: Any | None = None,
) -> dict:
    geo_env_cfg = _geo_env_config()
    logging.info("Step 1/6 – feature engineering + split")
    # Allow environment override for optimization metric (e.g., OPT_LOSS=rmse)
    try:
        _env_opt = os.getenv("OPT_LOSS")
        if _env_opt:
            opt_loss = _env_opt.strip().lower()
    except Exception:
        pass

    # Normalize flexible schema to canonical columns used in training
    try:
        df_raw, target_col, date_col, suburb_col = _normalize_new_schema(
            df_raw,
            target_preference=[target_col] + _ALIASES["target"],
            date_preference=[date_col] + _ALIASES["date_col"],
            suburb_preference=[suburb_col] + _ALIASES["suburb_col"],
        )
    except Exception:
        # Fallback: attempt to coerce provided names directly
        pass

    df_raw = df_raw.copy(deep=False)
    if date_col in df_raw.columns:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        nat_mask = df_raw[date_col].isna()
        if nat_mask.any():
            removed_ids = df_raw.loc[nat_mask, "__row_id"].astype(int, copy=False).tolist() if "__row_id" in df_raw.columns else []
            _record_removed_rows(removed_tracker, "train_drop_invalid_dates", df_raw, removed_ids)
            logging.warning("Dropping %d rows with invalid %s (NaT)", nat_mask.sum(), date_col)
            df_raw = df_raw.loc[~nat_mask].copy()
        # Sort chronologically for stability
        try:
            df_raw = df_raw.sort_values(by=[date_col]).copy()
        except Exception:
            pass

    # Optional date-range filter to align CV with recent TEST distribution
    try:
        if os.getenv("DATE_MIN"):
            cutoff = pd.to_datetime(os.getenv("DATE_MIN"), errors="coerce")
            if pd.notna(cutoff):
                mask = df_raw[date_col] >= cutoff
                removed_ids = df_raw.loc[~mask, "__row_id"].astype(int, copy=False).tolist() if "__row_id" in df_raw.columns else []
                _record_removed_rows(removed_tracker, "train_date_min_filter", df_raw, removed_ids)
                removed = int((~mask).sum())
                df_raw = df_raw.loc[mask].copy()
                logging.info("DATE_MIN filter %s removed %d rows", cutoff.date(), removed)
        elif os.getenv("RECENT_YEARS"):
            yrs = int(os.getenv("RECENT_YEARS", "0"))
            if yrs > 0:
                cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=yrs)
                mask = df_raw[date_col] >= cutoff
                removed_ids = df_raw.loc[~mask, "__row_id"].astype(int, copy=False).tolist() if "__row_id" in df_raw.columns else []
                _record_removed_rows(removed_tracker, "train_recent_years_filter", df_raw, removed_ids)
                removed = int((~mask).sum())
                df_raw = df_raw.loc[mask].copy()
                logging.info("RECENT_YEARS=%d filter (cutoff %s) removed %d rows", yrs, cutoff.date(), removed)
    except Exception:
        pass

    # Ensure numeric target; drop rows with NaN targets (post-cleaning safeguard)
    if not pd.api.types.is_numeric_dtype(df_raw[target_col]):
        df_raw[target_col] = pd.to_numeric(df_raw[target_col], errors="coerce")
    mask_target_na = df_raw[target_col].isna()
    if mask_target_na.any():
        removed_ids = df_raw.loc[mask_target_na, "__row_id"].astype(int, copy=False).tolist() if "__row_id" in df_raw.columns else []
        _record_removed_rows(removed_tracker, "train_drop_non_numeric_target", df_raw, removed_ids)
        logging.warning("Dropping %d rows with non-numeric %s", mask_target_na.sum(), target_col)
        df_raw = df_raw.loc[~mask_target_na].copy()

    # Safety guard: enforce target domain even if cleaning was bypassed
    try:
        _tmin = float(os.getenv("TARGET_MIN", "0"))
        _tmax = float(os.getenv("TARGET_MAX", "2000"))
        mask_range = (df_raw[target_col] >= _tmin) & (df_raw[target_col] <= _tmax)
        if mask_range.any():
            removed_ids = df_raw.loc[~mask_range, "__row_id"].astype(int, copy=False).tolist() if "__row_id" in df_raw.columns else []
            if removed_ids:
                logging.warning("Safety target filter removed %d rows outside [%.0f, %.0f]", len(removed_ids), _tmin, _tmax)
                _record_removed_rows(removed_tracker, "train_target_domain_filter", df_raw, removed_ids)
            df_raw = df_raw.loc[mask_range].copy()
        else:
            logging.warning("Safety target filter skipped: would remove all rows")
    except Exception:
        pass

    if "__row_id" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw.drop(columns=["__row_id"], inplace=True)

    # Clean strings + ordinals + date parts
    df = _apply_ordinals(_clean_strings(df_raw))
    df = add_date_parts(df, date_col)
    df = _try_add_geo_features(df)  # NEW: optional geo enrichment (no-op if not configured)
    df = df.drop(columns=[c for c in df.columns if c.lower() in _HELPER_LEAKS], errors="ignore")

     # Ensure log versions exist for key heavy‑tailed numerics used downstream
    try:
        _prefer_log_for = [s.strip() for s in os.getenv(
            "PREFER_LOG_FOR",
            "Capital Value,Land Value,Days on Market",
        ).split(",") if s.strip()]
        for _base in _prefer_log_for:
            if _base in df.columns:
                _col = pd.to_numeric(df[_base], errors="coerce").clip(lower=0)
                df[f"log_{_base}"] = np.log1p(_col)
    except Exception:
        pass

    # Safe log augment for heavy-tailed numerics (avoid double-logging)
    base_feats = [c for c in df.columns if c not in (target_col, date_col)]
    num_raw = [c for c in base_feats if pd.api.types.is_numeric_dtype(df[c]) and not str(c).startswith("log_")]
    _skews = {}
    for col in num_raw:
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        try:
            _skews[col] = abs(skew(df[col].dropna()))
        except Exception:
            pass
    for col in pd.Series(_skews)[pd.Series(_skews) > 2.0].index:
        # Do not log-transform angular features; log(bearing) is not meaningful
        if str(col).startswith("bearing_"):
            continue
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Lightweight, leak-safe engineered features to improve RMSE/R²
    try:
        # Age in years
        if "Year Built" in df.columns:
            _yr = pd.to_numeric(df["Year Built"], errors="coerce")
            _cur_year = pd.to_datetime(df[date_col], errors="coerce").dt.year
            df["AgeYears"] = (_cur_year - _yr).clip(lower=0).astype(float)
            df["is_newer_2010"] = (_yr >= 2010).astype("Int8")
        # NOTE: Avoid cap_per_sqm/log_cap_per_sqm (confounds small vs large properties)
        # Lot coverage ratio
        if ("Floor Size (sqm)" in df.columns) and ("Land Size (sqm)" in df.columns):
            flr = pd.to_numeric(df["Floor Size (sqm)"], errors="coerce").clip(lower=0)
            land = pd.to_numeric(df["Land Size (sqm)"], errors="coerce").clip(lower=0)
            df["floor_land_ratio"] = (flr / land.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        # Bed/bath ratio (stability guarded)
        if ("Bed" in df.columns) and ("Bath" in df.columns):
            bed = pd.to_numeric(df["Bed"], errors="coerce").clip(lower=0)
            bath = pd.to_numeric(df["Bath"], errors="coerce").clip(lower=0)
            df["bed_bath_ratio"] = (bed / bath.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
        # Car capacity flags to prevent negative association from rarity effects
        if "Car" in df.columns:
            carn = pd.to_numeric(df["Car"], errors="coerce").fillna(0)
            df["has_car"] = (carn >= 1).astype("Int8")
            df["has_two_car"] = (carn >= 2).astype("Int8")
            df["has_three_plus_car"] = (carn >= 3).astype("Int8")
    except Exception:
        pass

    # Sort chronologically
    df = df.sort_values(date_col).reset_index(drop=True)

    # Optional address deduplication (safeguard against duplicates)
    if dedup_by_address:
        addr_cols = [c for c in df.columns if c.lower() in {"address", "street address", "street", "full_address"}]
        if addr_cols:
            key_cols = [addr_cols[0], date_col]
            before = len(df)
            df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
            logging.info("Dedup by address+date removed %d duplicate rows", before - len(df))
        else:
            logging.info("Dedup requested but no address-like column found; skipping")

    # Block CV geometry & final split
    if block_cv:
        labels = make_block_labels(df[date_col])
        if os.getenv("WALK_FORWARD_CV", "1").lower() in {"1", "true", "yes"}:
            # Resolve gaps: GAP_DAYS > 0 → day embargo; else GAP_MONTHS; else auto derive day-gap
            gap_m = int(os.getenv("GAP_MONTHS", "0"))
            gap_d = GAP_DAYS if GAP_DAYS > 0 else (0 if gap_m > 0 else _auto_gap_days())
            folds_glob = make_walk_folds(labels, gap_months=(0 if gap_d > 0 else gap_m))
            # Optional group purge using address-like column if present
            grp_arr = None
            for cand in ("Address", "Street Address", "Full_Address", "property_id"):
                if cand in df.columns:
                    grp_arr = df[cand].astype("string").to_numpy()
                    break
            folds_glob = purge_groups(folds_glob, grp_arr)
            # Absolute day-level embargo (explicit or auto)
            folds_glob = apply_gap_days(folds_glob, df[date_col], gap_d)
            logging.info(
                "Block CV: walk-forward → %d folds (gap_months=%d, gap_days=%d, group_purged=%s)",
                len(folds_glob), (0 if gap_d > 0 else gap_m), gap_d, "yes" if grp_arr is not None else "no"
            )
        else:
            folds_glob = make_mc_folds(labels)
            logging.info("Block CV: Monte-Carlo (%d folds)", len(folds_glob))
    else:
        folds_glob = []

    # Prefer deterministic 12M TEST split if requested
    if TEST_HOLDOUT_MONTHS > 0:
        tr_i, ea_i, te_i = split_holdout_by_months(
            df[date_col],
            test_months=TEST_HOLDOUT_MONTHS,
            early_months=EARLY_HOLDOUT_MONTHS,
            gap_months=GAP_MONTHS_OVERRIDE,
        )
        if (te_i.size == 0) or (ea_i.size > 0 and ea_i.size < MIN_EARLY_ROWS) or (te_i.size < MIN_TEST_ROWS):
            # Not enough months → gracefully fall back to simple chronological split
            n_rows = len(df)
            train_end = max(1, int(n_rows * train_frac))
            early_end = min(max(train_end + 1, int(n_rows * (train_frac + valid_frac))), n_rows - 1)
            train_idx = np.arange(0, train_end, dtype=int)
            early_idx = np.arange(train_end, early_end, dtype=int)
            test_idx = np.arange(early_end, n_rows, dtype=int)
            cv_folds = None
        else:
            train_idx, early_idx, test_idx = tr_i, ea_i, te_i
            # Build walk-forward CV strictly before EARLY/TEST
            labels_all = make_block_labels(df[date_col])
            folds_all = make_walk_folds(labels_all, gap_months=GAP_MONTHS_OVERRIDE)
            folds_all = apply_gap_days(folds_all, df[date_col], GAP_DAYS if GAP_DAYS > 0 else _auto_gap_days())
            cutoff = int(np.min(early_idx)) if early_idx.size else int(np.min(test_idx))
            cv_folds = [(tr, va) for (tr, va) in folds_all if va.size and va.min() < cutoff]
    elif not folds_glob or len(folds_glob) < 3:
        # Fallback simple chronological split
        n_rows = len(df)
        train_end = max(1, int(n_rows * train_frac))
        early_end = min(max(train_end + 1, int(n_rows * (train_frac + valid_frac))), n_rows - 1)
        train_idx = np.arange(0, train_end, dtype=int)
        early_idx = np.arange(train_end, early_end, dtype=int)
        test_idx = np.arange(early_end, n_rows, dtype=int)
        cv_folds = None
    else:
        # Use disjoint EARLY/TEST blocks; TRAIN strictly before earliest EARLY
        early_idx = np.unique(np.concatenate([folds_glob[-3][1], folds_glob[-2][1]])).astype(int)
        test_idx = np.unique(folds_glob[-1][1]).astype(int)
        train_idx = np.unique(folds_glob[-3][0]).astype(int)
        cutoff = int(np.min(early_idx)) if early_idx.size else len(df)
        train_idx = train_idx[train_idx < cutoff]
        # Tail-align CV folds to the last window before EARLY/TEST
        cv_last_k = int(os.getenv("CV_LAST_K_FOLDS", "6"))
        start = max(0, len(folds_glob) - 3 - cv_last_k)
        cv_folds = folds_glob[start : len(folds_glob) - 3]

    # Make leak-safe time features using TRAIN history only
    df_time = make_time_features(df, train_idx, date_col=date_col, target_col=target_col, suburb_col=suburb_col)
    if df_time.empty:
        df_time = df.copy()
    # Leak-safe momentum feature: short vs long suburb median
    try:
        if USE_SUBURB_MEDIANS and "Suburb90dMedian" in df_time.columns and "Suburb365dMedian" in df_time.columns:
            den = pd.to_numeric(df_time["Suburb365dMedian"], errors="coerce").replace(0, np.nan)
            num = pd.to_numeric(df_time["Suburb90dMedian"], errors="coerce")
            df_time["SuburbMomentum90v365"] = (num / den).replace([np.inf, -np.inf], np.nan)
    except Exception as _e:
        logging.warning("Suburb momentum feature skipped: %s", _e)
    train_idx = train_idx[(train_idx >= 0) & (train_idx < len(df_time))]
    early_idx = early_idx[(early_idx >= 0) & (early_idx < len(df_time))]
    test_idx = test_idx[(test_idx >= 0) & (test_idx < len(df_time))]
    train_df = df_time.iloc[train_idx].copy()
    early_df = df_time.iloc[early_idx].copy()
    test_df = df_time.iloc[test_idx].copy()
    if test_df.empty and len(df_time) > 0:
        fallback_test = len(df_time) - 1
        test_idx = np.array([fallback_test], dtype=int)
        train_idx = train_idx[train_idx < fallback_test]
        early_idx = early_idx[early_idx < fallback_test]
        train_df = df_time.iloc[train_idx].copy()
        early_df = df_time.iloc[early_idx].copy()
        test_df = df_time.iloc[test_idx].copy()
    if train_df.empty and len(df_time) > 0:
        train_idx = np.array([0], dtype=int)
        train_df = df_time.iloc[train_idx].copy()

    # Dataset hash for artifact names
    ds_hash = hashlib.sha256(np.sort(df_raw[target_col].values).tobytes()).hexdigest()[:8]

    # Quick diagnostics
    for nm, ser in (("TRAIN", train_df[target_col]), ("EARLY", early_df[target_col]), ("TEST", test_df[target_col])):
        logging.info(
            "%s rent | n=%d | min=%.2f | p25=%.2f | p50=%.2f | p75 %.2f | p95 %.2f | max=%.2f",
            nm, len(ser), ser.min(), ser.quantile(0.25), ser.median(), ser.quantile(0.75), ser.quantile(0.95), ser.max()
        )

    # Build feature sets (now that LOO exists)
    leak_check = make_leak_checker(target_col)
    feature_cols = [c for c in train_df.columns if c not in (target_col, date_col) and not leak_check(c)]
    # Remove meta/string-heavy columns that should never enter the model
    for bad in ("street_address", "__geo_query__"):
        if bad in feature_cols:
            feature_cols.remove(bad)
    # Remove high-cardinality ID-like columns that add noise (address-level)
    feature_cols = [c for c in feature_cols if c not in {"Street Address", "Address", "Full_Address"}]
    # Ensure key suburb context features are included (gated by flags)
    if USE_SUBURB_FEATURES:
        always: list[str] = []
        if USE_SUBURB_MEDIANS:
            always.extend(["SuburbBed90dMedian", "Suburb90dMedian"])
            if USE_SUBURB_365D_MEDIAN:
                always.extend(["Suburb365dMedian", "SuburbBed365dMedian"])  # include bed-conditional 365D
        if USE_SUBURB_LOO:
            always.extend(["Suburb_LOO", "SuburbMonth_LOO"])
        for must in always:
            if must in train_df.columns and must not in feature_cols:
                feature_cols.append(must)
    num_all = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    # Ensure bed-normalized size kept
    if "floor_to_bed_med" in train_df.columns and "floor_to_bed_med" not in num_all:
        num_all.append("floor_to_bed_med")
    TOP_K_NUM = int(os.getenv("FINAL_TOPK_NUM", "28"))
    # Remove engineered logs we never want the model to learn from directly
    _forbidden_logs = {"log_Car", "log_Bath", "log_Garage", "log_Garage Parks", "log_garage_parks"}
    if len(num_all) > TOP_K_NUM:
        nonconst = [c for c in num_all if train_df[c].nunique(dropna=True) > 1]
        num_cols = _select_topk_numeric(train_df, train_df[target_col], nonconst, TOP_K_NUM)
        # Filter out low-signal or misleading engineered logs from core numeric list
        num_cols = [c for c in num_cols if c not in _forbidden_logs]
        # ensure LOO present if numeric
        for extra in ("Suburb_LOO", "SuburbMonth_LOO"):
            if extra in num_all and extra not in num_cols:
                num_cols.append(extra)
        # ensure bed-normalized size feature is kept
        if "floor_to_bed_med" in num_all and "floor_to_bed_med" not in num_cols:
            num_cols.append("floor_to_bed_med")
        # ensure raw Bath retained alongside ratio if available
        if "Bath" in num_all and "Bath" not in num_cols:
            num_cols.append("Bath")
        # ensure raw Garage Parks retained if available (numeric garage capacity)
        if "Garage Parks" in num_all and "Garage Parks" not in num_cols:
            num_cols.append("Garage Parks")
    else:
        num_cols = [c for c in num_all if c not in _forbidden_logs]
        if "floor_to_bed_med" in num_all and "floor_to_bed_med" not in num_cols:
            num_cols.append("floor_to_bed_med")
        if "Bath" in num_all and "Bath" not in num_cols:
            num_cols.append("Bath")
        if "Garage Parks" in num_all and "Garage Parks" not in num_cols:
            num_cols.append("Garage Parks")

    # Optionally exclude all geo bearings from numeric block (user toggle)
    if EXCLUDE_GEO_BEARING:
        before_n = len(num_cols)
        num_cols = [c for c in num_cols if not str(c).startswith("bearing_") and not str(c).startswith("log_bearing_")]
        if len(num_cols) != before_n:
            logging.info("Excluded geo bearings from numeric features (EXCLUDE_GEO_BEARING=1)")

    # Ensure critical geo features are always kept if present (prevents Top-K trimming)
    try:
        if ALWAYS_KEEP_GEO:
            kept = []
            for key in ALWAYS_KEEP_GEO:
                # exact match or suffix match (e.g., cnt_school_primary_4000m)
                present = [c for c in num_all if str(c).lower() == key or str(c).lower().endswith(key)]
                for c in present:
                    if c in num_all and c not in num_cols:
                        num_cols.append(c)
                        kept.append(c)
            if kept:
                logging.info("Always-kept geo features: %s", ", ".join(sorted(set(map(str, kept)))))
    except Exception:
        pass

    # Prefer log versions over raw for heavy‑tailed numerics to avoid duplicate signals
    prefer_log_for = [s.strip() for s in os.getenv(
        "PREFER_LOG_FOR",
        "Capital Value,Land Size (sqm),Floor Size (sqm),Days on Market",
    ).split(",") if s.strip()]
    for base in prefer_log_for:
        raw = base
        logn = f"log_{base}"
        if raw in num_cols and logn in num_cols:
            num_cols.remove(raw)
    cat_cols = [c for c in feature_cols if c not in num_cols and not pd.api.types.is_numeric_dtype(train_df[c])]
    # Drop leak-prone, high-cardinality categoricals (configurable)
    try:
        if EXCLUDE_CAT_FEATURES:
            _exc = set(EXCLUDE_CAT_FEATURES)
            _before = set(cat_cols)
            cat_cols = [c for c in cat_cols if c.lower() not in _exc]
            _dropped = list(sorted(_before - set(cat_cols)))
            if _dropped:
                logging.info("Excluded categoricals: %s", ", ".join(_dropped))
    except Exception:
        pass
    # Drop low-entropy categorical features to reduce noise and spurious importances
    try:
        cat_cols, dropped_low_entropy = _drop_low_entropy_cats_df(train_df, cat_cols)
        if dropped_low_entropy:
            logging.info("Dropped low-entropy categoricals: %s", ", ".join(sorted(dropped_low_entropy)))
    except Exception:
        pass

    # (optional) teacher→residual booster: add TeacherPred numeric feature
    teacher_pred = None
    if ENABLE_RESIDUAL_BOOSTER and TEACHER_PIPELINE:
        try:
            t_obj = joblib.load(TEACHER_PIPELINE)
            if isinstance(t_obj, dict) and "preprocessor" in t_obj and "model" in t_obj:
                class _T:
                    def __init__(self, o):
                        self.pre = o["preprocessor"]
                        self.mod = o["model"]

                    def predict(self, X):
                        return self.mod.predict(self.pre.transform(X))

                teacher = _T(t_obj)
            else:
                teacher = t_obj
            train_df["TeacherPred"] = teacher.predict(train_df[num_cols + cat_cols])
            early_df["TeacherPred"] = teacher.predict(early_df[num_cols + cat_cols])
            test_df["TeacherPred"] = teacher.predict(test_df[num_cols + cat_cols])
            if "TeacherPred" not in num_cols:
                num_cols.append("TeacherPred")
        except Exception as e:  # pragma: no cover - optional
            logging.warning("Residual booster skipped: %s", e)
            teacher_pred = None

    # Preprocessor
    try:
        cat_cols = _drop_raw_date_from_cats(list(cat_cols), date_col)
    except Exception:
        pass
    preproc, cat_idx = build_preprocessor(train_df, num_cols, cat_cols, scale_numeric, date_col=date_col)
    X_train = np.array(preproc.fit_transform(train_df[num_cols + cat_cols]), copy=False)
    X_early = np.array(preproc.transform(early_df[num_cols + cat_cols]), copy=False)
    X_test = np.array(preproc.transform(test_df[num_cols + cat_cols]), copy=False)

    y_train = train_df[target_col].to_numpy()
    y_early = early_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    # Preserve row order indices for later alignment of flags/weights
    idx_train = train_df.index.to_numpy()
    idx_early = early_df.index.to_numpy()
    idx_test = test_df.index.to_numpy()

    # TRAIN-fit OutlierScorer and apply to all splits (assign by position to avoid duplicate-index joins)
    if ENABLE_OUTLIER_SCORER:
        scorer = OutlierScorer(target_col=target_col, suburb_col=suburb_col, date_col=date_col).fit(train_df)
        of_train = scorer.transform(train_df)
        of_early = scorer.transform(early_df)
        of_test = scorer.transform(test_df)
        for col in ("_outlier", "_z_resid", "_z_pps"):
            if col in of_train.columns:
                train_df[col] = of_train[col].to_numpy()
            if col in of_early.columns:
                early_df[col] = of_early[col].to_numpy()
            if col in of_test.columns:
                test_df[col] = of_test[col].to_numpy()
        # artifacts: per-split flags
        for nm, _df in (("train", train_df), ("early", early_df), ("test", test_df)):
            _cols = [c for c in ["_outlier", "_z_resid", "_z_pps", target_col, suburb_col, date_col] if c in _df.columns]
            (_df[_cols]).to_csv(ARTIFACT_DIR / f"outlier_flags_{nm}_{ds_hash}.csv", index=False)
    else:
        train_df["_outlier"] = False
        early_df["_outlier"] = False
        test_df["_outlier"] = False

    # Step 2: Optuna tuning on CV folds with fold-level outlier weighting
    # Prefer runtime GPU detection; fallback to env hints
    gpu_env = _gpu_available() or bool(
        os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("NVIDIA_VISIBLE_DEVICES") or os.path.exists("/dev/nvidia0")
    )
    task_type = "GPU" if use_gpu and gpu_env else "CPU"
    if use_gpu and not gpu_env:
        logging.warning("GPU flag set but no GPU detected; falling back to CPU.")
    logging.info("Step 2/6 – Optuna tuning: trials=%d (task=%s)", trials, task_type)
    # Use a stable global cap derived from TRAIN to keep fold weighting consistent
    try:
        _cv_cap = float(np.percentile(train_df[target_col].to_numpy(), 99.5))
        if not np.isfinite(_cv_cap) or _cv_cap <= 0:
            _cv_cap = float(np.nanmax(train_df[target_col].to_numpy()))
    except Exception:
        _cv_cap = None
    best_params, cv_wmae = tune_catboost(
        df_full=df,
        feature_cols=feature_cols,
        target_col=target_col,
        date_col=date_col,
        suburb_col=suburb_col,
        cv_folds=cv_folds,
        task_type=task_type,
        scale_numeric=scale_numeric,
        n_trials=trials,
        timeout_sec=timeout_sec,
        leak_checker=leak_check,
        use_outlier_scorer=ENABLE_OUTLIER_SCORER,
        outlier_weight_mult=OUTLIER_WEIGHT_MULT,
        opt_metric=opt_loss,
        cap_value=_cv_cap,
    )

    # Recompute CV metrics under best params for apples-to-apples drift checks
    cv_metrics = {"CV_WMAE": float("nan"), "CV_WRMSE": float("nan"), "CV_MAE": float("nan"), "CV_RMSE": float("nan")}
    try:
        if cv_folds and len(cv_folds) >= 2:
            cv_true_all: list[np.ndarray] = []
            cv_pred_all: list[np.ndarray] = []
            cv_w_all: list[np.ndarray] = []
            for tr_idx, va_idx in cv_folds:
                df_fold = make_time_features(
                    df, tr_idx, date_col=date_col, target_col=target_col, suburb_col=suburb_col
                )
                feat_fold = [c for c in df_fold.columns if c not in (target_col, date_col) and not leak_check(c)]
                num_cols_fold = [c for c in feat_fold if pd.api.types.is_numeric_dtype(df_fold[c])]
                cat_cols_fold = [c for c in feat_fold if c not in num_cols_fold]

                preproc_fold, cat_idx_fold = build_preprocessor(
                    df_fold.iloc[tr_idx], num_cols_fold, cat_cols_fold, scale_numeric, date_col=date_col
                )
                X_tr_f = np.array(preproc_fold.fit_transform(df_fold.iloc[tr_idx][num_cols_fold + cat_cols_fold]), copy=False)
                X_va_f = np.array(preproc_fold.transform(df_fold.iloc[va_idx][num_cols_fold + cat_cols_fold]), copy=False)
                y_tr_f = df_fold.iloc[tr_idx][target_col].to_numpy()
                y_va_f = df_fold.iloc[va_idx][target_col].to_numpy()

                cap_tr_f = np.percentile(y_tr_f, 99.5)
                w_va_f = make_weights(y_va_f, cap_tr_f)

                params_eval = {**best_params, "task_type": task_type, "allow_writing_files": False, "loss_function": "RMSE", "eval_metric": "RMSE"}
                model_f = CatBoostRegressor(**params_eval)
                model_f.fit(
                    X_tr_f,
                    np.log1p(y_tr_f),
                    cat_features=cat_idx_fold,
                    eval_set=Pool(X_va_f, np.log1p(y_va_f), cat_features=cat_idx_fold),
                    early_stopping_rounds=200,
                    use_best_model=True,
                    verbose=False,
                )
                y_pred_va = _safe_expm1(model_f.predict(X_va_f))
                cv_true_all.append(y_va_f)
                cv_pred_all.append(y_pred_va)
                cv_w_all.append(w_va_f)

            y_cv_true = np.concatenate(cv_true_all)
            y_cv_pred = np.concatenate(cv_pred_all)
            w_cv = np.concatenate(cv_w_all)
            cv_metrics["CV_MAE"] = float(np.mean(np.abs(y_cv_true - y_cv_pred)))
            try:
                cv_metrics["CV_RMSE"] = float(mean_squared_error(y_cv_true, y_cv_pred, squared=False))
            except TypeError:
                cv_metrics["CV_RMSE"] = float(np.sqrt(mean_squared_error(y_cv_true, y_cv_pred)))
            cv_metrics["CV_WMAE"] = float(np.sum(w_cv * np.abs(y_cv_true - y_cv_pred)) / np.sum(w_cv))
            cv_metrics["CV_WRMSE"] = float(np.sqrt(np.sum(w_cv * (y_cv_true - y_cv_pred) ** 2) / np.sum(w_cv)))
    except Exception as _cve:
        logging.warning("CV metrics computation skipped: %s", _cve)

    # If no block CV, synthesize simple rolling folds for CV metrics
    if (not cv_folds) or len(cv_folds) < 2:
        try:
            labels_all = make_block_labels(df[date_col])
            cv_folds = make_walk_folds(labels_all, gap_months=0)
            cv_folds = apply_gap_days(cv_folds, df[date_col], GAP_DAYS if GAP_DAYS > 0 else 0)
        except Exception:
            cv_folds = []
    # Final safety: if still empty, use a simple chronological TimeSeriesSplit
    if (not cv_folds) or len(cv_folds) < 1:
        try:
            cv_folds = list(TimeSeriesSplit(n_splits=5).split(np.arange(len(df))))
        except Exception:
            cv_folds = []

    # Optional: Kaggle-style OOF R² on original and log scales (using tuned params)
    oof_r2 = None
    oof_r2_log = None
    if cv_folds and len(cv_folds) >= 2:
        try:
            oof_y_true_list: list[np.ndarray] = []
            oof_y_pred_list: list[np.ndarray] = []
            for fold_no, (tr_idx, val_idx) in enumerate(cv_folds, 1):
                df_fold = make_time_features(
                    df, tr_idx, date_col=date_col, target_col=target_col, suburb_col=suburb_col
                )

                feat_fold = [c for c in df_fold.columns if c not in (target_col, date_col) and not leak_check(c)]
                num_cols_fold = [c for c in feat_fold if pd.api.types.is_numeric_dtype(df_fold[c])]
                cat_cols_fold = [c for c in feat_fold if c not in num_cols_fold]

                TOP_K_CV = int(os.getenv("CV_TOPK_NUM", "18"))
                if len(num_cols_fold) > TOP_K_CV:
                    try:
                        tgt_log_cv = np.log1p(df_fold.iloc[tr_idx][target_col])
                        corrs_fold = (
                            df_fold.iloc[tr_idx][num_cols_fold].corrwith(tgt_log_cv).abs().fillna(0)
                        )
                        num_cols_fold = list(corrs_fold.sort_values(ascending=False).head(TOP_K_CV).index)
                    except Exception:
                        pass

                preproc_fold, cat_idx_fold = build_preprocessor(
                    df_fold.iloc[tr_idx], num_cols_fold, cat_cols_fold, scale_numeric, date_col=date_col
                )

                X_tr_f = np.array(preproc_fold.fit_transform(df_fold.iloc[tr_idx][num_cols_fold + cat_cols_fold]), copy=False)
                X_va_f = np.array(preproc_fold.transform(df_fold.iloc[val_idx][num_cols_fold + cat_cols_fold]), copy=False)
                y_tr_f = df_fold.iloc[tr_idx][target_col].to_numpy()
                y_va_f = df_fold.iloc[val_idx][target_col].to_numpy()

                # Weights + outlier down-weighting consistent with tuning
                cap_tr_f = np.percentile(y_tr_f, 99.5)
                w_tr_f = make_weights(y_tr_f, cap_tr_f)
                if ENABLE_OUTLIER_SCORER:
                    try:
                        scorer_f = OutlierScorer(target_col=target_col, suburb_col=suburb_col, date_col=date_col).fit(df_fold.iloc[tr_idx])
                        flags_tr_f = scorer_f.transform(df_fold.iloc[tr_idx])["_outlier"].to_numpy()
                        w_tr_f = w_tr_f * np.where(flags_tr_f, OUTLIER_WEIGHT_MULT, 1.0)
                    except Exception:
                        pass

                _bp = {k: v for k, v in best_params.items() if k != "halflife_days"}
                model_f = CatBoostRegressor(**{**_bp, "task_type": task_type, "allow_writing_files": False})
                model_f.fit(
                    X_tr_f,
                    np.log1p(y_tr_f),
                    sample_weight=w_tr_f,
                    cat_features=cat_idx_fold,
                    eval_set=Pool(X_va_f, np.log1p(y_va_f), cat_features=cat_idx_fold),
                    early_stopping_rounds=200,
                    use_best_model=True,
                    verbose=False,
                )
                y_pred_va = _safe_expm1(model_f.predict(X_va_f))
                oof_y_true_list.append(y_va_f)
                oof_y_pred_list.append(y_pred_va)

            y_oof = np.concatenate(oof_y_true_list)
            yhat_oof = np.concatenate(oof_y_pred_list)
            oof_r2 = float(r2_score(y_oof, yhat_oof))
            # Log-scale R² (guard against negatives)
            y_oof_log = np.log1p(y_oof)
            yhat_oof_log = np.log1p(np.clip(yhat_oof, a_min=0.0, a_max=None))
            oof_r2_log = float(r2_score(y_oof_log, yhat_oof_log))
            logging.info("OOF R² (orig): %.3f | OOF R² (log): %.3f", oof_r2, oof_r2_log)

            # Fallback CV metrics from OOF aggregates if missing
            try:
                cap_oof = np.percentile(y_oof, 99.5)
                if not np.isfinite(cap_oof) or cap_oof <= 0:
                    cap_oof = float(np.nanmax(y_oof)) if np.isfinite(np.nanmax(y_oof)) else 1.0
                w_oof = make_weights(y_oof, cap_oof)
                if not np.isfinite(cv_metrics.get("CV_WMAE", np.nan)):
                    cv_metrics["CV_WMAE"] = float(np.sum(w_oof * np.abs(y_oof - yhat_oof)) / np.sum(w_oof))
                if not np.isfinite(cv_metrics.get("CV_MAE", np.nan)):
                    cv_metrics["CV_MAE"] = float(np.mean(np.abs(y_oof - yhat_oof)))
                if not np.isfinite(cv_metrics.get("CV_RMSE", np.nan)):
                    cv_metrics["CV_RMSE"] = float(np.sqrt(np.mean((y_oof - yhat_oof) ** 2)))
                if not np.isfinite(cv_metrics.get("CV_WRMSE", np.nan)):
                    cv_metrics["CV_WRMSE"] = float(np.sqrt(np.sum(w_oof * (y_oof - yhat_oof) ** 2) / np.sum(w_oof)))
            except Exception:
                pass
        except Exception as e:
            logging.warning("OOF R² computation skipped: %s", e)

    # Step 3: final fit
    # Before we possibly include EARLY in training, report a clean pre-final-fit snapshot (TRAIN only)
    try:
        _cap_tr_snap = np.percentile(y_train, 99.5)
        if not np.isfinite(_cap_tr_snap) or _cap_tr_snap <= 0:
            _cap_tr_snap = float(np.nanmax(y_train)) if np.isfinite(np.nanmax(y_train)) else 1.0
        _w_train_snap = make_weights(y_train, _cap_tr_snap, scheme=os.getenv("WEIGHT_SCHEME", "inv_sqrt"))
        _snap_model = CatBoostRegressor(**{**best_params, "task_type": task_type, "allow_writing_files": False})
        _snap_model.fit(X_train, np.log1p(y_train), sample_weight=_w_train_snap, cat_features=cat_idx, verbose=False)
        _pre_early = _safe_expm1(_snap_model.predict(X_early))
        _pre_test = _safe_expm1(_snap_model.predict(X_test))
        _pre_early_mae = mean_absolute_error(y_early, _pre_early)
        _pre_test_mae = mean_absolute_error(y_test, _pre_test)
        _cap_early_snap = np.percentile(y_early, 99.5) if y_early.size else 1.0
        if not np.isfinite(_cap_early_snap) or _cap_early_snap <= 0:
            _cap_early_snap = float(np.nanmax(y_early)) if y_early.size and np.isfinite(np.nanmax(y_early)) else 1.0
        _pre_early_w = make_weights(y_early, _cap_early_snap)
        _pre_early_wmae = float(np.sum(_pre_early_w * np.abs(y_early - _pre_early)) / max(np.sum(_pre_early_w), 1e-9))
        _cap_test_snap = np.percentile(y_test, 99.5) if y_test.size else 1.0
        if not np.isfinite(_cap_test_snap) or _cap_test_snap <= 0:
            _cap_test_snap = float(np.nanmax(y_test)) if y_test.size and np.isfinite(np.nanmax(y_test)) else 1.0
        _pre_test_w = make_weights(y_test, _cap_test_snap)
        _pre_test_wmae = float(np.sum(_pre_test_w * np.abs(y_test - _pre_test)) / np.sum(_pre_test_w))
        logging.info(
            "Pre-final-fit snapshot | EARLY-MAE %.3f | EARLY-wMAE %.3f | TEST-MAE %.3f | TEST-wMAE %.3f (TRAIN only)",
            _pre_early_mae, _pre_early_wmae, _pre_test_mae, _pre_test_wmae
        )
    except Exception:
        pass
    logging.info("Step 3/6 – final fit")
    cap_global = np.percentile(y_train, 99.5)
    if not np.isfinite(cap_global) or cap_global <= 0:
        cap_global = float(np.nanmax(y_train)) if np.isfinite(np.nanmax(y_train)) else 1.0

    # Base weights
    w_scheme = os.getenv("WEIGHT_SCHEME", "inv_sqrt")
    # If optimizing RMSE, uniform weights often improve RMSE generalization
    if os.getenv("FINAL_OBJECTIVE", "").lower() == "rmse":
        w_scheme = "uniform"
    w_train = make_weights(y_train, cap_global, scheme=w_scheme)
    w_early = make_weights(y_early, cap_global, scheme=w_scheme)

    # Outlier down-weighting
    if ENABLE_OUTLIER_SCORER:
        flags_train_arr = train_df["_outlier"].reindex(idx_train).fillna(False).to_numpy()
        flags_early_arr = early_df["_outlier"].reindex(idx_early).fillna(False).to_numpy()
        w_train = w_train * np.where(flags_train_arr, OUTLIER_WEIGHT_MULT, 1.0)
        w_early = w_early * np.where(flags_early_arr, OUTLIER_WEIGHT_MULT, 1.0)

    # Optional: drop outliers on TRAIN (keeps EARLY weighted)
    if ENABLE_OUTLIER_SCORER and OUTLIER_DROP_TRAIN:
        keep = ~train_df["_outlier"].astype(bool).to_numpy()
        X_train = X_train[keep]
        y_train = y_train[keep]
        w_train = w_train[keep]
        logging.info("Dropped %d TRAIN outliers (kept %d)", (~keep).sum(), keep.sum())

    # Combine TRAIN+EARLY for final model if desired
    if final_fit_on_all is None:
        TRAIN_ONLY = os.getenv("TRAIN_ONLY_FINAL_FIT", "1").lower() in {"1", "true", "yes"}
    else:
        TRAIN_ONLY = not bool(final_fit_on_all)
    if TRAIN_ONLY:
        X_final = X_train
        y_final = np.log1p(y_train)
        eval_X, eval_y, eval_w = X_early, np.log1p(y_early), w_early
        w_final = w_train
        early_in_train = False
    else:
        X_final = sp.vstack([X_train, X_early]) if sp.issparse(X_train) else np.concatenate([X_train, X_early])
        y_final = np.log1p(np.concatenate([y_train, y_early]))
        eval_X, eval_y, eval_w = X_test, np.log1p(y_test), None
        w_final = np.concatenate([w_train, w_early])
        early_in_train = True

    # Map final loss:
    # Force RMSE on GPU (MAE not implemented). On CPU, honor FINAL_OBJECTIVE when set,
    # otherwise use MAE for MAE/wMAE tuning or RMSE by default.
    # Final objective to monitor; training stays on log-target with RMSE
    _final_loss = "MAE" if (os.getenv("FINAL_OBJECTIVE", "").lower() == "mae" or opt_loss.lower() in {"mae","wmae"}) else "RMSE"
    # Monotonic constraints for final fit (apply to numeric block only) – optional
    mono_final = None
    if ENFORCE_MONOTONE:
        try:
            _mono_pos_final = {"Bath", "Car", "has_car", "has_two_car", "has_three_plus_car"}
            mono_final = [1 if (c in _mono_pos_final) else 0 for c in num_cols]
        except Exception:
            mono_final = None

    # Choose base params for final fit: tuned vs production baseline (toggle via FINAL_USE_TUNED)
    _final_use_tuned = os.getenv("FINAL_USE_TUNED", "0").lower() in {"1", "true", "yes"}
    _base_final = dict(best_params) if _final_use_tuned and isinstance(best_params, dict) else dict(BEST_PARAMS_PRODUCTION)

    # Use chosen baseline for final fit
    final_params = {
        **_base_final,
        # Train with RMSE on log-target for hardware invariance; monitor requested metric
        "loss_function": "RMSE",
        "eval_metric": _final_loss,
        **({"monotone_constraints": mono_final} if mono_final is not None else {}),
        "random_seed": RANDOM_STATE,
        "task_type": task_type,
        "allow_writing_files": False,
        # Ensure CPU-safe bootstrap
        "bootstrap_type": "Bernoulli",
        "one_hot_max_size": 12,
        # Early-stop safety for long runs (align with fit() early_stopping_rounds)
        "od_type": "Iter",
        "od_wait": int(FINAL_ES_ROUNDS),
    }
    # Drop non-CB params
    final_params.pop("halflife_days", None)
    # Clamp iterations by MAX_FINAL_ITER (env). Set this to 5000 to allow longer final fits.
    try:
        it = int(final_params.get("iterations", 2000))
        final_params["iterations"] = int(min(MAX_FINAL_ITER, it))
    except Exception:
        final_params["iterations"] = min(final_params.get("iterations", 2000), MAX_FINAL_ITER)
    if task_type == "CPU":
        final_params["leaf_estimation_method"] = "Gradient"
        # For small training sets, disable subsampling to avoid CatBoost sampling errors
        try:
            n_train_final = X_train.shape[0] + (X_early.shape[0] if early_in_train else 0)
            if int(n_train_final) < 300:
                final_params["bootstrap_type"] = "No"
                final_params.pop("subsample", None)
        except Exception:
            pass

    # Prepare eval weights consistent with training geometry (when eval set exists)
    if eval_X is not None:
        if early_in_train:
            # Eval on TEST
            _w_eval = make_weights(y_test, cap_global)
            if ENABLE_OUTLIER_SCORER:
                _flags = test_df["_outlier"].astype(bool).to_numpy()
                _w_eval = _w_eval * np.where(_flags, OUTLIER_WEIGHT_MULT, 1.0)
        else:
            # Eval on EARLY
            _w_eval = eval_w  # already includes outlier weighting
        eval_pool = Pool(eval_X, eval_y, cat_features=cat_idx, weight=_w_eval)
    else:
        eval_pool = None

    # Increase CPU utilization: allow more threads if env set
    th = int(os.getenv("CB_THREAD_COUNT", "0"))
    if th <= 0 and CB_THREAD_COUNT > 0:
        th = CB_THREAD_COUNT
    if th > 0:
        final_params["thread_count"] = th
    if task_type != "CPU":
        final_params.pop("rsm", None)
    # Optional recency weighting for final fit as well
    if TIME_DECAY_HALFLIFE_DAYS > 0:
        try:
            ref_dt_final = pd.to_datetime(train_df[date_col]).max()
            w_train = w_train * _recency_weights(train_df[date_col], ref_dt_final, TIME_DECAY_HALFLIFE_DAYS)
            w_early = w_early * _recency_weights(early_df[date_col], ref_dt_final, TIME_DECAY_HALFLIFE_DAYS)
            w_final = w_train if not early_in_train else np.concatenate([w_train, w_early])
        except Exception:
            pass

    # Optional: importance-weighted final fit (mitigate drift)
    domain_importance = {"enabled": False}
    try:
        if ENABLE_IMPORTANCE_WEIGHTED_FIT:
            try:
                _adv_for_iw = _adversarial_auc(X_train, X_early, X_test)
            except Exception:
                _adv_for_iw = None
            if (_adv_for_iw is not None) and (_adv_for_iw >= IW_ENABLE_ON_ADV_AUC):
                w_tr_iw, w_ea_iw = _domain_importance_weights(
                    X_train, X_early, X_test, clip=(IW_CLIP_LOW, IW_CLIP_HIGH), temperature=IW_TEMP
                )
                w_train = w_train * w_tr_iw
                if early_in_train:
                    w_early = w_early * w_ea_iw
                w_final = w_train if not early_in_train else np.concatenate([w_train, w_early])
                domain_importance = {
                    "enabled": True,
                    "clip": [float(IW_CLIP_LOW), float(IW_CLIP_HIGH)],
                    "temperature": float(IW_TEMP),
                    "mean_weight": float(np.mean(w_final)),
                    "adv_auc": float(_adv_for_iw),
                }
                logging.info("Importance-weighted fit enabled (mean w=%.3f)", domain_importance["mean_weight"])
    except Exception as e:
        logging.warning("Importance weighting skipped: %s", e)

    # If optimizing RMSE, reduce outlier down-weighting impact in final fitting stage
    if os.getenv("FINAL_OBJECTIVE", "").lower() == "rmse" and ENABLE_OUTLIER_SCORER:
        try:
            w_final = np.asarray(w_final, dtype=float)
            w_final = np.clip(w_final, np.min(w_final)*0.9, np.max(w_final))
        except Exception:
            pass

    # Train 1..K seeds and build original-scale averaging ensemble
    models: list[LogTargetWrapper] = []
    seeds = [RANDOM_STATE + i * 97 for i in range(max(1, ENSEMBLE_SIZE))]
    for sd in seeds:
        fp = {**final_params, "random_seed": sd}
        base = CatBoostRegressor(**fp)
        base.fit(
            np.array(X_final, copy=False),
            y_final,
            sample_weight=w_final,
            cat_features=cat_idx,
            eval_set=eval_pool,
            early_stopping_rounds=FINAL_ES_ROUNDS,
            use_best_model=True,
            verbose=False,
        )
        models.append(LogTargetWrapper(base))

    model = models[0] if len(models) == 1 else AverageEnsemble(models)

    # Optional: quantile models for prediction intervals (original target scale), with conformal calibration
    q_paths: dict[str, str] = {}
    cqr: dict | None = None
    if SAVE_QUANTILE_MODELS and QUANTILE_ALPHAS:
        try:
            q_models: dict[float, CatBoostRegressor] = {}
            # y_final is on log scale; convert back when applicable
            y_final_orig = np.expm1(y_final) if np.any(y_final < 40) else y_train
            for a in QUANTILE_ALPHAS:
                qp = {k: v for k, v in final_params.items() if k not in {"loss_function", "eval_metric", "monotone_constraints"}}
                qp.update({"loss_function": f"Quantile:alpha={float(a)}", "eval_metric": f"Quantile:alpha={float(a)}"})
                qb = CatBoostRegressor(**qp)
                qb.fit(
                    np.array(X_final, copy=False),
                    y_final_orig,
                    sample_weight=w_final,
                    cat_features=cat_idx,
                    verbose=False,
                )
                q_models[float(a)] = qb
                q_path = ARTIFACT_DIR / f"pipeline_q{int(round(float(a)*100))}_{ds_hash}.joblib"
                joblib.dump({"preprocessor": preproc, "model": qb}, q_path, compress=3)
                q_paths[str(float(a))] = str(q_path)
            if ENABLE_CQR and (y_early.size > 0) and any(a < 0.5 for a in QUANTILE_ALPHAS) and any(a > 0.5 for a in QUANTILE_ALPHAS):
                a_lo = max(a for a in QUANTILE_ALPHAS if a < 0.5)
                a_hi = min(a for a in QUANTILE_ALPHAS if a > 0.5)
                qlo = q_models[float(a_lo)].predict(X_early)
                qhi = q_models[float(a_hi)].predict(X_early)
                qhat = _cqr_qhat(y_early, qlo, qhi, alpha=PI_ALPHA)
                cqr = {"alpha_pair": [float(a_lo), float(a_hi)], "pi_alpha": PI_ALPHA, "qhat": float(qhat)}
        except Exception as e:  # pragma: no cover - optional path
            logging.warning("Quantile/CQR step skipped: %s", e)

    # Step 4: evaluation
    logging.info("Step 4/6 – evaluation")
    y_pred_test = model.predict(X_test)
    y_pred_early = model.predict(X_early)
    if CLAMP_PRED_TO_TARGET_RANGE:
        try:
            _tmin = float(os.getenv("TARGET_MIN", "0"))
            _tmax = float(os.getenv("TARGET_MAX", "2000"))
            y_pred_test = np.clip(y_pred_test, _tmin, _tmax)
            y_pred_early = np.clip(y_pred_early, _tmin, _tmax)
        except Exception:
            pass

    # Weighted metrics (include outlier down-weighting for fairness)
    def _wmae(y_true, y_pred, w_base, flags) -> float:
        w = w_base * (np.where(flags, OUTLIER_WEIGHT_MULT, 1.0) if ENABLE_OUTLIER_SCORER else 1.0)
        return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))

    def _wrmse(y_true, y_pred, w_base, flags) -> float:
        w = w_base * (np.where(flags, OUTLIER_WEIGHT_MULT, 1.0) if ENABLE_OUTLIER_SCORER else 1.0)
        return float(np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w)))

    early_mae = mean_absolute_error(y_early, y_pred_early)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    w_test = make_weights(y_test, cap_global)
    flags_early = early_df["_outlier"].astype(bool).to_numpy()
    flags_test = test_df["_outlier"].astype(bool).to_numpy()

    early_wmae = _wmae(y_early, y_pred_early, w_early, flags_early)
    test_wmae = _wmae(y_test, y_pred_test, w_test, flags_test)
    early_wrmse = _wrmse(y_early, y_pred_early, w_early, flags_early)
    test_wrmse = _wrmse(y_test, y_pred_test, w_test, flags_test)

    try:
        rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Robust MAPE (avoid zero-division)
    mape_mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mape_mask] - y_pred_test[mape_mask]) / y_test[mape_mask])) * 100) if mape_mask.any() else float("nan")

    def _nan_to_none(v):
        try:
            return None if (v is None or (isinstance(v, float) and (np.isnan(v)))) else v
        except Exception:
            return v

    metrics = {
        "RMSE": rmse,
        "R2": r2_score(y_test, y_pred_test),
        "MAPE": _safe_mape(y_test, y_pred_test, w=w_test),
        "CV_WMAE": _nan_to_none(float(cv_metrics.get("CV_WMAE", float("nan")))),
        "CV_WRMSE": _nan_to_none(float(cv_metrics.get("CV_WRMSE", float("nan")))),
        "CV_MAE": _nan_to_none(float(cv_metrics.get("CV_MAE", float("nan")))),
        "CV_RMSE": _nan_to_none(float(cv_metrics.get("CV_RMSE", float("nan")))),
        "EARLY_WMAE": float("nan") if early_in_train else early_wmae,
        "TEST_WMAE": test_wmae,
        "EARLY_MAE": float("nan") if early_in_train else early_mae,
        "TEST_MAE": test_mae,
        "EARLY_WRMSE": float("nan") if early_in_train else early_wrmse,
        "TEST_WRMSE": test_wrmse,
        # Kaggle-style diagnostics
        "OOF_R2": _nan_to_none(oof_r2 if oof_r2 is not None else float("nan")),
        "OOF_R2_LOG": _nan_to_none(oof_r2_log if oof_r2_log is not None else float("nan")),
    }

    # Leak-safe post-fit blend: toward suburb medians, tuned on EARLY and applied on TEST
    blend = {"lambda": 0.0, "pre_TEST_WMAE": metrics["TEST_WMAE"], "post_TEST_WMAE": metrics["TEST_WMAE"]}
    try:
        if ENABLE_MEDIAN_BLEND:
            med_early = (
                "SuburbBed365dMedian"
                if "SuburbBed365dMedian" in early_df.columns
                else ("Suburb90dMedian" if "Suburb90dMedian" in early_df.columns else None)
            )
            med_test = (
                "SuburbBed365dMedian"
                if "SuburbBed365dMedian" in test_df.columns
                else ("Suburb90dMedian" if "Suburb90dMedian" in test_df.columns else None)
            )
            if med_early and med_test:
                lambdas = np.linspace(0.0, MEDIAN_BLEND_LAMBDA_MAX, num=8)
                best_lam, best_w = 0.0, early_wmae
                med_e = pd.to_numeric(early_df[med_early], errors="coerce").to_numpy()
                for lam in lambdas:
                    pe = (1 - lam) * y_pred_early + lam * med_e
                    _w = float(np.sum(w_early * np.abs(y_early - pe)) / np.sum(w_early))
                    if _w < best_w:
                        best_w, best_lam = _w, float(lam)
                if best_lam > 0:
                    med_t = pd.to_numeric(test_df[med_test], errors="coerce").to_numpy()
                    pt = (1 - best_lam) * y_pred_test + best_lam * med_t
                    test_wmae_blend = float(np.sum(w_test * np.abs(y_test - pt)) / np.sum(w_test))
                    blend = {
                        "lambda": best_lam,
                        "pre_TEST_WMAE": metrics["TEST_WMAE"],
                        "post_TEST_WMAE": test_wmae_blend,
                    }
                    y_pred_test = pt
                    metrics["TEST_WMAE"] = test_wmae_blend
                    metrics["TEST_MAE"] = mean_absolute_error(y_test, y_pred_test)
    except Exception as e:  # pragma: no cover - optional
        logging.warning("Median blend skipped: %s", e)
    # Adversarial validation + covariate-shift importance weighting on preprocessed features
    adv_auc: float | None = None
    try:
        if ENABLE_ADVERSARIAL_VALID:
            adv_auc, iw_tr, iw_ea = _adversarial_auc_and_weights(X_train, X_early, X_test)
            if adv_auc > ADV_AUC_WARN:
                logging.warning("Adversarial AUC=%.3f → notable TRAIN/EARLY→TEST shift", adv_auc)
                # Apply importance weights to TRAIN (+EARLY if used in final fit)
                try:
                    if len(w_train) == len(iw_tr):
                        w_train = (w_train.astype(np.float32) * iw_tr).astype(np.float32)
                    if early_in_train and len(w_early) == len(iw_ea):
                        w_early = (w_early.astype(np.float32) * iw_ea).astype(np.float32)
                    logging.info("Applied covariate-shift importance weighting (clip=[%.2f, %.2f]).", ADV_WEIGHT_FLOOR, ADV_WEIGHT_CAP)
                except Exception as _e:
                    logging.warning("Importance weighting skipped: %s", _e)
            if AUTO_DRIFT_SHRINK and adv_auc >= ADV_AUC_STRONG:
                logging.warning("Drift strong (AUC=%.3f). Consider DRIFT_RECENT_YEARS or DATE_MIN to narrow TRAIN window.", adv_auc)
    except Exception as e:  # pragma: no cover
        logging.warning("Adversarial validation skipped: %s", e)
    cv_disp = metrics["CV_WMAE"] if isinstance(metrics["CV_WMAE"], (float, int)) else float("nan")
    if early_in_train:
        logging.info(
            "CV-%s %s | TEST-wMAE %.3f | raw-MAE %.0f | R² %.3f (EARLY used in final training; early metrics suppressed)",
            opt_loss.upper(), (f"{cv_disp:.3f}" if np.isfinite(cv_disp) else "nan"), metrics["TEST_WMAE"], metrics["TEST_MAE"], metrics["R2"],
        )
    else:
        ew = metrics["EARLY_WMAE"] if isinstance(metrics["EARLY_WMAE"], (float, int)) else float("nan")
        logging.info(
            "CV-%s %s | EARLY-wMAE %s | TEST-wMAE %.3f | raw-MAE %.0f | R² %.3f",
            opt_loss.upper(), (f"{cv_disp:.3f}" if np.isfinite(cv_disp) else "nan"), (f"{ew:.3f}" if np.isfinite(ew) else "nan"), metrics["TEST_WMAE"], metrics["TEST_MAE"], metrics["R2"]
        )

    # Step 5: drift guard (compare CV vs TEST on the same objective)
    DRIFT_TOL = float(os.getenv("DRIFT_TOL", os.getenv("MAE_DRIFT_TOL", "0.30")))
    if DRIFT_METRIC == "rmse":
        cv_key, te_key = "CV_WRMSE", "TEST_WRMSE"
    elif DRIFT_METRIC == "mae":
        cv_key, te_key = "CV_MAE", "TEST_MAE"
    else:
        cv_key, te_key = "CV_WMAE", "TEST_WMAE"
    try:
        drift = abs(metrics[cv_key] - metrics[te_key]) / max(metrics[cv_key], 1e-9)
    except Exception:
        drift = float("nan")
    if np.isfinite(drift) and drift > DRIFT_TOL:
        logging.warning("⚠️ %s drift %.1f%% exceeds tol %.0f%%", DRIFT_METRIC.upper(), drift * 100, DRIFT_TOL * 100)
        # Continue execution but highlight potential leakage; do not raise to allow diagnostics artifacts

    # Step 6: artifacts
    pipe_path = ARTIFACT_DIR / f"pipeline_{ds_hash}.joblib"
    meta_path = ARTIFACT_DIR / f"meta_{ds_hash}.json"
    imp_path = ARTIFACT_DIR / f"importance_{ds_hash}.csv"
    shap_path = ARTIFACT_DIR / f"shap_{ds_hash}.png" if explain else None
    med90_path = ARTIFACT_DIR / f"med90_{ds_hash}.csv"
    enc_path = ARTIFACT_DIR / f"encoders_{ds_hash}.joblib"
    rank_path = ARTIFACT_DIR / f"ranker_{ds_hash}.json"

    # Feature importance
    try:
        feat_names = list(num_cols) + list(cat_cols)
        importances_vec = base.get_feature_importance(type="FeatureImportance")
        if len(feat_names) != len(importances_vec):
            feat_names = getattr(base, "feature_names_", None) or [f"f{i}" for i in range(len(importances_vec))]
        importances = pd.DataFrame({"feature": feat_names, "gain": importances_vec}).sort_values("gain", ascending=False)
        importances.to_csv(imp_path, index=False)
    except Exception as e:
        logging.warning("Feature importance failed: %s", e)
        pd.DataFrame({"feature": [], "gain": []}).to_csv(imp_path, index=False)

    # Save split datasets for audit
    try:
        train_df.assign(__split="train").to_csv(ARTIFACT_DIR / f"train_df_{ds_hash}.csv", index=False)
        early_df.assign(__split="early").to_csv(ARTIFACT_DIR / f"early_df_{ds_hash}.csv", index=False)
        test_df.assign(__split="test").to_csv(ARTIFACT_DIR / f"test_df_{ds_hash}.csv", index=False)
    except Exception as e:
        logging.warning("Saving split datasets failed: %s", e)

    # Persist a unified sklearn Pipeline for serve-time (train=serve)
    try:
        sk_pipeline = Pipeline([
            ("preprocessor", preproc),
            ("model", model),  # LogTargetWrapper exposes predict() in original scale
        ])
        joblib.dump(sk_pipeline, pipe_path, compress=3)
    except Exception:
        # Fallback to dict format; if that fails (e.g., unpicklable Dummy model in tests), skip persistence
        try:
            joblib.dump({"preprocessor": preproc, "model": model, "cat_idx": cat_idx}, pipe_path, compress=3)
        except Exception:
            logging.warning("Skipping pipeline persistence due to non-picklable model.")

    # Build and save serve-time bundle: rolling medians (TRAIN-only, shifted), LOO encoders (if available), ranker meta
    try:
        # Rolling 90D median per suburb time series (TRAIN rows only, shift to avoid same-day leakage)
        med_src = (
            train_df[[suburb_col, date_col, target_col]]
            .rename(columns={target_col: "Last Rental Price", date_col: "Date", suburb_col: "Suburb"})
            .dropna(subset=["Last Rental Price"])  # safety
        )
        med_src["Date"] = pd.to_datetime(med_src["Date"], errors="coerce")
        med_src = med_src.sort_values(["Suburb", "Date"])  
        def _roll_g(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("Date")["Last Rental Price"]
            s_med = _roll_median(s, f"{SUBURB_MED_WINDOW_DAYS}D", SUBURB_MED_MIN_COUNT)
            out = s_med.reset_index().rename(columns={"Last Rental Price": "Suburb90dMedian"})
            out["Suburb"] = g["Suburb"].iloc[0]
            return out
        med_all = med_src.groupby("Suburb", dropna=False, group_keys=False).apply(_roll_g)
        med_all = med_all[["Suburb", "Date", "Suburb90dMedian"]]
        med_all.to_csv(med90_path, index=False)
        # 90D median per suburb+bed (TRAIN-only)
        if "Bed" in train_df.columns:
            med_bed_src = (
                train_df[[suburb_col, "Bed", date_col, target_col]]
                .rename(columns={target_col: "Last Rental Price", date_col: "Date", suburb_col: "Suburb"})
                .dropna(subset=["Last Rental Price"])  # safety
            )
            med_bed_src["Date"] = pd.to_datetime(med_bed_src["Date"], errors="coerce")
            med_bed_src["Bed"] = med_bed_src["Bed"].astype("string")
            med_bed_src = med_bed_src.sort_values(["Suburb", "Bed", "Date"])  
            def _roll_gb(g: pd.DataFrame) -> pd.DataFrame:
                s = g.set_index("Date")["Last Rental Price"]
                s_med = _roll_median(s, f"{SUBURB_MED_WINDOW_DAYS}D", SUBURB_BED_MIN_COUNT)
                out = s_med.reset_index().rename(columns={"Last Rental Price": "SuburbBed90dMedian"})
                out["Suburb"] = g["Suburb"].iloc[0]
                out["Bed"] = g["Bed"].iloc[0]
                return out
            med_bed_all = med_bed_src.groupby(["Suburb", "Bed"], dropna=False, group_keys=False).apply(_roll_gb)
            med_bed_all = med_bed_all.drop_duplicates(subset=["Suburb", "Bed", "Date"], keep="last")
            med90_bed_path = ARTIFACT_DIR / f"med90_bed_{ds_hash}.csv"
            med_bed_all = med_bed_all[["Suburb", "Bed", "Date", "SuburbBed90dMedian"]]
            med_bed_all.to_csv(med90_bed_path, index=False)
        else:
            med90_bed_path = None

        # 365D medians (TRAIN-only) when flag is enabled
        if USE_SUBURB_MEDIANS and USE_SUBURB_365D_MEDIAN:
            med365_src = med_src.copy()
            def _roll_g365(g: pd.DataFrame) -> pd.DataFrame:
                s = g.set_index("Date")["Last Rental Price"]
                s_med = _roll_median(s, "365D", SUBURB_MED365_MIN_COUNT)
                out = s_med.reset_index().rename(columns={"Last Rental Price": "Suburb365dMedian"})
                out["Suburb"] = g["Suburb"].iloc[0]
                return out
            med365_all = med365_src.groupby("Suburb", dropna=False, group_keys=False).apply(_roll_g365)
            med365_all = med365_all.drop_duplicates(subset=["Suburb", "Date"], keep="last")
            med365_path = ARTIFACT_DIR / f"med365_{ds_hash}.csv"
            med365_all = med365_all[["Suburb", "Date", "Suburb365dMedian"]]
            med365_all.to_csv(med365_path, index=False)
            if "Bed" in train_df.columns:
                med365_bed_src = med_bed_src.copy()
                def _roll_gb365(g: pd.DataFrame) -> pd.DataFrame:
                    s = g.set_index("Date")["Last Rental Price"]
                    s_med = _roll_median(s, "365D", SUBURB_MED365_MIN_COUNT)
                    out = s_med.reset_index().rename(columns={"Last Rental Price": "SuburbBed365dMedian"})
                    out["Suburb"] = g["Suburb"].iloc[0]
                    out["Bed"] = g["Bed"].iloc[0]
                    return out
                med365_bed_all = med365_bed_src.groupby(["Suburb", "Bed"], dropna=False, group_keys=False).apply(_roll_gb365)
                med365_bed_all = med365_bed_all.drop_duplicates(subset=["Suburb", "Bed", "Date"], keep="last")
                med365_bed_path = ARTIFACT_DIR / f"med365_bed_{ds_hash}.csv"
                med365_bed_all = med365_bed_all[["Suburb", "Bed", "Date", "SuburbBed365dMedian"]]
                med365_bed_all.to_csv(med365_bed_path, index=False)
            else:
                med365_bed_path = None
        else:
            med365_path = None
            med365_bed_path = None
    except Exception as e:
        logging.warning("Serve bundle medians failed: %s", e)
        med90_path = None
        med90_bed_path = None
        med365_path = None
        med365_bed_path = None

    enc_saved = False
    try:
        if SuburbLOOEncoder is not None:
            enc_sub = SuburbLOOEncoder(col=suburb_col, sigma=0.1, random_state=RANDOM_STATE)
            enc_sub.fit(train_df[[suburb_col]], train_df[target_col])
        else:
            enc_sub = None
        if SuburbMonthLOOEncoder is not None:
            enc_sm = SuburbMonthLOOEncoder(suburb_col=suburb_col, date_col=date_col, sigma=0.1, random_state=RANDOM_STATE)
            enc_sm.fit(train_df[[suburb_col, date_col]], train_df[target_col])
        else:
            enc_sm = None
        if enc_sub is not None or enc_sm is not None:
            joblib.dump({"suburb_loo": enc_sub, "suburb_month_loo": enc_sm}, enc_path, compress=3)
            enc_saved = True
    except Exception as e:
        logging.warning("Serve bundle encoders failed: %s", e)
        enc_saved = False

    # Persist ranker meta (stateless) for discovery
    try:
        with open(rank_path, "w") as rp:
            json.dump({"type": "MonthlyRankingTransformer", "date_col": str(date_col), "price_col": str(target_col)}, rp)
    except Exception as e:
        logging.warning("Ranker meta save failed: %s", e)
        rank_path = None

    # Optional SHAP (grouped)
    if explain:
        try:
            import shap  # type: ignore
            expl = shap.TreeExplainer(base)
            X_test_d = np.array(X_test, copy=False)
            shap_vals = expl.shap_values(X_test_d)
            base_map = [*num_cols, *cat_cols]
            # crude grouping by base column
            grouped = {}
            for i, b in enumerate(base_map):
                grouped.setdefault(b, []).append(shap_vals[:, i])
            import numpy as _np
            grouped_vals = _np.column_stack([_np.abs(_np.column_stack(v)).mean(axis=1) for v in grouped.values()])
            import matplotlib.pyplot as plt
            shap.summary_plot(grouped_vals, feature_names=list(grouped.keys()), show=False)
            plt.tight_layout(); plt.savefig(shap_path, dpi=150); plt.close()
        except Exception as e:
            logging.warning("SHAP failed: %s", e)
            shap_path = None

    with open(meta_path, "w") as fp:
        json.dump(
            {
                "metrics": metrics,
                "best_params": best_params,
                "quantile_pipelines": q_paths if q_paths else None,
                "conformal": cqr,
                "adversarial_auc": adv_auc,
                "adv_auc_warn": ADV_AUC_WARN,
                "blend": blend,
                "domain_importance": domain_importance,
                "num_features": list(num_cols),
                "cat_features": list(cat_cols),
                "rows": {"train": int(len(train_df)), "early": int(len(early_df)), "test": int(len(test_df))},
                "importance_csv": str(imp_path),
                "pipeline": str(pipe_path),
                "shap_png": str(shap_path) if shap_path else None,
                "serve_bundle": {
                    "med90_csv": (str(med90_path) if med90_path else None),
                    "med90_bed_csv": (str(med90_bed_path) if ("med90_bed_path" in locals() and med90_bed_path) else None),
                    "med365_csv": (str(med365_path) if ("med365_path" in locals() and med365_path) else None),
                    "med365_bed_csv": (str(med365_bed_path) if ("med365_bed_path" in locals() and med365_bed_path) else None),
                    "encoders_joblib": (str(enc_path) if enc_saved else None),
                    "ranker_meta": (str(rank_path) if rank_path else None),
                },
                "flags": {
                    "enable_outlier_scorer": ENABLE_OUTLIER_SCORER,
                    "outlier_weight_mult": OUTLIER_WEIGHT_MULT,
                    "drop_train_outliers": OUTLIER_DROP_TRAIN,
                },
                "geo": (
                    {
                        "poi_csv": geo_env_cfg["poi_csv"],
                        "lat_col": geo_env_cfg["lat_col"],
                        "lon_col": geo_env_cfg["lon_col"],
                        "radii_km": list(geo_env_cfg["radii_km"]),
                        "decay_km": geo_env_cfg["decay_km"],
                        "max_decay_km": geo_env_cfg["max_decay_km"],
                        "categories": list(geo_env_cfg["categories"]) if geo_env_cfg["categories"] else None,
                    }
                ) if geo_env_cfg else None,
                "teacher_pipeline": TEACHER_PIPELINE if TEACHER_PIPELINE else None,
            },
            fp,
            indent=2,
        )
    logging.info("Artefacts saved → %s", pipe_path)
    try:
        with open(meta_path, "r") as _mf:
            meta_obj = json.load(_mf)
    except Exception:
        meta_obj = {}
    try:
        if "MAE" not in meta_obj and isinstance(meta_obj.get("metrics"), dict):
            meta_obj["MAE"] = meta_obj["metrics"].get("TEST_MAE")
    except Exception:
        pass
    return {"pipeline": str(pipe_path), "metadata": meta_obj, "importance": str(imp_path), "shap": shap_path, "model": model}


# ──────────────────────────────── CLI ──────────────────────────────── #

def cli() -> None:
    p = argparse.ArgumentParser(description="Train CatBoost rental model • v7.2")
    p.add_argument("csv_path")
    p.add_argument("--target", default="Rent")
    p.add_argument("--date", default="Date")
    p.add_argument("--suburb", default="Suburb")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--no-scale", action="store_true")
    p.add_argument("--no-block-cv", action="store_true", help="Disable monthly block CV (fallback to TimeSeriesSplit)")
    p.add_argument("--skip-featurealgo", action="store_true", help="(kept for API compat)")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--max-ohe-card", type=int, default=15)  # kept for API compat
    p.add_argument("--explain", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--walk-cv", action="store_true", help="Use walk-forward CV instead of Monte-Carlo")
    p.add_argument("--gap-months", type=int, default=0)
    p.add_argument("--gap-days", type=int, default=0, help="Absolute embargo in days between TRAIN and VAL")
    p.add_argument("--dedup-by-address", action="store_true")
    p.add_argument("--trials", type=int, help="Override Optuna trials (default from env or 200)")
    p.add_argument("--timeout-sec", type=int, help="Optuna timeout (0=unlimited)")
    args = p.parse_args()

    _setup_logging(args.log_level)
    if args.walk_cv:
        os.environ["WALK_FORWARD_CV"] = "1"
    if args.gap_months:
        os.environ["GAP_MONTHS"] = str(args.gap_months)
    if args.gap_days:
        os.environ["GAP_DAYS"] = str(args.gap_days)
    if args.trials is not None:
        os.environ["OPTUNA_TRIALS"] = str(args.trials)
    if args.timeout_sec is not None:
        os.environ["OPTUNA_TIMEOUT_SEC"] = str(args.timeout_sec)

    df = pd.read_csv(args.csv_path, low_memory=False)
    artefacts = train_model(
        df_raw=df,
        target_col=args.target,
        date_col=args.date,
        suburb_col=args.suburb,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        max_ohe_card=args.max_ohe_card,
        use_gpu=args.gpu,
        scale_numeric=not args.no_scale,
        explain=args.explain,
        block_cv=not args.no_block_cv,
        skip_featurealgo=args.skip_featurealgo,
        trials=int(os.getenv("OPTUNA_TRIALS", str(DEFAULT_TRIALS))),
        timeout_sec=int(os.getenv("OPTUNA_TIMEOUT_SEC", str(DEFAULT_TIMEOUT))),
        dedup_by_address=args.dedup_by_address,
        opt_loss=os.getenv("OPT_LOSS", "wmae"),
    )
    logging.info("🏁 Done. Artefacts: %s", artefacts)


if __name__ == "__main__":
    cli()
