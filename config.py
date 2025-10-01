"""Project configuration (single source of truth).

This file defines default paths and knobs used across training and serving.
Geo‑POI defaults are provided here so the pipeline can run without CLI flags.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "rental_data.csv")
CLEANED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, "cleaned_rental_data.csv")

PLOT_DIR = os.path.join(BASE_DIR, "plots")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rental_price_model.joblib")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")
STUDIES_DIR = os.path.join(BASE_DIR, "studies")
OPTUNA_STUDY_PATH = os.path.join(STUDIES_DIR, "optuna_study.pkl")

# Artifacts directory (pipelines, importance CSVs, meta, POIs, etc.)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# -------------------- Geo‑POI defaults (train=serve parity) -------------------- #
# These defaults are used automatically by main.py/model_training.py if
# environment variables are not provided and the files exist.

# Default POI CSV location shipped with the repo/artifacts
GEO_POI_CSV = os.path.join(ARTIFACT_DIR, "poi_christchurch.csv")

# Coordinate column names in the property dataset
GEO_LAT_COL = "Latitude"
GEO_LON_COL = "Longitude"

# POI feature geometry
GEO_RADII_KM = (0.5, 1.0, 4.0)      # include 4.0km to reflect typical high-school zones
GEO_DECAY_KM = 1.5                  # decay for accessibility score
GEO_MAX_DECAY_KM = 4.0              # widen accessibility aggregation to 4km

# Optional category whitelist (None = use all categories in the CSV)
# Example to restrict: ("SCHOOL_PRIMARY","SCHOOL_INTERMEDIATE","SCHOOL_HIGH","UNIVERSITY",
#                       "HOSPITAL","PARK","BEACH","TRAIN","BUS")
GEO_CATEGORIES = None

EXTRA_DIRS = [
    MODEL_DIR,
    PLOT_DIR,
    CACHE_DIR,
    METADATA_DIR,
    STUDIES_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACT_DIR,
]

print(f"MODEL_PATH: {MODEL_PATH}")

for d in EXTRA_DIRS:
    os.makedirs(d, exist_ok=True)
