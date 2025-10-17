# prediction.py

import logging
import argparse
import os
import sys
import traceback
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from datetime import timedelta
from config import MODEL_DIR, CLEANED_DATA_PATH
import json
import re
from typing import List, Dict, Tuple, Optional
import hashlib
from pathlib import Path

from log_policy import ALLOWED_LOG_FEATURES, apply_log_policy

UNWANTED_FEATURES = {
    "Month",
    "Month_sin",
    "Month_cos",
    "has_car",
    "has_two_car",
    "has_three_plus_car",
    "is_newer_2010",
}
GEO_COLUMNS = {"Latitude", "Longitude"}

# Serve-time policy override derived from training metadata.
# When True, keep raw bases alongside log_ features for allowed heavy-tailed columns.
_SERVE_KEEP_RAW_WHEN_LOG: bool | None = None



_FURNISHINGS_CANONICAL = {
    # Train-time uses lowercase categories; keep serve-time identical
    "unfurnished": "unfurnished",
    "furnished": "furnished",
    "partially furnished": "partially furnished",
    "partial": "partially furnished",
    "fully furnished": "furnished",
    "full": "furnished",
}


def _canonicalize_furnishings(value: object) -> str:
    """Return lowercase categories to match training artifacts.

    Expected outputs: 'unfurnished' | 'furnished' | 'partially furnished'.
    Unknown/empty → 'unfurnished'.
    """
    if value is None:
        return "unfurnished"
    if isinstance(value, float) and np.isnan(value):
        return "unfurnished"
    text = str(value).strip()
    if not text:
        return "unfurnished"
    lower = text.lower()
    if lower in {"nill", "nil", "none", "null"}:
        return "unfurnished"
    if "fully" in lower and "furnish" in lower:
        return "furnished"
    if "partial" in lower:
        return "partially furnished"
    return _FURNISHINGS_CANONICAL.get(lower, lower)

# Override with Title Case canonicalization expected by tests and cleaning
_FURNISHINGS_CANONICAL = {
    "unfurnished": "Unfurnished",
    "furnished": "Furnished",
    "partially furnished": "Partially Furnished",
    "partial": "Partially Furnished",
    "fully furnished": "Furnished",
    "full": "Furnished",
}

def _canonicalize_furnishings(value: object) -> str:  # type: ignore[no-redef]
    """Normalize furnishings to Title Case categories used in tests and cleaning."""
    if value is None:
        return "Unfurnished"
    if isinstance(value, float) and np.isnan(value):
        return "Unfurnished"
    text = str(value).strip()
    if not text:
        return "Unfurnished"
    lower = text.lower()
    if lower in {"nill", "nil", "none", "null"}:
        return "Unfurnished"
    if "fully" in lower and "furnish" in lower:
        return "Furnished"
    if "partial" in lower:
        return "Partially Furnished"
    return _FURNISHINGS_CANONICAL.get(lower, text.title())


_PETS_CANONICAL = {
    "no": "No Pets",
    "no pets": "No Pets",
    "not allowed": "No Pets",
    "pets not allowed": "No Pets",
    "pets ok": "Pets Allowed",
    "pets allowed": "Pets Allowed",
    "yes": "Pets Allowed",
    "pet friendly": "Pets Allowed",
    "pets negotiable": "Pets Negotiable",
    "negotiable": "Pets Negotiable",
}


def _canonicalize_pets(value: object) -> str:
    if value is None:
        return "No Pets"
    if isinstance(value, float) and np.isnan(value):
        return "No Pets"
    text_val = str(value).strip()
    if not text_val:
        return "No Pets"
    lower = text_val.lower()
    if lower in {"nill", "nil", "none", "null"}:
        return "No Pets"
    if "pets ok" in lower or "pet ok" in lower:
        return "Pets Allowed"
    if "allow" in lower and "pet" in lower:
        return "Pets Allowed"
    if "friendly" in lower and "pet" in lower:
        return "Pets Allowed"
    if "negotiable" in lower and "pet" in lower:
        return "Pets Negotiable"
    if lower in _PETS_CANONICAL:
        return _PETS_CANONICAL[lower]
    if lower == "ok":
        return "Pets Allowed"
    return text_val.title()


_GARAGE_CANONICAL = {
    "no": "No Garage",
    "none": "No Garage",
    "nil": "No Garage",
    "n": "No Garage",
    "0": "No Garage",
    "yes": "Has Garage",
    "y": "Has Garage",
}


def _canonicalize_garage(value: object) -> str:
    if value is None:
        return "No Garage"
    if isinstance(value, float) and np.isnan(value):
        return "No Garage"
    text_val = str(value).strip()
    if not text_val:
        return "No Garage"
    lower = text_val.lower()
    if lower in _GARAGE_CANONICAL:
        return _GARAGE_CANONICAL[lower]
    if "no garage" in lower:
        return "No Garage"
    # default positive form implies presence of a garage
    return "Has Garage"


def _apply_feature_policy(
    df: pd.DataFrame,
    *,
    drop_geo: bool = True,
    keep_raw: bool | None = None,
) -> pd.DataFrame:
    """Apply train↔serve log policy and geo/feature hygiene (serve-time).

    - Enforce the same whitelist used at training via apply_log_policy(drop_raw=True)
    - Clip geo distances if configured
    - Drop known unwanted/raw geo features
    """
    # Enforce train->serve parity for log feature handling.
    # Priority: explicit arg keep_raw -> metadata-derived global -> default(False)
    _keep_raw = keep_raw
    if _keep_raw is None:
        _keep_raw = _SERVE_KEEP_RAW_WHEN_LOG if _SERVE_KEEP_RAW_WHEN_LOG is not None else False
    df = apply_log_policy(df, drop_raw=not bool(_keep_raw))
    clip_km = float(os.getenv("GEO_CLIP_DIST_KM", "0") or 0)
    if clip_km > 0:
        for col in [c for c in df.columns if re.match(r"^dist_.*_km$", str(c))]:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=clip_km)
    drop = [c for c in UNWANTED_FEATURES if c in df.columns]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    if drop_geo:
        geo_drop = [c for c in GEO_COLUMNS if c in df.columns]
        if geo_drop:
            df = df.drop(columns=geo_drop, errors="ignore")
    return df


def _geocode_missing_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Latitude" not in df.columns:
        df["Latitude"] = np.nan
    if "Longitude" not in df.columns:
        df["Longitude"] = np.nan
    mask = df["Latitude"].isna() | df["Longitude"].isna()
    if not mask.any():
        return df
    if NominatimGeocoder is None or compose_address is None:
        logging.debug("Geocoder not available; skipping lookup for missing coordinates")
        return df
    geocoder = None
    try:
        geocoder = NominatimGeocoder()
    except Exception as exc:
        logging.debug("Unable to initialise geocoder: %s", exc)
    for idx in df[mask].index:
        row = df.loc[idx]
        try:
            addr = compose_address(
                row,
                street_cols=("Street Address", "street_address"),
                suburb_col="Suburb",
                city_col="City",
                postcode_col="Postcode",
            )
        except Exception as exc:
            logging.debug("Compose address failed for index %s: %s", idx, exc)
            addr = None
        query_val = str(row.get("__geo_query__", "")).strip()
        if not query_val:
            query_val = addr or ""
        if not addr:
            logging.debug("No address components available for geocoding index %s", idx)
            continue
        try:
            result = geocoder.geocode(addr)
        except Exception as exc:
            logging.debug("Geocode lookup failed for %s: %s", addr, exc)
            continue
        if result and getattr(result, "ok", False):
            try:
                lat_val = float(result.lat)
                lon_val = float(result.lon)
                df.at[idx, "Latitude"] = lat_val
                df.at[idx, "Longitude"] = lon_val
                logging.info("Geocoded %s -> (%0.6f, %0.6f)", addr, lat_val, lon_val)
            except Exception as exc:
                logging.debug("Failed to assign geocode result for %s: %s", addr, exc)
                continue
            try:
                _append_geocode_result(query_val or result.query, lat_val, lon_val)
            except Exception:
                logging.debug("Failed to persist geocode result for %s", addr)
    if geocode_properties is not None:
        try:
            df = geocode_properties(
                df,
                lat_col="Latitude",
                lon_col="Longitude",
                street_cols=("Street Address", "street_address"),
                suburb_col="Suburb",
                city_col="City",
                postcode_col="Postcode",
                geocoder=geocoder if geocoder is not None else None,
            )
        except Exception as exc:
            logging.debug("Batch geocode fallback failed: %s", exc)
    return df

def _append_geocode_result(query: str, lat: float, lon: float) -> None:
    path = Path(os.getenv("GEOCODE_RESULTS_CSV", "artifacts/geocode_query_results.csv"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([[query, lat, lon]], columns=["query", "latitude", "longitude"])
        df.to_csv(path, mode='a', header=not path.exists(), index=False)
    except Exception:
        logging.debug("Failed to append geocode result for %s", query)


try:
    from geo_features import compute_geo_features  # optional at serve-time
except Exception:  # pragma: no cover
    compute_geo_features = None  # type: ignore

try:
    from geo_coding import (
        assign_geo_queries,
        enrich_with_geocodes,
        compose_address,
        NominatimGeocoder,
        geocode_properties,
    )
except Exception:  # pragma: no cover
    assign_geo_queries = None  # type: ignore
    enrich_with_geocodes = None  # type: ignore
    compose_address = None  # type: ignore
    NominatimGeocoder = None  # type: ignore
    geocode_properties = None  # type: ignore

GEO_INCLUDE_BEARINGS = os.getenv("GEO_INCLUDE_BEARINGS", "0").lower() in {
    "1",
    "true",
    "yes",
}
GEO_USE_ACCESSIBILITY = os.getenv("GEO_USE_ACCESSIBILITY", "0").lower() in {
    "1",
    "true",
    "yes",
}
_GEO_SCHOOL_MODE_RAW = (
    (os.getenv("GEO_SCHOOL_FEATURE", "dist_min") or "dist_min").strip().lower()
)
_GEO_SCHOOL_MODES = {"dist_min", "primary", "intermediate", "high"}

GEO_SCHOOL_FEATURE = (
    _GEO_SCHOOL_MODE_RAW if _GEO_SCHOOL_MODE_RAW in _GEO_SCHOOL_MODES else "dist_min"
)

# Compatibility shim for scikit-learn version drift during unpickling
try:
    from sklearn.compose import _column_transformer as _ct  # type: ignore

    if not hasattr(_ct, "_RemainderColsList"):

        class _RemainderColsList(list):
            pass

        _ct._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
except Exception:
    pass

# Silence scikit-learn cross-version unpickle warnings for stable serve.
try:  # sklearn >=1.6
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
except Exception:  # sklearn <=1.5
    try:
        from sklearn.base import InconsistentVersionWarning  # type: ignore
    except Exception:  # fallback
        InconsistentVersionWarning = Warning  # type: ignore
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[logging.FileHandler("prediction.log"), logging.StreamHandler()],
)


def _artifact_candidates() -> List[Tuple[str, Optional[str]]]:
    try:
        art_dir = os.path.join(os.getcwd(), "artifacts")
        if not os.path.isdir(art_dir):
            return []
        candidates = []
        for name in os.listdir(art_dir):
            if not (name.startswith("pipeline_") and name.endswith(".joblib")):
                continue
            path = os.path.join(art_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0.0
            stem = os.path.splitext(name)[0].replace("pipeline_", "", 1)
            meta_path = os.path.join(art_dir, f"meta_{stem}.json")
            if not os.path.exists(meta_path):
                meta_path = None
            candidates.append((mtime, path, meta_path))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [(path, meta_path) for _, path, meta_path in candidates]
    except Exception:
        return []

def _find_latest_artifacts():
    candidates = _artifact_candidates()
    return candidates[0] if candidates else (None, None)


class LoadedPipelineAdapter:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.named_steps = {"preprocessor": preprocessor, "model": model}

    def predict(self, X: pd.DataFrame | np.ndarray):
        try:
            Xtr = self.preprocessor.transform(X)
            return self.model.predict(Xtr)
        except Exception as e:
            # Avoid refitting custom FeatureEngineering at serve-time; only refit if clearly unfitted
            try:
                from sklearn.exceptions import NotFittedError  # type: ignore
            except Exception:
                class NotFittedError(Exception):
                    pass
            # Heuristic: consider refit only for sklearn transformers lacking fitted attrs
            def _preproc_fitted(p):
                try:
                    if hasattr(p, "feature_names_out_"):
                        return True
                    if hasattr(p, "transformers_"):
                        return True
                    if hasattr(p, "named_steps") and hasattr(p.named_steps.get("ct", object()), "transformers_"):
                        return True
                except Exception:
                    return False
                return False
            if isinstance(e, NotFittedError) or not _preproc_fitted(self.preprocessor):
                try:
                    self.preprocessor.fit(X)
                    Xtr = self.preprocessor.transform(X)
                    return self.model.predict(Xtr)
                except Exception:
                    # Give up; bubble up to blending logic which will skip this pipeline
                    raise
            # Otherwise, re-raise so caller can handle the failure and fall back cleanly
            raise


def _coerce_pipeline_object(obj):
    if isinstance(obj, dict) and "preprocessor" in obj and "model" in obj:
        return LoadedPipelineAdapter(obj["preprocessor"], obj["model"])
    try:
        from sklearn.pipeline import Pipeline as SkPipeline  # type: ignore

        if isinstance(obj, SkPipeline) and hasattr(obj, "named_steps"):
            pre = obj.named_steps.get("preprocessor")
            mdl = obj.named_steps.get("model", obj)
            if pre is not None and mdl is not None:
                return LoadedPipelineAdapter(pre, mdl)
    except Exception:
        pass
    return obj


def _expected_columns_from_preprocessor(preproc) -> set:
    """Return the set of original DataFrame column names selected by the trained preprocessor.
    Walks Pipeline -> ColumnTransformer ('ct') to be robust to nesting.
    """
    cols: set[str] = set()
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        # Identify the ColumnTransformer whether top-level or nested under a 'ct' step
        ct = None
        if hasattr(preproc, "named_steps") and "ct" in getattr(preproc, "named_steps"):
            ct = preproc.named_steps["ct"]
        elif isinstance(preproc, ColumnTransformer) or hasattr(preproc, "transformers_"):
            ct = preproc
        if ct is None:
            return cols
        for name, trans, sel in getattr(ct, "transformers_", []):
            if sel is None or sel == "drop":
                continue
            # Only literal selectors (list/tuple/Index); ignore callables and masks
            if hasattr(sel, "__iter__") and not isinstance(sel, str):
                for c in sel:
                    cols.add(str(c))
        return cols
    except Exception:
        return cols


def _serve_feature_order(metadata: dict, full_pipeline) -> list[str]:
    """
    Build a safe, order-preserving schema for serve:
      serve_cols = metadata.feature_order  ∪  expected_columns_from_preprocessor
    This prevents dropping columns required by the preprocessor but missing from metadata.
    """
    meta_order = list((metadata or {}).get("feature_order") or [])
    expected = set()
    try:
        if hasattr(full_pipeline, "preprocessor"):
            expected = _expected_columns_from_preprocessor(full_pipeline.preprocessor)
    except Exception:
        expected = set()
    out: list[str] = list(meta_order)
    # Append any preprocessor-expected columns not already in meta_order (stable append)
    for c in sorted(expected):
        if c not in out:
            out.append(c)
    return out


def _get_frequent_levels_from_preprocessor(preproc) -> dict:
    """Return {categorical_col: set(frequent_levels)} learned during training.

    Traverses preprocessor Pipeline -> ColumnTransformer 'cat' -> RareCategoryBinner.
    """
    try:
        # Our preprocessor is usually a Pipeline([('preinput', ...), ('ct', ColumnTransformer(...))])
        ct = None
        if hasattr(preproc, "named_steps") and "ct" in preproc.named_steps:
            ct = preproc.named_steps["ct"]
        elif hasattr(preproc, "transformers_"):
            ct = preproc
        if ct is None:
            return {}
        for name, trans, sel in getattr(ct, "transformers_", []):
            if name == "cat" and hasattr(trans, "steps"):
                # Expect Pipeline([('bin', RareCategoryBinner), ('imp', SimpleImputer), ('to_str', ToStringTransformer)])
                for step_name, step_obj in trans.steps:
                    if step_name == "bin" and hasattr(step_obj, "frequent_"):
                        return {
                            str(k): set(map(str, v))
                            for k, v in step_obj.frequent_.items()
                        }
        return {}
    except Exception:
        return {}


def _harmonize_categories_to_frequent_levels(df: pd.DataFrame, preproc) -> pd.DataFrame:
    """Map incoming categorical strings to the exact canonical labels seen in training.

    - Uses RareCategoryBinner.frequent_ captured from the fitted preprocessor.
    - Case-insensitive match: builds a lowercased lookup to canonical labels per column.
    - Only affects columns present in the frequent_ dict; leaves others untouched.
    """
    try:
        freq = _get_frequent_levels_from_preprocessor(preproc)
        if not freq:
            return df
        # Make column-name lookup case-insensitive
        freq_ci = {str(k).lower(): set(map(str, v)) for k, v in freq.items()}
        out = df.copy()
        debug = os.getenv("PREDICT_DEBUG", "0").lower() in {"1", "true", "yes"}
        lower_map = {str(c).lower(): c for c in out.columns}
        for col_l, levels in freq_ci.items():
            orig_col = lower_map.get(col_l)
            if orig_col is None or not isinstance(levels, (set, list, tuple)):
                continue
            # Build lowercase -> canonical map
            canon: dict[str, str] = {}
            for lv in levels:
                s = str(lv)
                key = s.lower()
                if key not in canon:
                    canon[key] = s
            # Apply to column values
            try:
                before = out[orig_col].astype(str)
                lower = before.str.lower()
                mapped = lower.map(lambda k: canon.get(k, None))
                mask = mapped.notna()
                if mask.any():
                    out.loc[mask, orig_col] = mapped[mask].astype(str)
                    if debug:
                        try:
                            nrep = int(mask.sum())
                            logging.info("Harmonized '%s': %d value(s) snapped to training labels", orig_col, nrep)
                        except Exception:
                            pass
            except Exception:
                continue
        return out
    except Exception:
        return df


def _ensure_heavy_tail_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Create logs only for whitelisted heavy-tailed bases, consistent with training
    for base in ALLOWED_LOG_FEATURES:
        if base in out.columns and f"log_{base}" not in out.columns:
            out[f"log_{base}"] = np.log1p(
                pd.to_numeric(out[base], errors="coerce").clip(lower=0)
            )
    # Enforce policy to keep logs and drop raws accordingly
    return _apply_feature_policy(out, drop_geo=False)


def _engineer_features_serve(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Age and recency
    try:
        if "Year Built" in out.columns and "Last Rental Date" in out.columns:
            yr = pd.to_numeric(out["Year Built"], errors="coerce")
            cur_year = pd.to_datetime(out["Last Rental Date"], errors="coerce").dt.year
            out["AgeYears"] = (cur_year - yr).clip(lower=0).astype(float)
            out["is_newer_2010"] = (yr >= 2010).astype("Int8")
    except Exception:
        pass
    # Intentionally avoid cap_per_sqm/log_cap_per_sqm (confounds small vs large units)
    # Floor/Land ratio
    try:
        if ("Floor Size (sqm)" in out.columns) and ("Land Size (sqm)" in out.columns):
            flr = pd.to_numeric(out["Floor Size (sqm)"], errors="coerce").clip(lower=0)
            land = pd.to_numeric(out["Land Size (sqm)"], errors="coerce").clip(lower=0)
            out["floor_land_ratio"] = (
                (flr / land.replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
    except Exception:
        pass
    # Bed/Bath ratio
    try:
        if ("Bed" in out.columns) and ("Bath" in out.columns):
            bed = pd.to_numeric(out["Bed"], errors="coerce").clip(lower=0)
            bath = pd.to_numeric(out["Bath"], errors="coerce").clip(lower=0)
            out["bed_bath_ratio"] = (
                (bed / bath.replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
    except Exception:
        pass
    # Car capacity flags
    try:
        if "Car" in out.columns:
            carn = pd.to_numeric(out["Car"], errors="coerce").fillna(0)
            out["has_car"] = (carn >= 1).astype("Int8")
            out["has_two_car"] = (carn >= 2).astype("Int8")
            out["has_three_plus_car"] = (carn >= 3).astype("Int8")
    except Exception:
        pass
    # Bed-normalized floor size using historical distribution if available later
    return out


def _geo_kwargs_from_metadata(metadata: dict) -> Dict[str, object] | None:
    # Hard-off switch: if GEO_FORCE_OFF is set, skip geo features at serve-time
    try:
        if os.getenv("GEO_FORCE_OFF", "0").lower() in {"1", "true", "yes"}:
            return None
    except Exception:
        return None
    geo_meta = (metadata or {}).get("geo") or {}
    poi_csv = geo_meta.get("poi_csv") or os.getenv("GEO_POI_CSV") or "artifacts/poi_christchurch.csv"
    if poi_csv and not os.path.exists(poi_csv):
        alt = Path(os.getenv("MODEL_DIR", "artifacts")).joinpath(Path(poi_csv).name)
        if alt.exists():
            poi_csv = str(alt)
    if not poi_csv or not os.path.exists(poi_csv):
        logging.warning("Geo metadata missing POI CSV; skipping geo feature augmentation")
        return None

    radii = geo_meta.get("radii_km") or os.getenv("GEO_RADII_KM") or "0.5,1.0,2.0"
    if isinstance(radii, str):
        radii = [float(x.strip()) for x in radii.split(",") if x.strip()]
    include_bearings = geo_meta.get("include_bearings")
    if include_bearings is None:
        include_bearings = GEO_INCLUDE_BEARINGS
    use_accessibility = geo_meta.get("use_accessibility")
    if use_accessibility is None:
        use_accessibility = GEO_USE_ACCESSIBILITY
    school_mode = geo_meta.get("school_mode") or GEO_SCHOOL_FEATURE or "dist_min"

    categories = geo_meta.get("categories")
    if isinstance(categories, str):
        categories = [
            c.strip().upper().replace(" ", "_")
            for part in categories.split(",")
            for c in part.split(" ")
            if c.strip()
        ]
    elif isinstance(categories, (list, tuple)):
        categories = [str(c).upper().replace(" ", "_") for c in categories]
    else:
        categories = None

    lat_col = geo_meta.get("lat_col") or os.getenv("LAT_COL", "Latitude")
    lon_col = geo_meta.get("lon_col") or os.getenv("LON_COL", "Longitude")

    return {
        "poi_csv": poi_csv,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "categories": tuple(categories) if categories else None,
        "radii_km": tuple(float(r) for r in radii),
        "decay_km": geo_meta.get("decay_km") or float(os.getenv("GEO_DECAY_KM", 1.5)),
        "max_decay_km": geo_meta.get("max_decay_km")
        or float(os.getenv("GEO_MAX_DECAY_KM", 3.0)),
        "include_bearings": bool(include_bearings),
        "use_accessibility": bool(use_accessibility),
        "school_mode": school_mode,
    }


def _maybe_add_geo_serve(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    try:
        cfg = _geo_kwargs_from_metadata(metadata)
        if not cfg or compute_geo_features is None:
            return df
        return compute_geo_features(
            df,
            poi_csv=cfg["poi_csv"],
            lat_col=cfg["lat_col"],
            lon_col=cfg["lon_col"],
            categories=cfg["categories"],
            radii_km=cfg["radii_km"],
            decay_km=cfg["decay_km"],
            max_decay_km=cfg["max_decay_km"],
            include_bearings=cfg.get("include_bearings"),
            use_accessibility=cfg.get("use_accessibility"),
            school_mode=cfg.get("school_mode"),
        )
    except Exception:
        pass
    return df


# -------------------- Nearest-properties helper (visual only) -------------------- #

EARTH_R_KM = 6371.0088


def _nearest_properties(
    historical: pd.DataFrame,
    *,
    query_lat: float,
    query_lon: float,
    k: int = 5,
    prefer_same_bed: bool = True,
    bed_str: Optional[str] = None,
    radius_km: Optional[float] = None,
) -> pd.DataFrame:
    """Return up to k nearest properties. Visual aid only; no effect on predictions.

    - Filters by same Bed when possible, then relaxes to all if needed.
    - Computes haversine distance (km) via BallTree.
    - Optionally filters by radius_km.
    """
    if historical is None or historical.empty:
        return pd.DataFrame()

    work = historical.copy()
    # Coerce coordinates
    work["Latitude"] = pd.to_numeric(work.get("Latitude"), errors="coerce")
    work["Longitude"] = pd.to_numeric(work.get("Longitude"), errors="coerce")
    work = work.dropna(subset=["Latitude", "Longitude"]).copy()
    if work.empty:
        return work

    def _nn(df_sub: pd.DataFrame, kk: int) -> pd.DataFrame:
        if df_sub.empty:
            return df_sub.assign(distance_km=np.nan).iloc[:0]
        coords = np.deg2rad(df_sub[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))
        tree = BallTree(coords, metric="haversine")
        q = np.deg2rad(np.array([[float(query_lat), float(query_lon)]], dtype=np.float64))
        k_eff = min(max(int(kk), 1), len(df_sub))
        d_rad, idx = tree.query(q, k=k_eff)
        out = df_sub.iloc[idx[0]].copy()
        out["distance_km"] = (d_rad[0] * EARTH_R_KM).astype(float)
        out = out.sort_values("distance_km")
        return out

    # Stage 1: prefer same Bed (if requested and possible)
    out = pd.DataFrame()
    if prefer_same_bed and ("Bed" in work.columns) and isinstance(bed_str, str) and len(bed_str) > 0:
        same = work[work["Bed"].astype(str) == str(bed_str)]
        out = _nn(same, k)
    if out.shape[0] < k:
        # Relax to all
        out_all = _nn(work, k)
        if out.empty:
            out = out_all
        else:
            out = pd.concat([out, out_all[~out_all.index.isin(out.index)]], axis=0)
            out = out.sort_values("distance_km").head(k)

    if radius_km is not None and np.isfinite(radius_km):
        out = out[out["distance_km"] <= float(radius_km)].copy()
        out = out.sort_values("distance_km").head(k)

    return out.head(k)


def _prepare_features_for_model(
    features: pd.DataFrame,
    full_pipeline,
    metadata: dict,
    historical_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, set, List[str], List[str]]:
    """Replicate prediction backfill so the same prepared frame is used for SHAP.

    Returns: (prepared_features, expected_cols, missing_before, categorical_features)
    """
    work = features.copy()
    work = _apply_feature_policy(work, drop_geo=False)
    try:
        if assign_geo_queries is not None:
            work = assign_geo_queries(work)
        if enrich_with_geocodes is not None:
            work = enrich_with_geocodes(work)
        work = _geocode_missing_latlon(work)
    except Exception:
        pass
    # Ensure date type
    if "Last Rental Date" in work.columns and not pd.api.types.is_datetime64_any_dtype(
        work["Last Rental Date"]
    ):
        work["Last Rental Date"] = pd.to_datetime(work["Last Rental Date"])

    # Pre-derive logs and engineered features BEFORE expected column check
    work = _ensure_heavy_tail_logs(work)
    work = _engineer_features_serve(work)
    work = _apply_feature_policy(work, drop_geo=False)
    work = _maybe_add_geo_serve(work, metadata)
    work = _apply_feature_policy(work)
    # Optional recency v2 anchor features
    try:
        if os.getenv("RECENCY_V2", "1").lower() in {"1", "true", "yes"}:
            work = _add_rec2_features_serve(work, historical_data)
    except Exception:
        pass

    # Categorical lists from metadata or inference
    categorical_features = (
        metadata.get("cat_features") or metadata.get("categorical_features") or []
    )
    if not categorical_features:
        categorical_features = [
            c
            for c in [
                "Suburb",
                "Property Type",
                "Active Listing",
                "Land Use",
                "Development Zone",
                "Postcode",
            ]
            if c in work.columns
        ]
    numeric_features = metadata.get("num_features") or []

    # Date parts
    if "Last Rental Date" in work.columns:
        dt = pd.to_datetime(work["Last Rental Date"])
        work["Year"] = dt.dt.year
        work["Month"] = dt.dt.month
        work["WeekOfYear"] = dt.dt.isocalendar().week.astype("Int64")
        work["Month_sin"] = np.sin(2 * np.pi * work["Month"] / 12.0)
        work["Month_cos"] = np.cos(2 * np.pi * work["Month"] / 12.0)

    # Expected columns from preprocessor
    expected_cols = set(numeric_features) | set(categorical_features)
    # If LOO features were not present during training run, avoid creating them at serve time
    if "Suburb_LOO" not in expected_cols:
        expected_cols.discard("Suburb_LOO")
    if "SuburbMonth_LOO" not in expected_cols:
        expected_cols.discard("SuburbMonth_LOO")
    if hasattr(full_pipeline, "preprocessor"):
        expected_cols |= _expected_columns_from_preprocessor(full_pipeline.preprocessor)

    # Normalize aliases and categorical cleaning
    work = _normalize_aliases2(work, expected_cols)
    work = _clean_categoricals(work)
    # Snap common categoricals (e.g., Suburb/Agency/Type) to training's canonical labels
    try:
        if hasattr(full_pipeline, "preprocessor"):
            work = _harmonize_categories_to_frequent_levels(work, full_pipeline.preprocessor)
    except Exception:
        pass

    # Compute bed-normalized floor size using historical_data (if expected)
    try:
        if (
            "floor_to_bed_med" in expected_cols
            and "Bed" in work.columns
            and "Floor Size (sqm)" in work.columns
        ):
            hd = historical_data.copy()
            hd["Bed"] = hd["Bed"].astype(str)
            hd["Floor Size (sqm)"] = pd.to_numeric(
                hd["Floor Size (sqm)"], errors="coerce"
            )
            # Use time-consistent median prior to the as-of date per-row for leak-safety
            vals = []
            for _, row in work.iterrows():
                asof = pd.to_datetime(row.get("Last Rental Date"))
                bed = str(row.get("Bed"))
                dfb = hd[(hd["Bed"] == bed) & (hd["Last Rental Date"] < asof)]
                med = (
                    dfb["Floor Size (sqm)"].median()
                    if not dfb.empty
                    else hd[hd["Bed"] == bed]["Floor Size (sqm)"].median()
                )
                vals.append(float(med) if pd.notna(med) else np.nan)
            med_map = pd.Series(vals, index=work.index)
            flr = pd.to_numeric(work["Floor Size (sqm)"], errors="coerce")
            work["floor_to_bed_med"] = (
                (flr / med_map.replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
            )
    except Exception:
        pass

    original_cols = set(work.columns)
    missing_before = sorted(list(expected_cols - original_cols))

    # Serve bundle assets
    bundle = metadata.get("serve_bundle") or {}
    med_df = None
    med_bed_df = None
    med365_df = None
    med365_bed_df = None
    try:
        med_csv = bundle.get("med90_csv")
        if med_csv and os.path.exists(med_csv):
            med_df = pd.read_csv(med_csv)
            med_df["Date"] = pd.to_datetime(med_df["Date"])
            if "Suburb" in med_df.columns:
                med_df["Suburb"] = med_df["Suburb"].astype(str).str.upper()
        med_bed_csv = bundle.get("med90_bed_csv")
        if med_bed_csv and os.path.exists(med_bed_csv):
            med_bed_df = pd.read_csv(med_bed_csv)
            med_bed_df["Date"] = pd.to_datetime(med_bed_df["Date"])
            for c in ("Suburb", "Bed"):
                if c in med_bed_df.columns:
                    med_bed_df[c] = med_bed_df[c].astype(str)
            if "Suburb" in med_bed_df.columns:
                med_bed_df["Suburb"] = med_bed_df["Suburb"].str.upper()
        med365_csv = bundle.get("med365_csv")
        if med365_csv and os.path.exists(med365_csv):
            med365_df = pd.read_csv(med365_csv)
            med365_df["Date"] = pd.to_datetime(med365_df["Date"])
            if "Suburb" in med365_df.columns:
                med365_df["Suburb"] = med365_df["Suburb"].astype(str).str.upper()
        med365_bed_csv = bundle.get("med365_bed_csv")
        if med365_bed_csv and os.path.exists(med365_bed_csv):
            med365_bed_df = pd.read_csv(med365_bed_csv)
            med365_bed_df["Date"] = pd.to_datetime(med365_bed_df["Date"])
            for c in ("Suburb", "Bed"):
                if c in med365_bed_df.columns:
                    med365_bed_df[c] = med365_bed_df[c].astype(str)
            if "Suburb" in med365_bed_df.columns:
                med365_bed_df["Suburb"] = med365_bed_df["Suburb"].str.upper()
    except Exception:
        med_df = None
        med_bed_df = None
        med365_df = None
        med365_bed_df = None
    enc = {}
    try:
        enc_path = bundle.get("encoders_joblib")
        if enc_path and os.path.exists(enc_path):
            enc = joblib.load(enc_path) or {}
    except Exception:
        enc = {}

    # Fill any missing expected columns with derived values or safe defaults
    for col in sorted(expected_cols):
        if col in work.columns:
            continue

        # derive logs if possible
        if col.startswith("log_"):
            base = col[4:]
            work[col] = (
                np.log1p(pd.to_numeric(work[base], errors="coerce"))
                if base in work.columns
                else np.nan
            )
            continue

        # simple engineered placeholders
        if col in {
            "Year",
            "Month",
            "WeekOfYear",
            "Month_sin",
            "Month_cos",
            "AgeYears",
            "is_newer_2010",
            "floor_land_ratio",
            "bed_bath_ratio",
            "floor_to_bed_med",
        }:
            work[col] = work.get(col, pd.Series([np.nan] * len(work)))
            continue

        # time medians (90d suburb; prefer bed-aware if available)
        if col == "Suburb90dMedian":
            try:
                if (med_bed_df is not None) and (not med_bed_df.empty) and ("Bed" in work.columns):
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        bed = str(row.get("Bed", ""))
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        m = (
                            med_bed_df[
                                (med_bed_df["Suburb"].astype(str) == sub)
                                & (med_bed_df["Bed"].astype(str) == bed)
                                & (med_bed_df["Date"] <= asof)
                            ]
                            .sort_values("Date")
                            .tail(1)["SuburbBed90dMedian"]
                        )
                        vals.append(float(m.iloc[0]) if len(m) else np.nan)
                    work[col] = vals
                elif (med_df is not None) and (not med_df.empty):
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        m = (
                            med_df[
                                (med_df["Suburb"].astype(str) == sub)
                                & (med_df["Date"] <= asof)
                            ]
                            .sort_values("Date")
                            .tail(1)["Suburb90dMedian"]
                        )
                        vals.append(float(m.iloc[0]) if len(m) else np.nan)
                    work[col] = vals
                else:
                    med_vals = []
                    for _, row in work.iterrows():
                        med_vals.append(
                            _compute_suburb90d_median(
                                historical_data,
                                str(row.get("Suburb", "")),
                                pd.to_datetime(row.get("Last Rental Date")),
                            )
                        )
                    work[col] = med_vals
            except Exception:
                work[col] = np.nan
            continue

        if col == "SuburbBed90dMedian":
            try:
                if (med_bed_df is not None) and (not med_bed_df.empty) and ("Bed" in work.columns):
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        bed = str(row.get("Bed", ""))
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        m = (
                            med_bed_df[
                                (med_bed_df["Suburb"].astype(str) == sub)
                                & (med_bed_df["Bed"].astype(str) == bed)
                                & (med_bed_df["Date"] <= asof)
                            ]
                            .sort_values("Date")
                            .tail(1)["SuburbBed90dMedian"]
                        )
                        vals.append(float(m.iloc[0]) if len(m) else np.nan)
                    work[col] = vals
                else:
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        bed = str(row.get("Bed", ""))
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        df = historical_data.copy()
                        df = (
                            df[
                                (df["Suburb"].astype(str).str.upper() == sub)
                                & (df["Bed"].astype(str) == bed)
                            ]
                        )
                        df = (
                            df[df["Last Rental Date"] < asof]
                            .sort_values("Last Rental Date")
                            .set_index("Last Rental Date")
                        )
                        med = (
                            df["Last Rental Price"].rolling("90D", min_periods=1).median().iloc[-1]
                            if not df.empty
                            else np.nan
                        )
                        vals.append(float(med) if pd.notna(med) else np.nan)
                    work[col] = vals
            except Exception:
                work[col] = np.nan
            continue

        if col == "Suburb365dMedian":
            try:
                if (med365_df is not None) and (not med365_df.empty):
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        m = (
                            med365_df[
                                (med365_df["Suburb"].astype(str) == sub)
                                & (med365_df["Date"] <= asof)
                            ]
                            .sort_values("Date")
                            .tail(1)["Suburb365dMedian"]
                        )
                        vals.append(float(m.iloc[0]) if len(m) else np.nan)
                    work[col] = vals
                else:
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        df = historical_data.copy()
                        df = df[df["Suburb"].astype(str).str.upper() == sub]
                        df = (
                            df[df["Last Rental Date"] < asof]
                            .sort_values("Last Rental Date")
                            .set_index("Last Rental Date")
                        )
                        med = (
                            df["Last Rental Price"].rolling("365D", min_periods=1).median().iloc[-1]
                            if not df.empty
                            else np.nan
                        )
                        vals.append(float(med) if pd.notna(med) else np.nan)
                    work[col] = vals
            except Exception:
                work[col] = np.nan
            continue

        if col == "SuburbBed365dMedian":
            try:
                if (med365_bed_df is not None) and (not med365_bed_df.empty) and ("Bed" in work.columns):
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", "")).upper()
                        bed = str(row.get("Bed", ""))
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        m = (
                            med365_bed_df[
                                (med365_bed_df["Suburb"].astype(str) == sub)
                                & (med365_bed_df["Bed"].astype(str) == bed)
                                & (med365_bed_df["Date"] <= asof)
                            ]
                            .sort_values("Date")
                            .tail(1)["SuburbBed365dMedian"]
                        )
                        vals.append(float(m.iloc[0]) if len(m) else np.nan)
                    work[col] = vals
                else:
                    work[col] = np.nan
            except Exception:
                work[col] = np.nan
            continue

        if col in {"Suburb_LOO", "SuburbMonth_LOO", "Rank30dX12m"}:
            try:
                if col == "Suburb_LOO" and enc.get("suburb_loo") is not None:
                    work[col] = (
                        enc["suburb_loo"].transform(work[["Suburb"]])["Suburb_LOO"].values
                    )
                elif col == "SuburbMonth_LOO" and enc.get("suburb_month_loo") is not None:
                    tmp = work[["Suburb", "Last Rental Date"]].copy()
                    tmp.columns = ["Suburb", "Date"]
                    work[col] = (
                        enc["suburb_month_loo"].transform(tmp)["SuburbMonth_LOO"].values
                    )
                else:
                    vals = []
                    for _, row in work.iterrows():
                        sub = str(row.get("Suburb", ""))
                        asof = pd.to_datetime(row.get("Last Rental Date"))
                        if col == "Suburb_LOO":
                            v = _fallback_suburb_loo(historical_data, sub, asof)
                        elif col == "SuburbMonth_LOO":
                            v = _fallback_suburb_month_loo(historical_data, sub, asof)
                        else:
                            v = _compute_rank30d12m_global(historical_data, asof)
                        vals.append(np.nan if v is None else v)
                    work[col] = vals
            except Exception:
                work[col] = np.nan
            continue

        # default fallback
        if col in categorical_features:
            work[col] = "UNKNOWN"
        else:
            work[col] = np.nan

    # Ensure all categorical features exist (filled with a safe placeholder)
    for col in categorical_features:
        if col in work.columns:
            work[col] = work[col].astype(str)
        else:
            work[col] = "Unknown"

    # Final schema alignment: ensure all columns required by the trained pipeline exist
    try:
        target_order = list((metadata or {}).get("feature_order") or [])
        if not target_order:
            target_order = list(sorted(expected_cols))
        for col in target_order:
            if col in work.columns:
                continue
            if col in categorical_features:
                work[col] = "Unknown"
            else:
                work[col] = np.nan
        # Keep original column set plus any new adds; reindex later in _predict_blended
    except Exception:
        pass

    return work, expected_cols, missing_before, categorical_features


def _normalize_aliases(df: pd.DataFrame, expected: set) -> pd.DataFrame:
    """Map common alias columns to the names expected by training."""
    out = df.copy()
    if "Furnishings" in out.columns:
        out["Furnishings"] = out["Furnishings"].apply(_canonicalize_furnishings)
    if "Pets" in out.columns:
        out["Pets"] = out["Pets"].apply(_canonicalize_pets)
    alias_pairs = [
        ("Land Size (sqm)", "Land Size (m²)"),
        ("Floor Size (sqm)", "Floor Size (m²)"),
        ("YearBuilt", "Year Built"),
    ]
    for a, b in alias_pairs:
        # If expected contains a and we only have b → copy b to a
        if a in expected and a not in out.columns and b in out.columns:
            out[a] = out[b]
        # Or if expected contains b and we only have a → copy a to b
        if b in expected and b not in out.columns and a in out.columns:
            out[b] = out[a]
    return out


def _add_rec2_features_serve(work: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
    """Compute rec2_* as-of features by merging historical_data and request rows.

    Produces:
      - rec2_suburb_med_{30,90,180,365}d
      - rec2_suburbbed_med_{30,90,180,365}d
      - rec2_mom_suburb_{30,90} and rec2_mom_suburbbed_{30,90} vs 365
    """
    try:
        if work.empty or historical_data is None or historical_data.empty:
            return work
        need_w = {"Suburb", "Last Rental Date"}
        need_h = {"Suburb", "Last Rental Date", "Last Rental Price"}
        if not need_w.issubset(work.columns) or not need_h.issubset(historical_data.columns):
            return work
        w = work.copy()
        w["Bed"] = w.get("Bed", np.nan).astype(str)
        w["Last Rental Date"] = pd.to_datetime(w["Last Rental Date"], errors="coerce")
        if "Last Rental Price" not in w.columns:
            w["Last Rental Price"] = np.nan
        w["__row_id_tmp"] = np.arange(len(w))

        hd = historical_data[["Suburb", "Bed", "Last Rental Date", "Last Rental Price"]].copy()
        hd["Bed"] = hd.get("Bed", np.nan).astype(str)
        hd["Last Rental Date"] = pd.to_datetime(hd["Last Rental Date"], errors="coerce")
        hd = hd.dropna(subset=["Last Rental Date"]).copy()

        combo = pd.concat([hd, w], axis=0, ignore_index=True, sort=False)
        combo = combo.sort_values(["Suburb", "Last Rental Date"]).reset_index(drop=True)

        def _assign_roll(keys: list[str], win: str, name: str):
            try:
                ser = (
                    combo.set_index("Last Rental Date").groupby(keys)["Last Rental Price"]
                    .apply(lambda s: pd.to_numeric(s, errors="coerce").shift().rolling(win, closed="left", min_periods=1).median())
                    .rename(name)
                    .reset_index()[name]
                )
                combo[name] = ser.values
            except Exception:
                combo[name] = np.nan

        for wsize in ("30D", "90D", "180D", "365D"):
            _assign_roll(["Suburb"], wsize, f"rec2_suburb_med_{wsize[:-1]}d")
        for wsize in ("30D", "90D", "180D", "365D"):
            _assign_roll(["Suburb", "Bed"], wsize, f"rec2_suburbbed_med_{wsize[:-1]}d")

        req = combo[pd.notna(combo["__row_id_tmp"])].copy()
        req = req.sort_values("__row_id_tmp")
        idx = req["__row_id_tmp"].astype(int).to_numpy()
        cols = [
            "rec2_suburb_med_30d",
            "rec2_suburb_med_90d",
            "rec2_suburb_med_180d",
            "rec2_suburb_med_365d",
            "rec2_suburbbed_med_30d",
            "rec2_suburbbed_med_90d",
            "rec2_suburbbed_med_180d",
            "rec2_suburbbed_med_365d",
        ]
        for c in cols:
            try:
                work[c] = np.nan
                work.loc[idx, c] = req[c].to_numpy()
            except Exception:
                work[c] = work.get(c, np.nan)

        def _fallback_recency_value(bed_val: str, asof_date: pd.Timestamp, window: str) -> float | np.nan:
            """Return rolling median for a bed cohort, falling back to global history when suburb data is missing."""
            if pd.isna(asof_date):
                return np.nan
            target_subset = hd.loc[(hd["Last Rental Date"] < asof_date)].copy()
            if target_subset.empty:
                return np.nan
            bed_only = target_subset.loc[target_subset["Bed"] == bed_val].copy()
            if not bed_only.empty:
                target_subset = bed_only
            target_subset = target_subset.sort_values("Last Rental Date")
            series = (
                target_subset.set_index("Last Rental Date")["Last Rental Price"]
                .pipe(pd.to_numeric, errors="coerce")
            )
            try:
                rolled = series.shift().rolling(window, closed="left", min_periods=1).median()
            except Exception:
                rolled = pd.Series(dtype=float)
            if rolled is None or rolled.empty:
                return np.nan
            val = rolled.iloc[-1]
            return float(val) if pd.notna(val) and np.isfinite(val) else np.nan

        fallback_windows = {
            "rec2_suburb_med_30d": "30D",
            "rec2_suburb_med_90d": "90D",
            "rec2_suburb_med_180d": "180D",
            "rec2_suburb_med_365d": "365D",
            "rec2_suburbbed_med_30d": "30D",
            "rec2_suburbbed_med_90d": "90D",
            "rec2_suburbbed_med_180d": "180D",
            "rec2_suburbbed_med_365d": "365D",
        }
        fallback_events: list[tuple[str, str, str]] = []
        for col_name, win in fallback_windows.items():
            if col_name not in work.columns:
                continue
            for row_idx in work.index:
                if not pd.isna(work.at[row_idx, col_name]):
                    continue
                bed_val = str(work.at[row_idx, "Bed"]) if "Bed" in work.columns else None
                asof_date = work.at[row_idx, "Last Rental Date"]
                if not bed_val or bed_val.lower() == "nan":
                    continue
                fallback_val = _fallback_recency_value(bed_val, asof_date, win)
                if pd.notna(fallback_val):
                    work.at[row_idx, col_name] = fallback_val
                    suburb_val = str(work.at[row_idx, "Suburb"]) if "Suburb" in work.columns else "<unknown>"
                    fallback_events.append((col_name, suburb_val, bed_val))

        if fallback_events:
            try:
                logging.info(
                    "Recency fallback applied for %d feature(s): %s",
                    len(fallback_events),
                    fallback_events,
                )
            except Exception:
                pass

        with np.errstate(all="ignore"):
            if {"rec2_suburb_med_30d", "rec2_suburb_med_365d"}.issubset(work.columns):
                work["rec2_mom_suburb_30_365"] = (
                    pd.to_numeric(work["rec2_suburb_med_30d"], errors="coerce")
                    / pd.to_numeric(work["rec2_suburb_med_365d"], errors="coerce").replace(0, np.nan)
                )
            if {"rec2_suburb_med_90d", "rec2_suburb_med_365d"}.issubset(work.columns):
                work["rec2_mom_suburb_90_365"] = (
                    pd.to_numeric(work["rec2_suburb_med_90d"], errors="coerce")
                    / pd.to_numeric(work["rec2_suburb_med_365d"], errors="coerce").replace(0, np.nan)
                )
            if {"rec2_suburbbed_med_30d", "rec2_suburbbed_med_365d"}.issubset(work.columns):
                work["rec2_mom_suburbbed_30_365"] = (
                    pd.to_numeric(work["rec2_suburbbed_med_30d"], errors="coerce")
                    / pd.to_numeric(work["rec2_suburbbed_med_365d"], errors="coerce").replace(0, np.nan)
                )
            if {"rec2_suburbbed_med_90d", "rec2_suburbbed_med_365d"}.issubset(work.columns):
                work["rec2_mom_suburbbed_90_365"] = (
                    pd.to_numeric(work["rec2_suburbbed_med_90d"], errors="coerce")
                    / pd.to_numeric(work["rec2_suburbbed_med_365d"], errors="coerce").replace(0, np.nan)
                )
        try:
            work.drop(columns=["__row_id_tmp"], inplace=True, errors="ignore")
        except Exception:
            pass
        return work
    except Exception:
        return work
def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("Land Use", "Development Zone", "Property Type", "Suburb"):
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r"[\\//]", "_", regex=True)
                .fillna("UNKNOWN")
            )
    return out


def _fallback_suburb_loo(
    hdata: pd.DataFrame, suburb: str, asof_date: pd.Timestamp
) -> float | None:
    try:
        df = hdata[
            hdata["Suburb"].astype(str).str.upper() == str(suburb).upper()
        ].copy()
        if df.empty:
            return None
        df = df[df["Last Rental Date"] < asof_date]
        if df.empty:
            return None
        m = pd.to_numeric(df["Last Rental Price"], errors="coerce").mean()
        return float(m) if pd.notna(m) else None
    except Exception:
        return None


def _fallback_suburb_month_loo(
    hdata: pd.DataFrame, suburb: str, asof_date: pd.Timestamp
) -> float | None:
    try:
        df = hdata.copy()
        df["Suburb"] = df["Suburb"].astype(str).str.upper()
        df = df[df["Last Rental Date"] < asof_date]
        ym = pd.to_datetime(asof_date).to_period("M")
        # Prefer same-suburb same-month mean, else same-suburb overall, else global same-month, else global overall
        sub_df = df[df["Suburb"] == str(suburb).upper()].copy()
        if not sub_df.empty:
            sub_df["_ym"] = sub_df["Last Rental Date"].dt.to_period("M")
            m = pd.to_numeric(
                sub_df.loc[sub_df["_ym"] == ym, "Last Rental Price"], errors="coerce"
            ).mean()
            if not np.isfinite(m) or pd.isna(m):
                m = pd.to_numeric(sub_df["Last Rental Price"], errors="coerce").mean()
        else:
            df["_ym"] = df["Last Rental Date"].dt.to_period("M")
            m = pd.to_numeric(
                df.loc[df["_ym"] == ym, "Last Rental Price"], errors="coerce"
            ).mean()
            if not np.isfinite(m) or pd.isna(m):
                m = pd.to_numeric(df["Last Rental Price"], errors="coerce").mean()
        return float(m) if (m is not None and np.isfinite(m)) else None
    except Exception:
        return None


def _fallback_rank_30d_x_12m(
    hdata: pd.DataFrame, suburb: str, asof_date: pd.Timestamp
) -> float | None:
    try:
        df = hdata[
            hdata["Suburb"].astype(str).str.upper() == str(suburb).upper()
        ].copy()
        if df.empty:
            return None
        df = df[df["Last Rental Date"] < asof_date].sort_values("Last Rental Date")
        if df.empty:
            return None
        last_30d_start = asof_date - pd.DateOffset(days=30)
        last_12m_start = asof_date - pd.DateOffset(months=12)
        p30 = pd.to_numeric(
            df.loc[df["Last Rental Date"] >= last_30d_start, "Last Rental Price"],
            errors="coerce",
        ).median()
        p12 = pd.to_numeric(
            df.loc[df["Last Rental Date"] >= last_12m_start, "Last Rental Price"],
            errors="coerce",
        ).median()
        if (
            not np.isfinite(p30)
            or pd.isna(p30)
            or not np.isfinite(p12)
            or pd.isna(p12)
            or p12 == 0
        ):
            return None
        # Ratio as a simple momentum-like rank proxy
        return float(p30 / p12)
    except Exception:
        return None


def _compute_rank30d12m_global(
    hdata: pd.DataFrame, asof_date: pd.Timestamp
) -> float | None:
    """Stable ratio momentum: (rm30 / rm30_12m) - 1, clipped to [-1, 1]."""
    try:
        df = hdata.copy()
        df = df[df["Last Rental Date"] < asof_date]
        if df.empty:
            return None
        daily = (
            df.set_index("Last Rental Date")["Last Rental Price"]
            .groupby(pd.Grouper(freq="D"))
            .mean()
        )
        if daily.empty:
            return None
        rm30 = daily.rolling("30D", closed="left").mean()
        rm30_12m = rm30.shift(365, freq="D")
        val = rm30.asof(asof_date)
        val_12 = rm30_12m.asof(asof_date)
        if not (pd.notna(val) and pd.notna(val_12) and val_12 != 0):
            return None
        ratio = float((val / val_12) - 1.0)
        # Clip extreme moves
        if ratio > 1.0:
            ratio = 1.0
        if ratio < -1.0:
            ratio = -1.0
        return ratio
    except Exception:
        return None


def load_full_pipeline_and_metadata():
    # 1) Prefer explicitly provided candidates (supports tests/monkeypatching)
    try:
        cand = _artifact_candidates()
        if cand:
            for path, meta_path in cand:
                # Try provided meta first
                metadata: Dict[str, object] = {}
                if meta_path and os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            metadata = json.load(f)
                    except Exception:
                        metadata = {}
                # If no meta provided, try to derive by trailing hash in filename
                if not metadata:
                    try:
                        m = re.search(r"([0-9a-fA-F]{8})(?=\.joblib$)", os.path.basename(path))
                        if m:
                            h = m.group(1).lower()
                            meta_guess = os.path.join(os.getcwd(), "artifacts", f"meta_{h}.json")
                            if os.path.exists(meta_guess):
                                with open(meta_guess, "r") as f:
                                    metadata = json.load(f)
                                p_ref = (metadata or {}).get("pipeline")
                                if isinstance(p_ref, str) and os.path.exists(p_ref):
                                    obj = joblib.load(p_ref)
                                    logging.info("Full pipeline loaded successfully from '%s'. (resolved by hash)", p_ref)
                                    logging.info("Metadata loaded successfully from '%s'.", meta_guess)
                                    return _coerce_pipeline_object(obj), metadata
                    except Exception:
                        metadata = {}
                # If metadata exists and no hash-redirect occurred, load the candidate pipeline
                try:
                    # Skip Ridge/quantile as primary
                    bn = os.path.basename(path)
                    if bn.startswith("pipeline_Ridge_") or bn.startswith("pipeline_q"):
                        continue
                    obj = joblib.load(path)
                    # Fittedness guard: require a ColumnTransformer fitted under 'preprocessor'
                    pre = getattr(obj, "preprocessor", None)
                    if pre is None and hasattr(obj, "named_steps"):
                        pre = obj.named_steps.get("preprocessor")
                    fitted = hasattr(pre, "transformers_")
                    if not fitted:
                        logging.warning("Skipping unfitted primary pipeline: %s", path)
                        continue
                    logging.info("Full pipeline loaded successfully from '%s'.", path)
                    if metadata:
                        logging.info("Metadata loaded successfully from '%s'.", meta_path or "<derived/empty>")
                    return _coerce_pipeline_object(obj), metadata
                except Exception:
                    continue
    except Exception:
        pass

    # 2) Metadata-first pairing so the pipeline is always matched to its run metadata
    try:
        arts = os.path.join(os.getcwd(), "artifacts")
        if os.path.isdir(arts):
            metas = [
                os.path.join(arts, n)
                for n in os.listdir(arts)
                if n.startswith("meta_") and n.endswith(".json")
            ]
            metas.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for meta_path in metas:
                try:
                    with open(meta_path, "r") as f:
                        metadata: Dict[str, object] = json.load(f)
                except Exception:
                    continue
                p = (metadata or {}).get("pipeline")
                if isinstance(p, str) and os.path.exists(p):
                    # Skip Ridge/quantile as primary
                    bn = os.path.basename(p)
                    if bn.startswith("pipeline_Ridge_") or bn.startswith("pipeline_q"):
                        continue
                    try:
                        obj = joblib.load(p)
                        # Fittedness guard
                        pre = getattr(obj, "preprocessor", None)
                        if pre is None and hasattr(obj, "named_steps"):
                            pre = obj.named_steps.get("preprocessor")
                        fitted = hasattr(pre, "transformers_")
                        if not fitted:
                            logging.warning("Skipping unfitted primary pipeline: %s", p)
                            continue
                        logging.info("Full pipeline loaded successfully from '%s'.", p)
                        logging.info("Metadata loaded successfully from '%s'.", meta_path)
                        return _coerce_pipeline_object(obj), metadata
                    except Exception:
                        continue
                # Fallback to stem-matched pipeline_<stem>.joblib
                stem = os.path.splitext(os.path.basename(meta_path))[0].replace("meta_", "", 1)
                cand = os.path.join(arts, f"pipeline_{stem}.joblib")
                if os.path.exists(cand):
                    # Skip Ridge/quantile as primary
                    bn = os.path.basename(cand)
                    if bn.startswith("pipeline_Ridge_") or bn.startswith("pipeline_q"):
                        continue
                    try:
                        obj = joblib.load(cand)
                        # Fittedness guard
                        pre = getattr(obj, "preprocessor", None)
                        if pre is None and hasattr(obj, "named_steps"):
                            pre = obj.named_steps.get("preprocessor")
                        fitted = hasattr(pre, "transformers_")
                        if not fitted:
                            logging.warning("Skipping unfitted primary pipeline: %s", cand)
                            continue
                        logging.info("Full pipeline loaded successfully from '%s'.", cand)
                        logging.info("Metadata loaded successfully from '%s'.", meta_path)
                        return _coerce_pipeline_object(obj), metadata
                    except Exception:
                        continue
    except Exception:
        pass

    # Next: explicit candidate list path is already handled above

    # Last fallback: prefer primary CatBoost pipelines with matching meta; exclude Ridge/quantiles
    try:
        arts = os.path.join(os.getcwd(), "artifacts")
        if os.path.isdir(arts):
            pipes = []
            for n in os.listdir(arts):
                if not (n.startswith("pipeline_") and n.endswith(".joblib")):
                    continue
                if n.startswith("pipeline_Ridge_") or n.startswith("pipeline_q"):
                    continue
                stem = os.path.splitext(n)[0].replace("pipeline_", "", 1)
                meta_path = os.path.join(arts, f"meta_{stem}.json")
                if not os.path.exists(meta_path):
                    continue
                path = os.path.join(arts, n)
                pipes.append((os.path.getmtime(path), path, meta_path))
            pipes.sort(key=lambda t: t[0], reverse=True)
            for _, p, m in pipes:
                try:
                    obj = joblib.load(p)
                    with open(m, "r") as f:
                        metadata = json.load(f)
                    logging.info("Full pipeline loaded successfully from '%s'.", p)
                    logging.info("Metadata loaded successfully from '%s'.", m)
                    return _coerce_pipeline_object(obj), metadata
                except Exception:
                    continue
    except Exception:
        pass

    # Final fallback to model dir
    try:
        p = os.path.join(MODEL_DIR, "full_pipeline.joblib")
        m = os.path.join(MODEL_DIR, "model_metadata.json")
        obj = joblib.load(p)
        metadata: Dict[str, object] = {}
        if os.path.exists(m):
            with open(m, "r") as f:
                metadata = json.load(f)
        logging.info("Full pipeline loaded successfully from '%s'.", p)
        if metadata:
            logging.info("Metadata loaded successfully from '%s'.", m)
        return _coerce_pipeline_object(obj), metadata
    except Exception as exc:
        logging.error("Error loading full pipeline and metadata: %s", exc)
        logging.debug(traceback.format_exc())
        sys.exit(1)

def _load_extra_pipelines(metadata: dict) -> Dict[str, object]:
    """Load additional base pipelines (e.g., 'Ridge') for blending.

    For sklearn Pipelines, return the full pipeline unmodified so all steps run.
    For dict artifacts, coerce to a LoadedPipelineAdapter.
    Skip legacy Ridge artifacts that lack an embedded preprocessor (matrix-in).
    """
    out: Dict[str, object] = {}
    try:
        from sklearn.pipeline import Pipeline as SkPipeline  # type: ignore
    except Exception:
        SkPipeline = None  # type: ignore
    try:
        bundle = (metadata or {}).get("serve_bundle") or {}
        base = bundle.get("base_pipelines") or {}
        for name, path in base.items():
            label = str(name)
            if not path or not os.path.exists(path):
                continue
            try:
                obj = joblib.load(path)
            except Exception:
                continue
            try:
                # Prefer returning full sklearn Pipelines unmodified so internal steps run
                if SkPipeline is not None and isinstance(obj, SkPipeline) and hasattr(obj, "named_steps"):
                    # For Ridge, require an embedded preprocessor (DF-in). Skip matrix-in artifacts.
                    if label.lower() == "ridge" and "preprocessor" not in getattr(obj, "named_steps", {}):
                        logging.warning("Skipping Ridge extra pipeline (no embedded preprocessor): %s", path)
                        continue
                    out[label] = obj
                    continue
                # Coerce dict-based bundles to an adapter
                if isinstance(obj, dict) and ("preprocessor" in obj and "model" in obj):
                    out[label] = _coerce_pipeline_object(obj)
                    continue
                # Fallback: keep as-is; _predict_extra will try both DF and matrix inputs
                out[label] = obj
            except Exception:
                continue
    except Exception:
        return {}
    return out

def _predict_blended(prepared_df, full_pipeline, metadata, extra_pipes: Dict[str, object]) -> np.ndarray:
    """Return blended predictions if blend_head is present; else CatBoost only."""
    try:
        # CRITICAL: Align prepared_df to union of training metadata feature_order and preprocessor selection
        feature_order = _serve_feature_order(metadata, full_pipeline)
        if feature_order:
            # Optional EXCLUDE_CAT_FEATURES drop: only when env overrides allowed AND not part of training schema
            try:
                allow_env = os.getenv("SERVE_ALLOW_ENV_OVERRIDES", "1").lower() in {"1", "true", "yes"}
                if allow_env:
                    raw_exc = os.getenv("EXCLUDE_CAT_FEATURES", "") or ""
                    exc = {s.strip().lower() for s in re.split(r"[;]", raw_exc) if s.strip()}
                    if exc:
                        lower_map = {str(c).lower(): c for c in prepared_df.columns}
                        # Do not drop columns that exist in the training schema (protected)
                        protected = {str(c).lower() for c in feature_order}
                        drop_cols = [
                            lower_map[k]
                            for k in exc
                            if (k in lower_map) and (k not in protected)
                        ]
                        if drop_cols:
                            prepared_df = prepared_df.drop(columns=drop_cols, errors="ignore")
            except Exception:
                pass

            # Build a train→serve schema: metadata.feature_order ∪ columns the preprocessor selected during training
            try:
                for c in feature_order:
                    if c not in prepared_df.columns:
                        prepared_df[c] = np.nan
            except Exception:
                pass
            # Only keep and reorder columns according to the serve schema
            cols_to_use = [c for c in feature_order if c in prepared_df.columns]
            total = len(feature_order)
            used = len(cols_to_use)
            missing = total - used
            prepared_df = prepared_df[cols_to_use]

            # One-time diagnostics to reduce schema confusion in logs
            try:
                need = {"position", "wall_construction", "roof_construction"}
                has = sorted(c for c in need if c in prepared_df.columns)
                logging.info(
                    "Serve schema union prepared | union_len=%d | needed_cats=%s | present=%s",
                    len(feature_order),
                    sorted(list(need)),
                    has,
                )
            except Exception:
                pass

            if used != total:
                missing_ratio = 1.0 - (used / max(1, total))
                logging.warning(
                    "Serve schema mismatch: using %d/%d features present in serve_feature_order",
                    used,
                    total,
                )
                # Strict abort when divergence beyond threshold
                try:
                    strict_env = os.getenv("SERVE_STRICT_SCHEMA", "0").lower() in {"1", "true", "yes"}
                    thresh = float(os.getenv("SERVE_SCHEMA_THRESHOLD", "0.20") or 0.20)
                except Exception:
                    strict_env, thresh = False, 0.20
                if strict_env and (missing_ratio > thresh):
                    raise RuntimeError(
                        f"Serve schema mismatch: missing {missing}/{total} unioned columns (> {thresh:.2f} threshold)."
                    )
        
        bh = (metadata or {}).get("blend_head") or {}
        # Honor env overrides only when allowed
        _allow_env = os.getenv("SERVE_ALLOW_ENV_OVERRIDES", "1").lower() in {"1", "true", "yes"}
        def _cb_point() -> np.ndarray:
            # Prefer averaging seed pipelines when available
            try:
                seeds = _load_seed_pipelines(metadata)
            except Exception:
                seeds = []
            if seeds:
                parts: List[np.ndarray] = []
                for obj in seeds:
                    try:
                        parts.append(np.asarray(obj.predict(prepared_df)).reshape(-1))
                    except Exception:
                        continue
                if parts:
                    P = np.vstack(parts).T
                    return np.mean(P, axis=1)
            return np.asarray(full_pipeline.predict(prepared_df)).reshape(-1)
        if not bh or "weights" not in bh or "labels" not in bh:
            return _cb_point()
        # Metadata-gated blending with optional env override
        meta_blend = bool((metadata or {}).get("apply_blend_at_serve", True))
        if _allow_env:
            try:
                force_cb = os.getenv("BLEND_FORCE_CATBOOST", "0").lower() in {"1", "true", "yes"}
                env_blend_on = os.getenv("BLEND_AT_SERVE", "1").lower() in {"1", "true", "yes"}
            except Exception:
                force_cb, env_blend_on = False, True
            if force_cb or not env_blend_on:
                logging.info("Blend disabled by env override; using CatBoost-only")
                return _cb_point()
            # If metadata disabled blend but env allows it, proceed with blend
            if (not meta_blend) and env_blend_on:
                logging.info("Blend override: env BLEND_AT_SERVE=1 with SERVE_ALLOW_ENV_OVERRIDES=1 — attempting blend anyway")
        else:
            if not meta_blend:
                logging.info("Blend disabled by metadata.apply_blend_at_serve=false; using CatBoost-only")
                return _cb_point()
        labels: List[str] = list(bh["labels"])
        weights = np.asarray(bh["weights"], float).reshape(-1)
        preds: List[np.ndarray] = []
        # Precompute preprocessor transform once (used by legacy matrix-in Ridge artifacts)
        X_for_ridge = None
        try:
            if hasattr(full_pipeline, "preprocessor"):
                X_for_ridge = full_pipeline.preprocessor.transform(prepared_df)  # legacy matrix-in support
        except Exception:
            X_for_ridge = None

        # Helper: choose the correct input for an extra pipeline by signature + introspection
        def _predict_extra(label: str) -> np.ndarray:
            obj = extra_pipes[label]
            expect_df = None
            try:
                sig = ((metadata or {}).get("serve_bundle") or {}).get("ridge_signature") or {}
                if str(label).lower() == "ridge" and isinstance(sig, dict):
                    expect_df = bool(sig.get("expects_dataframe", False))
            except Exception:
                expect_df = None
            # detect if the object has an internal preprocessor (DF-in) via common attributes
            uses_preproc = False
            try:
                uses_preproc = bool(getattr(obj, "preprocessor", None))
                if not uses_preproc:
                    ns = getattr(obj, "named_steps", {})
                    if hasattr(ns, "get"):
                        uses_preproc = ns.get("preprocessor") is not None
            except Exception:
                uses_preproc = False
            # Decide candidate inputs by priority
            if expect_df is True:
                candidates = [prepared_df, X_for_ridge]
            elif expect_df is False:
                candidates = [X_for_ridge, prepared_df]
            else:
                candidates = [prepared_df, X_for_ridge] if uses_preproc else [X_for_ridge, prepared_df]
            for Xin in candidates:
                if Xin is None:
                    continue
                try:
                    return np.asarray(obj.predict(Xin)).reshape(-1)
                except Exception:
                    continue
            # Last resort
            return np.asarray(obj.predict(prepared_df)).reshape(-1)
        for lab in labels:
            if lab == "CatBoost":
                # Prefer averaging seed pipelines when available; fallback to main pipeline
                cb_pred = None
                try:
                    seeds = _load_seed_pipelines(metadata)
                except Exception:
                    seeds = []
                if seeds:
                    parts: List[np.ndarray] = []
                    for obj in seeds:
                        try:
                            parts.append(np.asarray(obj.predict(prepared_df)).reshape(-1))
                        except Exception:
                            continue
                    if parts:
                        P = np.vstack(parts).T
                        cb_pred = np.mean(P, axis=1)
                if cb_pred is None:
                    cb_pred = np.asarray(full_pipeline.predict(prepared_df)).reshape(-1)
                preds.append(cb_pred)
            elif lab in extra_pipes:
                try:
                    if lab.lower() == "ridge":
                        pred_r = _predict_extra(lab)
                        # Optional delta gating (disabled by default)
                        if os.getenv("ENABLE_BLEND_DELTA_GATES", "0").lower() in {"1", "true", "yes"} and len(preds) > 0:
                            try:
                                cb_est = preds[0]
                                abs_max = float(os.getenv("BLEND_DELTA_ABS_MAX", "300"))
                                rel_max = float(os.getenv("BLEND_DELTA_REL_MAX", "0.35"))
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    rel = np.where(np.isfinite(cb_est) & (cb_est != 0), np.abs(pred_r - cb_est) / np.abs(cb_est), 0.0)
                                mask_ok = (np.abs(pred_r - cb_est) <= abs_max) & (rel <= rel_max)
                                pred_r = np.where(mask_ok, pred_r, cb_est)
                            except Exception:
                                pass
                        preds.append(pred_r)
                    else:
                        preds.append(np.asarray(extra_pipes[lab].predict(prepared_df)).reshape(-1))
                except Exception as exc:
                    logging.warning("Blend extra '%s' failed; skipping: %s", lab, exc)
        if not preds:
            return np.asarray(full_pipeline.predict(prepared_df)).reshape(-1)
        P = np.vstack(preds).T  # (n, m)
        w = weights[: P.shape[1]]
        if w.sum() <= 0:
            w = np.ones_like(w) / max(1, len(w))
        # Optional segment-aware gating for Ridge (Bed >= ridge_min_bed)
        try:
            flags = (metadata or {}).get("flags") or {}
            ridge_min_bed = int(flags.get("ridge_min_bed", 0) or 0)
        except Exception:
            ridge_min_bed = 0
        # Env override for serve-time gating (RIDGE_MIN_BED_SERVE or RIDGE_MIN_BED)
        try:
            env_gate = os.getenv("RIDGE_MIN_BED_SERVE", os.getenv("RIDGE_MIN_BED", None))
            if env_gate is not None and str(env_gate).strip() != "":
                ridge_min_bed = max(ridge_min_bed, int(env_gate))
        except Exception:
            pass
        if ridge_min_bed > 0 and "Ridge" in labels and "Bed" in prepared_df.columns and "CatBoost" in labels:
            try:
                ridge_idx = labels.index("Ridge")
                cb_idx = labels.index("CatBoost")
                bed_vals = pd.to_numeric(prepared_df["Bed"], errors="coerce").to_numpy()
                mask = np.isfinite(bed_vals) & (bed_vals >= ridge_min_bed)
                # Combine row-wise: Ridge contributes only where mask is true
                wnorm = w / w.sum()
                base = (P * wnorm.reshape(1, -1)).sum(axis=1)
                # Adjust for rows where Ridge should be gated out
                if ridge_idx < P.shape[1] and cb_idx < P.shape[1]:
                    delta = wnorm[ridge_idx] * (P[:, ridge_idx] - P[:, cb_idx])
                    base = base - (~mask).astype(float) * delta
                return base.reshape(-1)
            except Exception:
                pass
        # Default global blend
        return (P @ (w / w.sum())).reshape(-1)
    except Exception:
        return np.asarray(full_pipeline.predict(prepared_df)).reshape(-1)

    if not candidates:
        logging.error(
            "No pipeline artifacts available in artifacts/ or models/ directories."
        )

    for path, exc, tb in errors:
        logging.error("Attempt to load pipeline '%s' failed: %s", path, exc)
        logging.debug(tb)

    logging.error(
        "Error loading full pipeline and metadata: no valid pipeline artifacts found."
    )
    sys.exit(1)


_QPIPE_CACHE: Dict[str, Dict[float, LoadedPipelineAdapter]] = {}


def _load_quantile_pipelines(metadata: dict):
    """Load and cache quantile pipelines keyed by a metadata fingerprint."""
    try:
        q = (metadata or {}).get("quantile_pipelines") or {}
        items = []
        for k, p in sorted(q.items(), key=lambda kv: float(kv[0])):
            try:
                m = os.path.getmtime(p) if p and os.path.exists(p) else 0.0
            except Exception:
                m = 0.0
            items.append((float(k), str(p), float(m)))
        fp = hashlib.sha256(repr(items).encode("utf-8")).hexdigest()
        if fp in _QPIPE_CACHE:
            return _QPIPE_CACHE[fp]
        out: Dict[float, LoadedPipelineAdapter] = {}
        for k, p in q.items():
            if p and os.path.exists(p):
                obj = joblib.load(p)
                out[float(k)] = LoadedPipelineAdapter(obj["preprocessor"], obj["model"])
        _QPIPE_CACHE[fp] = out
        return out
    except Exception:
        return {}


def calculate_vwap(data, bed_type, last_rental_date):
    try:
        # Ensure 'Bed' is of string type
        data["Bed"] = data["Bed"].astype(str)

        # Filter data for the given bed_type and before last_rental_date
        bed_data = data[
            (data["Bed"] == bed_type) & (data["Last Rental Date"] < last_rental_date)
        ].copy()
        bed_data = bed_data.sort_values("Last Rental Date")

        if bed_data.empty:
            logging.warning(
                f"No historical data available for VWAP calculation for Bed='{bed_type}' before {last_rental_date.date()}. Falling back to all beds."
            )
            fallback = data[data["Last Rental Date"] < last_rental_date].copy()
            if fallback.empty:
                logging.error("No historical rental data available to compute VWAP.")
                return None, None
            bed_data = fallback

        # Log the number of records
        logging.info(
            f"Number of historical records for Bed='{bed_type}' (after fallback handling): {len(bed_data)}"
        )

        # Calculate VWAP_3M
        start_date_3M = last_rental_date - pd.DateOffset(months=3)
        vw_data_3M = bed_data[bed_data["Last Rental Date"] >= start_date_3M]
        if len(vw_data_3M) >= 5:
            VWAP_3M = vw_data_3M["Last Rental Price"].mean()
        else:
            VWAP_3M = bed_data["Last Rental Price"].mean()  # Fallback to all available data

        # Calculate VWAP_12M
        start_date_12M = last_rental_date - pd.DateOffset(months=12)
        vw_data_12M = bed_data[bed_data["Last Rental Date"] >= start_date_12M]
        if len(vw_data_12M) >= 20:
            VWAP_12M = vw_data_12M["Last Rental Price"].mean()
        else:
            VWAP_12M = bed_data["Last Rental Price"].mean()  # Fallback to all available data

        return VWAP_3M, VWAP_12M

    except Exception as e:
        logging.error(f"Error calculating VWAP: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def _compute_suburb90d_median(
    hdata: pd.DataFrame, suburb: str, asof_date: pd.Timestamp
) -> float | None:
    try:
        df = hdata[
            hdata["Suburb"].astype(str).str.upper() == str(suburb).upper()
        ].copy()
        if df.empty:
            return None
        df = df[df["Last Rental Date"] < asof_date]
        if df.empty:
            return None
        df = df.sort_values("Last Rental Date").set_index("Last Rental Date")
        med = df["Last Rental Price"].rolling("90D", min_periods=1).median().iloc[-1]
        return float(med) if pd.notna(med) else None
    except Exception:
        return None


def predict_rental_price(
    features: pd.DataFrame, full_pipeline, metadata, historical_data: pd.DataFrame
):
    try:
        # Derive serve-time log policy from training metadata
        global _SERVE_KEEP_RAW_WHEN_LOG
        try:
            _SERVE_KEEP_RAW_WHEN_LOG = bool(
                ((metadata or {}).get("log_policy") or {}).get("keep_raw_when_log", False)
            )
        except Exception:
            _SERVE_KEEP_RAW_WHEN_LOG = None
        logging.info(f"Initial features:\n{features}\n")

        # Prepare features using shared logic (ensures medians/LOO/rank are computed consistently)
        (
            prepared,
            expected_cols,
            missing_before,
            categorical_features,
        ) = _prepare_features_for_model(
            features, full_pipeline, metadata, historical_data
        )

        # Report: show base training features, their values, and whether derived/auto-filled
        try:
            numeric_features = metadata.get("num_features") or []
            report_cols = list(
                dict.fromkeys(list(numeric_features) + list(categorical_features))
            )
            rows = []
            for col in report_cols:
                v = prepared[col].iloc[0] if col in prepared.columns else np.nan
                is_derived = col.startswith("log_") or col in {
                    "Year",
                    "Month",
                    "WeekOfYear",
                    "DayOfWeek",
                    "Month_sin",
                    "Month_cos",
                }
                auto = (col in missing_before) and (not is_derived)
                rows.append(
                    (col, None if pd.isna(v) else v, bool(auto), bool(is_derived))
                )
            rows.sort(key=lambda x: x[0].lower())
            name_w = max(8, *(len(r[0]) for r in rows))
            header = f"\nFeature inputs used (first row):\n"
            lines = []
            for name, val, auto, derived in rows:
                val_str = str(val)
                lines.append(
                    f"  {name.ljust(name_w)} : {val_str}    (derived: {'Yes' if derived else 'No'} | auto-filled: {'Yes' if auto else 'No'})"
                )
            logging.info(header + "\n".join(lines))
            freq_map = {}
            try:
                if hasattr(full_pipeline, "preprocessor"):
                    freq_map = _get_frequent_levels_from_preprocessor(
                        full_pipeline.preprocessor
                    )
            except Exception:
                freq_map = {}
            suspicious = [
                (n, v, a, d)
                for (n, v, a, d) in rows
                if (a and not d)
                or (isinstance(v, str) and v in {"UNKNOWN", "OTHER", "MISSING"})
            ]
            unseen = []
            for n, v, a, d in rows:
                if (
                    n in categorical_features
                    and isinstance(v, str)
                    and v not in {"UNKNOWN", "OTHER", "MISSING"}
                ):
                    lvlset = freq_map.get(n)
                    if lvlset is not None and len(lvlset) > 0 and (v not in lvlset):
                        unseen.append((n, v))
            if suspicious:
                logging.warning(
                    "Potential issues in inputs (unknown/other/missing or auto-filled):\n%s",
                    "\n".join(
                        [
                            f"  {n.ljust(name_w)} : {v}    (derived: {'Yes' if d else 'No'} | auto-filled: {'Yes' if a else 'No'})"
                            for n, v, a, d in suspicious
                        ]
                    ),
                )
            if unseen:
                logging.warning(
                    "Unseen categories (will be treated as OTHER – check spelling/valid labels):\n%s",
                    "\n".join([f"  {n.ljust(name_w)} : {v}" for n, v in unseen]),
                )
            # Schema guard
            try:
                fill_count = len([c for c in missing_before if c in prepared.columns])
                total_req = max(1, len(expected_cols))
                fill_ratio = fill_count / total_req
                if fill_ratio > 0.20:
                    logging.warning(
                        "Schema guard: %.0f%% (%d/%d) of required columns were auto-filled. Review input schema.",
                        fill_ratio * 100.0,
                        fill_count,
                        total_req,
                    )
                    logging.info("Auto-filled columns: %s", ", ".join(missing_before))
                try:
                    strict = args.strict_schema  # type: ignore[name-defined]
                    thresh = args.schema_threshold if args.schema_threshold is not None else 0.20  # type: ignore[name-defined]
                    if strict and fill_ratio > float(thresh):
                        raise RuntimeError(
                            f"Strict schema: auto-fill ratio {fill_ratio:.2f} exceeds threshold {float(thresh):.2f}"
                        )
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        extras = _load_extra_pipelines(metadata)
        predicted_price = _predict_blended(prepared, full_pipeline, metadata, extras)
        return float(predicted_price[0])

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def predict_with_interval(
    features: pd.DataFrame, full_pipeline, metadata, historical_data: pd.DataFrame
) -> Tuple[float, float | None, float | None]:
    # Ensure metadata-derived log policy applies within nested calls
    global _SERVE_KEEP_RAW_WHEN_LOG
    try:
        _SERVE_KEEP_RAW_WHEN_LOG = bool(
            ((metadata or {}).get("log_policy") or {}).get("keep_raw_when_log", False)
        )
    except Exception:
        _SERVE_KEEP_RAW_WHEN_LOG = None
    point = predict_rental_price(features, full_pipeline, metadata, historical_data)
    qpipes = _load_quantile_pipelines(metadata)
    lo = hi = None
    try:
        if qpipes:
            prepared, _, _, _ = _prepare_features_for_model(
                features, full_pipeline, metadata, historical_data
            )
            lo_a = (
                0.1
                if 0.1 in qpipes
                else max([a for a in qpipes if a < 0.5], default=None)
            )
            hi_a = (
                0.9
                if 0.9 in qpipes
                else min([a for a in qpipes if a > 0.5], default=None)
            )
            if lo_a is not None and hi_a is not None:
                lo = float(qpipes[lo_a].predict(prepared)[0])
                hi = float(qpipes[hi_a].predict(prepared)[0])
                cqr = (metadata or {}).get("conformal") or {}
                qhat = float(cqr.get("qhat", 0.0))
                if qhat > 0:
                    lo -= qhat
                    hi += qhat
                if lo > hi:
                    lo, hi = min(lo, hi), max(lo, hi)
    except Exception:
        lo = hi = None
    return point, lo, hi


# --- NEW: vectorized batch inference with PIs ---
def predict_many(
    features: pd.DataFrame,
    full_pipeline,
    metadata: dict,
    historical_data: pd.DataFrame,
    return_intervals: bool = True,
) -> pd.DataFrame:
    """
    Vectorized scoring for a DataFrame.
    Returns a DataFrame with columns: point[, lo, hi] aligned to `features.index`.
    """
    # Apply metadata-aligned log policy for batch scoring
    global _SERVE_KEEP_RAW_WHEN_LOG
    try:
        _SERVE_KEEP_RAW_WHEN_LOG = bool(
            ((metadata or {}).get("log_policy") or {}).get("keep_raw_when_log", False)
        )
    except Exception:
        _SERVE_KEEP_RAW_WHEN_LOG = None

    prepared, _, _, _ = _prepare_features_for_model(
        features, full_pipeline, metadata, historical_data
    )
    # Use blending/seed averaging when available for batch as well
    extras = _load_extra_pipelines(metadata)
    point = _predict_blended(prepared, full_pipeline, metadata, extras)
    out = pd.DataFrame({"point": point}, index=features.index)
    if not return_intervals:
        return out
    qpipes = _load_quantile_pipelines(metadata)
    if not qpipes:
        return out
    lo_a = 0.1 if 0.1 in qpipes else max([a for a in qpipes if a < 0.5], default=None)
    hi_a = 0.9 if 0.9 in qpipes else min([a for a in qpipes if a > 0.5], default=None)
    if lo_a is None or hi_a is None:
        return out
    lo = np.asarray(qpipes[lo_a].predict(prepared)).reshape(-1)
    hi = np.asarray(qpipes[hi_a].predict(prepared)).reshape(-1)
    cqr = (metadata or {}).get("conformal") or {}
    qhat = float(cqr.get("qhat", 0.0))
    if qhat > 0:
        lo = lo - qhat
        hi = hi + qhat
    out["lo"] = np.minimum(lo, hi)
    out["hi"] = np.maximum(lo, hi)
    return out


def _load_seed_pipelines(metadata: dict) -> List[object]:
    out: List[object] = []
    try:
        bundle = (metadata or {}).get("serve_bundle") or {}
        seeds = bundle.get("catboost_seed_pipelines") or []
        if not isinstance(seeds, (list, tuple)):
            return []
        for path in seeds:
            if not path or not os.path.exists(path):
                continue
            try:
                obj = joblib.load(path)
                out.append(_coerce_pipeline_object(obj))
            except Exception:
                continue
    except Exception:
        return []
    return out

def display_feature_importances(full_pipeline):
    try:
        # Extract the model from the pipeline
        model = None
        if hasattr(full_pipeline, "named_steps") and "model" in getattr(
            full_pipeline, "named_steps"
        ):
            model = full_pipeline.named_steps["model"]
        elif hasattr(full_pipeline, "model"):
            model = getattr(full_pipeline, "model")
        else:
            model = full_pipeline

        # Get feature importances from the model (prettified)
        # CatBoost prettified importances if available, else skip
        if hasattr(model, "get_feature_importance"):
            feature_importances = model.get_feature_importance(prettified=True)
        else:
            feature_importances = None

        # Log the feature importances
        if feature_importances is not None:
            logging.info("\nModel Feature Importances:")
            logging.info(feature_importances.to_string(index=False))

    except Exception as e:
        logging.error(f"Error displaying feature importances: {str(e)}")
        logging.error(traceback.format_exc())


def _bucketize_feature_name(name: str) -> str:
    """Conservative bucketing to avoid double-counting.

    - Keep log_* separate from base (don't merge 'log_Car' into 'Car').
    - Keep time parts separate (Year, Month, WeekOfYear, DayOfWeek, Month_sin, Month_cos).
    - Map explicit Suburb engineered names to 'Suburb'.
    - Split OHE-like names by '__' only if present (kept as base prefix).
    """
    if name in {
        "Suburb_LOO",
        "SuburbMonth_LOO",
        "Rank30dX12m",
        "Suburb90dMedian",
        "SuburbBed90dMedian",
        "Suburb365dMedian",
    }:
        return "Suburb"
    if "__" in name:
        # Prefer suffix after transformer prefix, e.g., 'num__Capital Value' → 'Capital Value'
        return name.split("__", 1)[1]
    return name


def _load_importance_df_from_meta(metadata) -> pd.DataFrame | None:
    try:
        imp_path = metadata.get("importance_csv")
        if not imp_path or not os.path.exists(imp_path):
            return None
        df = pd.read_csv(imp_path)
        if "feature" in df.columns and (
            "gain" in df.columns or "Importances" in df.columns
        ):
            if "gain" not in df.columns:
                df = df.rename(columns={"Importances": "gain"})
            return df[["feature", "gain"]]
        return None
    except Exception:
        return None


def display_grouped_importance(metadata):
    df = _load_importance_df_from_meta(metadata)
    if df is None or df.empty:
        logging.info("Grouped importance: no importance CSV available.")
        return
    try:
        df["bucket"] = df["feature"].astype(str).map(_bucketize_feature_name)
        grouped = df.groupby("bucket")["gain"].sum().sort_values(ascending=False)
        logging.info(
            "\nGrouped importances (by base feature):\n%s", grouped.to_string()
        )
    except Exception:
        logging.warning("Grouped importance aggregation failed.")


def display_row_shap(
    full_pipeline, metadata, features: pd.DataFrame, historical_data: pd.DataFrame
):
    try:
        # Resolve model
        model = None
        if hasattr(full_pipeline, "named_steps") and "model" in getattr(
            full_pipeline, "named_steps"
        ):
            model = full_pipeline.named_steps["model"]
        elif hasattr(full_pipeline, "model"):
            model = getattr(full_pipeline, "model")
        else:
            model = full_pipeline
        if hasattr(model, "base"):
            model_cat = model.base
        else:
            model_cat = model

        # Prepare features using the same backfill logic as prediction
        prepared, _, _, _ = _prepare_features_for_model(
            features, full_pipeline, metadata, historical_data
        )
        # Transform features using the trained preprocessor; skip CatBoost SHAP if unavailable
        if hasattr(full_pipeline, "preprocessor"):
            try:
                X = full_pipeline.preprocessor.transform(prepared)
            except Exception:
                # If transform fails (e.g., unfitted preprocessor in rare cases), bail to fallback explainer
                X = None
        else:
            X = None
        # Ensure dense for single-row SHAP
        try:
            import scipy.sparse as _sp  # type: ignore

            if _sp.issparse(X):
                X = X.toarray()
        except Exception:
            pass

        # Feature names hint
        names_hint: List[str] | None = None
        imp_df = _load_importance_df_from_meta(metadata)
        if imp_df is not None and not imp_df.empty:
            names = imp_df["feature"].astype(str).tolist()
            # SHAP length must match names length; will check later
            names_hint = names

        # Compute SHAP using CatBoost if available
        shap_vals = None
        try:
            if X is not None:
                from catboost import Pool  # type: ignore

                # Detect categorical columns directly from the transformed matrix X.
                # A column is considered categorical if any sampled value is a non-empty string.
                def _detect_cat_idx_from_matrix(mat: np.ndarray, max_rows: int = 64) -> list[int]:
                    try:
                        if mat is None or getattr(mat, "ndim", 0) != 2:
                            return []
                        n = int(min(max_rows, mat.shape[0]))
                        out: list[int] = []
                        for j in range(int(mat.shape[1])):
                            col = mat[:n, j]
                            is_cat = False
                            for v in col:
                                if isinstance(v, (str, np.str_, bytes, np.bytes_)) and str(v) != "":
                                    is_cat = True
                                    break
                            if is_cat:
                                out.append(j)
                        return out
                    except Exception:
                        return []

                cat_idx = _detect_cat_idx_from_matrix(X)
                # Fallback to metadata-derived indices if detection is empty
                if not cat_idx:
                    cat_idx = list((metadata.get("cat_idx_final") or []))
                    if not cat_idx:
                        numf = metadata.get("num_features") or []
                        catf = metadata.get("cat_features") or []
                        cat_idx = list(range(len(numf), len(numf) + len(catf)))
                # Clip against current matrix width for safety
                try:
                    w = int(getattr(X, "shape", (0, 0))[1])
                    cat_idx = [int(i) for i in cat_idx if isinstance(i, (int, np.integer)) and 0 <= int(i) < w]
                except Exception:
                    cat_idx = list(cat_idx)

                # Optional debug
                try:
                    if os.getenv("PREDICT_DEBUG", "0").lower() in {"1", "true", "yes"}:
                        logging.info("SHAP debug: X.shape=%s | detected_cat_idx=%s", getattr(X, "shape", None), cat_idx)
                except Exception:
                    pass

                pool = Pool(X, cat_features=cat_idx if len(cat_idx) > 0 else None)
                shap_vals = model_cat.get_feature_importance(pool, type="ShapValues")
        except Exception as e:
            logging.error(f"Per-row SHAP (CatBoost) failed: {e}")
            logging.error(traceback.format_exc())
        import numpy as _np

        if shap_vals is None or len(shap_vals) == 0:
            # Fallback with shap.TreeExplainer
            try:
                import shap as _shap  # type: ignore

                expl = _shap.TreeExplainer(model_cat)
                sv = expl.shap_values(_np.array(X))
                if isinstance(sv, list):
                    sv = _np.array(sv[0])
                sv_feat = _np.array(sv)[0]
                # Names from metadata
                numf = metadata.get("num_features") or []
                catf = metadata.get("cat_features") or []
                feat_names = list(map(str, numf)) + list(map(str, catf))
                buckets: Dict[str, float] = {}
                for name, val in zip(feat_names, sv_feat):
                    b = _bucketize_feature_name(str(name))
                    buckets[b] = buckets.get(b, 0.0) + float(abs(val))
                items = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
                lines = [f"  {k.ljust(20)} : {v:.4f}" for k, v in items[:20]]
                logging.info(
                    "\nPer-row SHAP (TreeExplainer fallback) by base feature:\n%s",
                    "\n".join(lines),
                )
                return
            except Exception as e2:
                logging.error(f"Per-row SHAP fallback failed: {e2}")
                logging.error(traceback.format_exc())
                return
        sv = _np.array(shap_vals)
        # shape: (n_rows, n_features + 1) last col is expected value
        row = sv[0]
        sv_feat = row[:-1]

        # Build names list – prefer metadata order (num_features + cat_features).
        numf = metadata.get("num_features") or []
        catf = metadata.get("cat_features") or []
        composed = list(map(str, numf)) + list(map(str, catf))

        def _informative(ns):
            try:
                import re

                return any(re.search(r"[A-Za-z_ ]", str(n)) for n in ns)
            except Exception:
                return True

        if len(composed) == len(sv_feat) and _informative(composed):
            feat_names = composed
        elif (
            hasattr(model_cat, "feature_names_")
            and len(getattr(model_cat, "feature_names_")) == len(sv_feat)
            and _informative(getattr(model_cat, "feature_names_"))
        ):
            feat_names = list(getattr(model_cat, "feature_names_"))
        elif (
            names_hint and len(names_hint) == len(sv_feat) and _informative(names_hint)
        ):
            feat_names = names_hint
        else:
            feat_names = [f"f{i}" for i in range(len(sv_feat))]

        # 1) Top raw per-feature contributions (no grouping)
        raw_items = sorted(
            ((str(n), float(abs(v))) for n, v in zip(feat_names, sv_feat)),
            key=lambda x: x[1],
            reverse=True,
        )
        raw_lines = [f"  {k.ljust(30)} : {v:.4f}" for k, v in raw_items[:20]]
        logging.info(
            "\nPer-row SHAP (top features, no grouping):\n%s", "\n".join(raw_lines)
        )

        # 2) Conservatively grouped by base (no log_* merge)
        buckets: Dict[str, float] = {}
        for name, val in zip(feat_names, sv_feat):
            b = _bucketize_feature_name(str(name))
            buckets[b] = buckets.get(b, 0.0) + float(abs(val))
        grp_items = sorted(buckets.items(), key=lambda x: x[1], reverse=True)
        grp_lines = [f"  {k.ljust(20)} : {v:.4f}" for k, v in grp_items[:20]]
        logging.info(
            "\nPer-row SHAP (|contribution|) grouped conservatively:\n%s",
            "\n".join(grp_lines),
        )
    except Exception as e:
        logging.error(f"Per-row SHAP failed: {e}")
        logging.error(traceback.format_exc())


def main():
    try:
        # Optional CLI for manual overrides
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--capital-value",
            type=float,
            dest="capital_value",
            help="Override Capital Value for prediction",
        )
        parser.add_argument("--suburb", type=str, dest="suburb", help="Suburb (string)")
        parser.add_argument(
            "--property-type",
            type=str,
            dest="property_type",
            help="Property Type (string)",
        )
        parser.add_argument(
            "--category",
            type=str,
            dest="category",
            help="Category (string)",
        )
        parser.add_argument("--agency", type=str, dest="agency", help="Agency (string)")
        parser.add_argument("--agent", type=str, dest="agent", help="Agent (string)")
        parser.add_argument("--city", type=str, dest="city", help="City (string)")
        parser.add_argument(
            "--postcode", type=str, dest="postcode", help="Postcode (string)"
        )
        parser.add_argument(
            "--land-use", type=str, dest="land_use", help="Land Use (string)"
        )
        parser.add_argument(
            "--dev-zone", type=str, dest="dev_zone", help="Development Zone (string)"
        )
        parser.add_argument(
            "--street-address",
            type=str,
            dest="street_address",
            help="Street Address (string)",
        )
        parser.add_argument(
            "--land-value", type=float, dest="land_value", help="Land Value (number)"
        )
        parser.add_argument(
            "--improvement-value",
            type=float,
            dest="improvement_value",
            help="Improvement Value (number)",
        )
        parser.add_argument(
            "--garage-parks",
            type=float,
            dest="garage_parks",
            help="Garage Parks (number)",
        )
        parser.add_argument(
            "--offstreet-parks",
            type=float,
            dest="offstreet_parks",
            help="Offstreet Parks (number)",
        )
        parser.add_argument("--garage", type=str, dest="garage", help="Garage (string)")
        parser.add_argument(
            "--furnishings",
            type=str,
            dest="furnishings",
            help="Furnishings (string)",
        )
        parser.add_argument(
            "--furnishings-evidence",
            type=str,
            dest="furnishings_evidence",
            help="Furnishings evidence (string)",
        )
        parser.add_argument("--pets", type=str, dest="pets", help="Pets policy (string)")
        # Training-used construction/position categoricals (keep names in lower_snake case to match training)
        parser.add_argument(
            "--wall-construction",
            dest="wall_construction",
            type=str,
            help="Wall construction (e.g., BRICK, WEATHERBOARD)",
        )
        parser.add_argument(
            "--roof-construction",
            dest="roof_construction",
            type=str,
            help="Roof construction (e.g., TILE, IRON)",
        )
        parser.add_argument(
            "--position",
            dest="position",
            type=str,
            help="Position (e.g., STREET FRONT, REAR)",
        )
        parser.add_argument("--latitude", type=float, dest="latitude", help="Latitude override")
        parser.add_argument("--longitude", type=float, dest="longitude", help="Longitude override")
        parser.add_argument(
            "--valuation-date",
            type=str,
            dest="valuation_date",
            help="Valuation Date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--bed", type=str, dest="bed", help="Bed as string (e.g., '3')"
        )
        parser.add_argument("--bath", type=float, dest="bath", help="Bath count")
        parser.add_argument("--car", type=float, dest="car", help="Car count")
        parser.add_argument(
            "--land-sqm", type=float, dest="land_sqm", help="Land Size (sqm)"
        )
        parser.add_argument(
            "--floor-sqm", type=float, dest="floor_sqm", help="Floor Size (sqm)"
        )
        parser.add_argument(
            "--year-built", type=int, dest="year_built", help="Year Built"
        )
        parser.add_argument(
            "--dom", type=int, dest="days_on_market", help="Days on Market"
        )
        parser.add_argument(
            "--last-date",
            type=str,
            dest="last_date",
            help="Last Rental Date (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--strict-schema",
            action="store_true",
            dest="strict_schema",
            help="Fail if auto-fill ratio exceeds threshold",
        )
        parser.add_argument(
            "--schema-threshold",
            type=float,
            dest="schema_threshold",
            help="Auto-fill ratio threshold (0-1)",
        )
        # Visual neighbors (comparables) controls
        parser.add_argument(
            "--neighbors",
            type=int,
            dest="neighbors",
            default=int(os.getenv("NEIGHBORS", "0") or 0),
            help="Return up to K nearest properties as visual comparables (default 0 = disabled)",
        )
        parser.add_argument(
            "--neighbors-radius-km",
            type=float,
            dest="neighbors_radius_km",
            default=float(os.getenv("NEIGHBORS_RADIUS_KM", "0") or 0),
            help="Optional radius filter in km for neighbors (0=ignore)",
        )
        parser.add_argument(
            "--no-neighbors-prefer-same-bed",
            action="store_false",
            dest="neighbors_prefer_same_bed",
            default=True,
            help="Do not prefer same Bed type when selecting neighbors",
        )
        parser.add_argument("--help", action="store_true", dest="_help")
        args, _ = parser.parse_known_args()
        if getattr(args, "_help", False):
            print("Usage: py prediction.py [..existing args..] --neighbors 5 --neighbors-radius-km 3.0 [--no-neighbors-prefer-same-bed]")
            return
        # Load the full pipeline and metadata
        full_pipeline, metadata = load_full_pipeline_and_metadata()

        # Optionally display feature importances
        display_feature_importances(full_pipeline)
        display_grouped_importance(metadata)
        try:
            # Align with training default for log preferences
            os.environ.setdefault(
                "PREFER_LOG_FOR",
                "Capital Value,Land Value,Improvement Value,Land Size (sqm),Floor Size (sqm),Days on Market",
            )
        except Exception:
            pass

        # Load historical data for VWAP calculation
        historical_data = pd.read_csv(CLEANED_DATA_PATH)
        historical_data["Last Rental Date"] = pd.to_datetime(
            historical_data["Last Rental Date"]
        )
        historical_data["Last Rental Price"] = pd.to_numeric(
            historical_data["Last Rental Price"], errors="coerce"
        )
        # Ensure 'Bed' is of string type
        historical_data["Bed"] = historical_data["Bed"].astype(str)

        # Log the date range and unique 'Bed' values
        logging.info(
            f"Historical data date range: {historical_data['Last Rental Date'].min()} to {historical_data['Last Rental Date'].max()}"
        )
        logging.info(
            f"Unique 'Bed' values in historical data: {historical_data['Bed'].unique()}"
        )

        # Define base features
        base_features = {
            "Bed": args.bed or "3",
            "Bath": args.bath if args.bath is not None else 1,
            "Car": args.car if args.car is not None else 2,
            "Land Size (sqm)": args.land_sqm if args.land_sqm is not None else 769,
            "land_size": args.land_sqm if args.land_sqm is not None else 769,
            "Floor Size (sqm)": args.floor_sqm if args.floor_sqm is not None else 140,
            "floor_size": args.floor_sqm if args.floor_sqm is not None else 140,
            "Year Built": args.year_built if args.year_built is not None else 1940,
            "Days on Market": args.days_on_market
            if args.days_on_market is not None
            else 6,
            "Suburb": args.suburb or "BURWOOD, CHRISTCHURCH",
            "city": args.city or "Christchurch",
            "property_type2": args.property_type,
            "category": args.category,
            "agency": args.agency,
            "agent": args.agent,
            "postcode": args.postcode or "8083",
            "Street Address": args.street_address or "9 Achilles Street",
        "land_use": args.land_use,
        "development_zone": args.dev_zone,
        # Construction/position categoricals align with training metadata names (lower_snake case)
        "wall_construction": (args.wall_construction or "").strip() or None,
        "roof_construction": (args.roof_construction or "").strip() or None,
        "position": (args.position or "").strip() or None,
        "garage": _canonicalize_garage(args.garage),
        "Garage": _canonicalize_garage(args.garage),
            "garage_parks": args.garage_parks,
            "offstreet_parks": args.offstreet_parks,
            "furnishings": _canonicalize_furnishings(args.furnishings),
            "Furnishings": _canonicalize_furnishings(args.furnishings),
            "furnishings_evidence": args.furnishings_evidence,
            "pets": _canonicalize_pets(args.pets),
            "Pets": _canonicalize_pets(args.pets),
            "Owner Type": "Rented",
            "Active Listing": "No",
            "Land Value": args.land_value if args.land_value is not None else 375000,
            "Improvement Value": args.improvement_value,
            "Latitude": args.latitude,
            "Longitude": args.longitude,
            "Capital Value": 615000,  # Will be set in loop
            "Valuation Date": pd.to_datetime(args.valuation_date)
            if args.valuation_date
            else "1-Aug-22",
            "Last Rental Date": pd.to_datetime(args.last_date)
            if args.last_date
            else pd.to_datetime("2025-08-18"),
            "VWAP_3M": None,
            "VWAP_12M": None,
            "Percentage_Diff": None,
        }
        # Provide canonical aliases expected by training normalization
        base_features["Agency"] = base_features.get("agency")
        base_features["Agent"] = base_features.get("agent")
        base_features["Property Type"] = args.property_type
        base_features["Land Use"] = base_features.get("land_use")
        base_features["Development Zone"] = base_features.get("development_zone")
        base_features["City"] = base_features.get("city")

        base_features["Agency"] = base_features.get("agency")
        base_features["Agent"] = base_features.get("agent")
        base_features["Property Type"] = args.property_type
        base_features["Land Use"] = base_features.get("land_use")
        base_features["Development Zone"] = base_features.get("development_zone")
        base_features["City"] = base_features.get("city")
        # Get dynamic VWAP values
        VWAP_3M, VWAP_12M = calculate_vwap(
            historical_data,
            bed_type=base_features["Bed"],
            last_rental_date=base_features["Last Rental Date"],
        )

        if VWAP_3M is None or VWAP_12M is None:
            logging.error("Insufficient historical data to calculate VWAP features.")
            sys.exit(1)

        base_features["VWAP_3M"] = VWAP_3M
        base_features["VWAP_12M"] = VWAP_12M

        # Calculate Percentage_Diff
        base_features["Percentage_Diff"] = ((VWAP_3M - VWAP_12M) / VWAP_12M) * 100

        # Define Capital Value(s) to test or override via CLI
        if args.capital_value is not None:
            capital_values = [args.capital_value]
        else:
            capital_values = [550000, 645000, 1000000]  # -20%  # Original  # +20%

        logging.info("\nTesting different Capital Values:")
        for cv in capital_values:
            logging.info(f"\n{'='*50}")
            logging.info(f"Testing Capital Value: ${cv:,.2f}")
            logging.info(f"{'='*50}")

            # Create a copy of the base features and update 'Capital Value'
            features = base_features.copy()
            features["Capital Value"] = cv

            # Create DataFrame for prediction
            example_features = pd.DataFrame([features])

            # Predict rental price
            point, lo, hi = predict_with_interval(
                example_features, full_pipeline, metadata, historical_data
            )
            # Defer final printing until after neighbor adjustments
            final_lo, final_hi = lo, hi
            # Optional: nearest comparables (visual only)
            try:
                k = int(getattr(args, "neighbors", 0) or 0)
                if k > 0:
                    # Geocode if lat/lon missing
                    _ef = example_features.copy()
                    _ef = _geocode_missing_latlon(_ef)
                    q_lat = float(pd.to_numeric(_ef.get("Latitude"), errors="coerce").iloc[0]) if "Latitude" in _ef.columns else float("nan")
                    q_lon = float(pd.to_numeric(_ef.get("Longitude"), errors="coerce").iloc[0]) if "Longitude" in _ef.columns else float("nan")
                    if np.isfinite(q_lat) and np.isfinite(q_lon):
                        prefer_bed = bool(getattr(args, "neighbors_prefer_same_bed", True))
                        bed_str = str(example_features.get("Bed", _ef.get("Bed", "")).iloc[0]) if "Bed" in example_features.columns else None
                        rad = float(getattr(args, "neighbors_radius_km", 0) or 0)
                        rad = None if rad <= 0 else rad
                        nn_df = _nearest_properties(
                            historical_data,
                            query_lat=q_lat,
                            query_lon=q_lon,
                            k=k,
                            prefer_same_bed=prefer_bed,
                            bed_str=bed_str,
                            radius_km=rad,
                        )
                        if not nn_df.empty:
                            # Prefer richer columns when available
                            addr_col = next((c for c in ["Street Address", "Address", "Full_Address"] if c in nn_df.columns), None)
                            garage_col = next((c for c in ["Garage", "garage"] if c in nn_df.columns), None)
                            furn_col = next((c for c in ["Furnishings", "furnishings"] if c in nn_df.columns), None)

                            preferred = [
                                addr_col,
                                "Suburb",
                                "Latitude",
                                "Longitude",
                                "Bed",
                                "Bath",
                                "Car",
                                "Floor Size (sqm)",
                                "Land Size (sqm)",
                                garage_col,
                                furn_col,
                                "Capital Value",
                                "Last Rental Date",
                                "Last Rental Price",
                                "distance_km",
                            ]
                            keep_cols = [c for c in preferred if c and c in nn_df.columns]
                            if not keep_cols:
                                keep_cols = [c for c in ["Suburb", "Bed", "Last Rental Price", "distance_km"] if c in nn_df.columns]
                            show = nn_df[keep_cols].copy()
                            # Pretty print
                            print("Nearest comparables (by distance):")
                            try:
                                # Format distances and price
                                if "distance_km" in show.columns:
                                    show["distance_km"] = show["distance_km"].map(lambda x: f"{x:.2f}")
                                if "Last Rental Price" in show.columns:
                                    show["Last Rental Price"] = pd.to_numeric(show["Last Rental Price"], errors="coerce").map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                                # Coordinates
                                if "Latitude" in show.columns:
                                    show["Latitude"] = pd.to_numeric(show["Latitude"], errors="coerce").map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
                                if "Longitude" in show.columns:
                                    show["Longitude"] = pd.to_numeric(show["Longitude"], errors="coerce").map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
                                # Capital Value
                                if "Capital Value" in show.columns:
                                    show["Capital Value"] = pd.to_numeric(show["Capital Value"], errors="coerce").map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                                print(show.to_string(index=False))
                            except Exception:
                                # Fallback: dicts
                                print(show.to_dict(orient="records"))

                            # Leak-safe comparable predictions (do not use their price as a feature)
                            try:
                                nn_infer = nn_df.copy()
                                if "Last Rental Price" in nn_infer.columns:
                                    # Drop target to avoid any possibility of it being used downstream
                                    nn_infer = nn_infer.drop(columns=["Last Rental Price"], errors="ignore")
                                preds_df = predict_many(
                                    nn_infer,
                                    full_pipeline,
                                    metadata,
                                    historical_data,
                                    return_intervals=False,
                                )
                                pred_col = np.asarray(preds_df["point"]).reshape(-1)
                                actual_col = pd.to_numeric(nn_df.get("Last Rental Price"), errors="coerce").to_numpy().reshape(-1)
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    pct_diff = np.where(
                                        np.isfinite(actual_col) & (actual_col != 0),
                                        100.0 * (pred_col - actual_col) / actual_col,
                                        np.nan,
                                    )
                                comp_cols = [
                                    c
                                    for c in [
                                        addr_col,
                                        "Suburb",
                                        "Bed",
                                        "Bath",
                                        "Car",
                                        "Floor Size (sqm)",
                                        "Land Size (sqm)",
                                        "Last Rental Date",
                                        "Last Rental Price",
                                        "distance_km",
                                    ]
                                    if c and c in nn_df.columns
                                ]
                                comp_out = nn_df[comp_cols].copy()
                                comp_out["Predicted"] = pred_col
                                comp_out["Pct_Diff_%"] = pct_diff
                                print("\nComparable predictions (leak-safe; model does not use their price):")
                                try:
                                    if "Last Rental Price" in comp_out.columns:
                                        comp_out["Last Rental Price"] = pd.to_numeric(comp_out["Last Rental Price"], errors="coerce").map(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
                                    comp_out["Predicted"] = comp_out["Predicted"].map(lambda x: f"${x:,.0f}")
                                    comp_out["Pct_Diff_%"] = comp_out["Pct_Diff_%"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                                    print(comp_out.head(k).to_string(index=False))
                                except Exception:
                                    print(comp_out.head(k).to_dict(orient="records"))

                                # Summary across K comparables
                                try:
                                    valid = np.isfinite(pct_diff)
                                    n_valid = int(np.sum(valid))
                                    if n_valid > 0:
                                        mean_signed = float(np.nanmean(pct_diff[valid]))
                                        mean_abs = float(np.nanmean(np.abs(pct_diff[valid])))
                                        total_abs = float(np.nansum(np.abs(pct_diff[valid])))
                                        print(
                                            f"\nComparables summary: N={n_valid} | mean Δ%={mean_signed:.1f}% | mean |Δ|%={mean_abs:.1f}% | total |Δ|%={total_abs:.1f}%"
                                        )
                                        # Comparables-informed confidence interval (optional)
                                        try:
                                            base_lo, base_hi = lo, hi  # from predict_with_interval above
                                            base_hw = None
                                            if base_lo is not None and base_hi is not None:
                                                base_hw = float(max(float(point) - float(base_lo), float(base_hi) - float(point)))
                                            # Neighbor-based symmetric half-width from mean absolute pct error
                                            nei_hw = float((mean_abs / 100.0) * max(float(point), 1e-9))
                                            # Weight neighbor info by coverage (up to 0.6 influence)
                                            w_nei = 0.0
                                            try:
                                                w_nei = min(0.6, (n_valid / max(1, k)) * 0.6)
                                            except Exception:
                                                w_nei = 0.6
                                            w_base = 1.0 - w_nei
                                            new_hw = nei_hw if base_hw is None else (w_base * base_hw + w_nei * nei_hw)
                                            adj_lo = max(0.0, float(point) - new_hw)
                                            adj_hi = float(point) + new_hw
                                            print(f"Adjusted PI (comparables-informed): ${adj_lo:.0f}-${adj_hi:.0f}")
                                            # Percentile-based neighbor PI (more robust than mean)
                                            try:
                                                dist = pd.to_numeric(nn_df.get("distance_km"), errors="coerce").to_numpy().reshape(-1)
                                            except Exception:
                                                dist = np.full_like(pct_diff, np.nan, dtype=float)
                                            w = 1.0 / (1.0 + np.where(np.isfinite(dist), dist, 0.0))
                                            try:
                                                if bed_str is not None and "Bed" in nn_df.columns:
                                                    same_bed = (nn_df["Bed"].astype(str).to_numpy() == str(bed_str))
                                                    w = w * (1.25 * same_bed + (~same_bed))
                                            except Exception:
                                                pass
                                            abs_pct = np.abs(pct_diff)
                                            q = float(os.getenv("NEI_PI_PCTL", "0.80") or 0.80)
                                            pctl_abs = _weighted_quantile(abs_pct[valid], w[valid], q)
                                            if np.isfinite(pctl_abs):
                                                nei_hw_q = float((pctl_abs / 100.0) * max(float(point), 1e-9))
                                                max_w = float(os.getenv("NEI_PI_MAX_WEIGHT", "0.7") or 0.7)
                                                try:
                                                    w_nei_q = min(max_w, (n_valid / max(1, k)) * max_w)
                                                except Exception:
                                                    w_nei_q = max_w
                                                w_base_q = 1.0 - w_nei_q
                                                new_hw_q = nei_hw_q if base_hw is None else (w_base_q * base_hw + w_nei_q * nei_hw_q)
                                                adj_lo_q = max(0.0, float(point) - new_hw_q)
                                                adj_hi_q = float(point) + new_hw_q
                                                print(f"Adjusted PI (percentile-based): ${adj_lo_q:.0f}-${adj_hi_q:.0f}")
                                                min_k = int(os.getenv("NEI_PI_MIN_K", "3") or 3)
                                                if (os.getenv("FORCE_NEIGHBOR_PI", "0").lower() in {"1","true","yes"}) or (n_valid >= max(min_k, int(0.5 * max(1, k)))):
                                                    final_lo, final_hi = adj_lo_q, adj_hi_q
                                            # Adopt adjusted PI for the final message when coverage is sufficient
                                            try:
                                                if n_valid >= max(3, int(0.5 * max(1, k))):
                                                    final_lo, final_hi = adj_lo, adj_hi
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            except Exception as _cmp_exc:
                                logging.warning("Comparable predictions skipped: %s", _cmp_exc)
                    else:
                        logging.info("Neighbors disabled: missing/invalid coordinates for query.")
            except Exception:
                # Never fail prediction due to neighbors
                logging.debug("Neighbors computation skipped due to error.")
                try:
                    if final_lo is not None and final_hi is not None:
                        msg = f"Final predicted rental price: ${point:.2f}  (PI: ${final_lo:.0f}-${final_hi:.0f})"
                    else:
                        msg = f"Final predicted rental price: ${point:.2f}"
                    logging.info(msg)
                    print(msg)
                except Exception:
                    pass
            else:
                try:
                    if final_lo is not None and final_hi is not None:
                        msg = f"Final predicted rental price: ${point:.2f}  (PI: ${final_lo:.0f}-${final_hi:.0f})"
                    else:
                        msg = f"Final predicted rental price: ${point:.2f}"
                    logging.info(msg)
                    print(msg)
                except Exception:
                    pass
            # Show per-row SHAP breakdown
            try:
                display_row_shap(
                    full_pipeline, metadata, example_features, historical_data
                )
            except Exception:
                pass
            logging.info(f"{'='*50}\n")

    except Exception as e:
        logging.error(f"An error occurred during the prediction process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)


def _normalize_aliases2(df: pd.DataFrame, expected: set) -> pd.DataFrame:
    """Map common alias columns (legacy/new) to canonical training names.

    This is a superset shim to preserve older behavior while enabling
    new CSV schema ingestion.
    """
    out = df.copy()
    alias_pairs = [
        ("Land Size (sqm)", "Land Size (m�)"),
        ("Floor Size (sqm)", "Floor Size (m�)"),
        ("YearBuilt", "Year Built"),
    ]
    new_to_old = {
        # Construction/position synonyms into the training names (lowercase)
        "wall construction": "wall_construction",
        "roof construction": "roof_construction",
        "site position": "position",
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
        "land_size": "Land Size (sqm)",
        "land_size (m²)": "Land Size (sqm)",
        "land_size (m2)": "Land Size (sqm)",
        "floor_size": "Floor Size (sqm)",
        "floor_size (m²)": "Floor Size (sqm)",
        "floor_size (m2)": "Floor Size (sqm)",
        "year_built": "Year Built",
        "property_type": "Property Type",
        "days_on_market": "Days on Market",
        "capital_value": "Capital Value",
        "land_value": "Land Value",
        "improvement_value": "Improvement Value",
        "valuation": "Valuation Date",
        "valuation_date": "Valuation Date",
        "furnishingsF": "Furnishings",
        "furnishings": "Furnishings",
        "pets": "Pets",
        "garage": "garage",
    }
    for a, b in alias_pairs:
        if a in expected and a not in out.columns and b in out.columns:
            out[a] = out[b]
        if b in expected and b not in out.columns and a in out.columns:
            out[b] = out[a]
    lower_cols = {c.lower(): c for c in out.columns}
    for new, old in new_to_old.items():
        if old in expected and old not in out.columns and new in lower_cols:
            out[old] = out[lower_cols[new]]
    # Carparks normalization if needed
    if "Car" in expected and "Car" not in out.columns:
        car = None
        for key in ("carparks", "car_parks_total", "carparks_total", "car"):
            if key in lower_cols:
                car = pd.to_numeric(out[lower_cols[key]], errors="coerce")
                break
        g = (
            pd.to_numeric(out[lower_cols["garage_parks"]], errors="coerce")
            if "garage_parks" in lower_cols
            else 0
        )
        o = (
            pd.to_numeric(out[lower_cols["offstreet_parks"]], errors="coerce")
            if "offstreet_parks" in lower_cols
            else 0
        )
        comp = (g + o) if (hasattr(g, "__len__") or hasattr(o, "__len__")) else 0
        if car is not None:
            out["Car"] = np.fmax(
                pd.to_numeric(car, errors="coerce").fillna(0),
                comp if hasattr(comp, "__len__") else 0,
            )
        elif hasattr(comp, "__len__"):
            out["Car"] = comp
    # Ensure compatibility for snake_case size columns
    if "Land Size (sqm)" in out.columns:
        out["land_size"] = out["Land Size (sqm)"]
    elif "land_size" in out.columns and "Land Size (sqm)" in expected:
        out["Land Size (sqm)"] = out["land_size"]
    if "Floor Size (sqm)" in out.columns:
        out["floor_size"] = out["Floor Size (sqm)"]
    elif "floor_size" in out.columns and "Floor Size (sqm)" in expected:
        out["Floor Size (sqm)"] = out["floor_size"]
    # Canonicalize Furnishings/Pets/Garage values
    if "Furnishings" in out.columns:
        out["Furnishings"] = out["Furnishings"].apply(_canonicalize_furnishings)
    if "furnishings" in out.columns:
        out["furnishings"] = out["furnishings"].apply(_canonicalize_furnishings)
        if "Furnishings" not in out.columns:
            out["Furnishings"] = out["furnishings"]
    if "Pets" in out.columns:
        out["Pets"] = out["Pets"].apply(_canonicalize_pets)
    if "pets" in out.columns:
        out["pets"] = out["pets"].apply(_canonicalize_pets)
        if "Pets" not in out.columns:
            out["Pets"] = out["pets"]
    if "Garage" in out.columns:
        out["Garage"] = out["Garage"].apply(_canonicalize_garage)
    if "garage" in out.columns:
        out["garage"] = out["garage"].apply(_canonicalize_garage)
        if "Garage" not in out.columns:
            out["Garage"] = out["garage"]

    # Rename common camel-case headers to model column names
    rename_candidates = {
        "Agency": "agency",
        "Development Zone": "development_zone",
        "Property Type": "property_type2",
    }
    for src, dest in rename_candidates.items():
        if dest not in out.columns and src in out.columns:
            out[dest] = out[src]

    # Promote snake_case counterparts when only camel-case present
    if "property_type2" not in out.columns and "property_type" in out.columns:
        out["property_type2"] = out["property_type"]

    # Normalise casing for categorical features to match training vocabulary
    uppercase_targets = [
        "agency",
        "category",
        "development_zone",
        "property_type2",
        "position",
        "roof_construction",
        "wall_construction",
    ]
    for col in uppercase_targets:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .fillna("UNKNOWN")
                .str.strip()
                .str.upper()
            )

    # When CLI omits furnis evidence, keep explicit UNKNOWN for downstream parity
    if "furnishings_evidence" in out.columns:
        out["furnishings_evidence"] = (
            out["furnishings_evidence"]
            .astype("string")
            .fillna("UNKNOWN")
            .str.strip()
        )

    if "Postcode" in out.columns:
        out["Postcode"] = (
            pd.to_numeric(out["Postcode"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .str.zfill(4)
        )
    return out


if __name__ == "__main__":
    main()



def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile (q in [0,1]). Falls back to unweighted if bad input."""
    try:
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not m.any():
            return float(np.nanquantile(v, q))
        v, w = v[m], w[m]
        idx = np.argsort(v)
        v = v[idx]
        w = w[idx]
        cw = np.cumsum(w)
        cw = cw / cw[-1]
        return float(v[np.searchsorted(cw, q, side="right")])
    except Exception:
        try:
            return float(np.nanquantile(values, q))
        except Exception:
            return float("nan")
