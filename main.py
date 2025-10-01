#!/usr/bin/env python3
"""
main.py — end‑to‑end rental‑price pipeline controller  • 2025‑06
----------------------------------------------------------------
• Ensures required packages
• Runs data‑clean → EDA → model training (v6.2) → reporting
• Compatible with sparse output & LogTargetWrapper
"""

from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
import sys
import traceback
import platform
import inspect
import os
from pathlib import Path
from typing import List
from functools import wraps

import numpy as np
import pandas as pd

# ────────────────────────── CONFIG ────────────────────────── #
# ──────── Project paths with config.py fallback ──────── #
ROOT = Path(__file__).resolve().parent

try:
    import config as _cfg  # type: ignore

    RAW_CSV = Path(_cfg.RAW_DATA_PATH)
    ARTIFACT_DIR = Path(getattr(_cfg, "ARTIFACT_DIR", ROOT / "artifacts"))
except Exception:  # noqa: BLE001  config missing or invalid
    RAW_CSV = ROOT / "data" / "raw_rentals.csv"
    ARTIFACT_DIR = ROOT / "artifacts"

# other dirs
PLOT_DIR, ART_DIR = ROOT / "plots", ROOT / "models"
for _d in (PLOT_DIR, ART_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Force headless plotting to avoid Tkinter backend crashes in non-main threads
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:  # pragma: no cover
    import matplotlib  # type: ignore

    if str(getattr(matplotlib, "get_backend", lambda: "")()).lower().endswith("agg") is False:
        matplotlib.use("Agg", force=True)
except Exception:
    pass

REQUIRED_PKGS: List[str] = [
    "optuna",
    "catboost",
    "pyarrow",  # parquet engine for outlier artefacts
    "joblib",
    "shap",
    "category_encoders",
    "scipy",        # sparse stack
    "seaborn",
]

# optional local utility modules (no hard fail if absent)
OPTIONAL_STEPS = {
    "data_cleaning": "data_cleaning.data_cleaning",
    # Pre-split outlier removal is disabled by default to prevent leakage.
    # Enable explicitly via --pre-split-outliers or PRE_SPLIT_OUTLIERS=1.
    "outlier_removal": "outlier_removal.sophisticated_outlier_removal",
    "eda": "eda.perform_eda",
}

# ───────────────────────── HELPERS ────────────────────────── #


class RemovedRowsTracker:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self._records: list[pd.DataFrame] = []

    def track(self, before_df: pd.DataFrame, after_df: pd.DataFrame, step: str) -> None:
        if before_df is None or "__row_id" not in before_df.columns:
            return
        before_ids = set(before_df["__row_id"].astype(int, copy=False))
        if after_df is None or "__row_id" not in after_df.columns:
            after_ids = before_ids
        else:
            after_ids = set(after_df["__row_id"].astype(int, copy=False))
        removed_ids = [rid for rid in before_ids if rid not in after_ids]
        self.track_ids(removed_ids, before_df, step)

    def track_ids(self, removed_ids: List[int], reference_df: pd.DataFrame, step: str) -> None:
        if not removed_ids or reference_df is None or "__row_id" not in reference_df.columns:
            return
        removed = reference_df[reference_df["__row_id"].isin(removed_ids)].copy()
        if removed.empty:
            return
        removed["__removed_step"] = step
        self._records.append(removed)

    def flush(self) -> None:
        if not self._records:
            return
        out_df = pd.concat(self._records, ignore_index=True)
        ordered_cols = ["__removed_step"] + [c for c in out_df.columns if c not in {"__removed_step"}]
        out_df = out_df[ordered_cols]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.output_path, index=False)
def ensure_packages(pkgs: List[str]) -> None:
    for pkg in pkgs:
        if importlib.util.find_spec(pkg) is None:
            logging.warning("Installing %s …", pkg)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def patch_onehotencoder_sparse_kwarg() -> None:
    """Monkey‑patch sklearn <1.4 to accept `sparse=` kwargs when `sparse_output` exists."""
    from sklearn.preprocessing import OneHotEncoder  # type: ignore

    # new sklearn (>=1.4) removed 'sparse' so no patch needed
    if 'sparse' not in inspect.signature(OneHotEncoder.__init__).parameters:
        return

    if hasattr(OneHotEncoder, "__patched"):
        return

    original_init = OneHotEncoder.__init__

    @wraps(original_init)
    def _patched_init(self, *args, **kwargs):  # type: ignore
        # Accept both 'sparse' and 'sparse_output' keys; map as needed
        if "sparse" in kwargs and "sparse_output" not in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        original_init(self, *args, **kwargs)

    OneHotEncoder.__init__ = _patched_init  # type: ignore
    OneHotEncoder.__patched = True  # type: ignore


def run_optional(step_key: str, df: pd.DataFrame) -> pd.DataFrame:
    path = OPTIONAL_STEPS[step_key]
    try:
        mod_path, func_name = path.rsplit(".", 1)
        module = importlib.import_module(mod_path)
        func = getattr(module, func_name)
        logging.info("Running optional step: %s", step_key)

        # Some optional steps (e.g., perform_eda) expect extra args; inspect signature.
        sig = inspect.signature(func)
        if len(sig.parameters) == 2:
            return func(df, PLOT_DIR)  # assume (df, plot_dir)
        return func(df)
    except ModuleNotFoundError:
        logging.warning("Skipping %s – module not found", step_key)
        return df
    except Exception:  # noqa: BLE001
        logging.error("Step %s failed but pipeline will proceed:\n%s", step_key, traceback.format_exc())
        # If cleaning fails, ensure at least the core target/date coercions for robustness
        if step_key == "data_cleaning":
            if "Last Rental Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Last Rental Date"]):
                df = df.copy()
                df["Last Rental Date"] = pd.to_datetime(df["Last Rental Date"], errors="coerce")
                df = df.dropna(subset=["Last Rental Date"]).reset_index(drop=True)
            tgt_candidates = ["Rent", "Last Rental Price"]
            for _t in tgt_candidates:
                if _t in df.columns and not pd.api.types.is_numeric_dtype(df[_t]):
                    df[_t] = pd.to_numeric(df[_t], errors="coerce")
                    df = df.dropna(subset=[_t]).reset_index(drop=True)
        return df


# ────────────────────────── MAIN FLOW ───────────────────────── #
def main(
    *,
    skip_train: bool,
    loglevel: str,
    use_gpu: bool,
    trials: int | None,
    timeout_sec: int | None,
    no_scale: bool,
    train_frac: float,
    valid_frac: float,
    max_ohe_card: int,
    explain: bool,
    block_cv: bool,
    skip_featurealgo: bool,
    pre_split_outliers: bool,
    no_train_outlier_removal: bool,
    final_fit: bool,
    # NEW: suburb feature toggles
    no_suburb_features: bool = False,
    use_suburb_365d: bool = False,
    use_suburb_medians: bool = False,
    use_suburb_rank: bool = False,
    no_suburb_loo: bool = False,
    # NEW: CV/holdout control
    walk_cv: bool = False,
    cv_last_k_folds: int | None = None,
    gap_months: int = 1,
    test_holdout_months: int = 12,
    early_holdout_months: int = 2,
    search_halflife: bool = True,
) -> None:
    ensure_packages(REQUIRED_PKGS)
    patch_onehotencoder_sparse_kwarg()

    # Use UTF-8 console encoding on Windows to prevent UnicodeEncodeError for symbols.
    console_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.FileHandler("main.log", encoding="utf-8")

    logging.basicConfig(
        level=getattr(logging, loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[file_handler, console_handler],
    )

    if platform.system() == "Windows":
        try:
            import ctypes  # pragma: no cover

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass  # best-effort

    logging.info("🚚 Loading raw data from %s", RAW_CSV)
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df["__row_id"] = np.arange(len(df), dtype=int)
    removed_tracker = RemovedRowsTracker(PLOT_DIR / "removed_properties.csv")

    # set env var for downstream modules if skipping featurealgo
    if skip_featurealgo:
        os.environ["SKIP_FEATUREALGO"] = "1"

    # ---------- optional preprocessing / EDA ----------
    before_clean = df.copy()
    df = run_optional("data_cleaning", df)
    removed_tracker.track(before_clean, df, "data_cleaning")
    # Guard pre-split outlier removal behind an explicit flag to avoid leakage.
    if pre_split_outliers or os.getenv("PRE_SPLIT_OUTLIERS", "0").lower() in {"1", "true", "yes"}:
        before_outlier = df.copy()
        df = run_optional("outlier_removal", df)
        removed_tracker.track(before_outlier, df, "outlier_removal")
        # Save a convenience copy under data/interim without overwriting the canonical cleaned file
        try:
            try:
                import config as _cfg2  # type: ignore
                interim_dir = Path(getattr(_cfg2, "INTERIM_DATA_DIR", ROOT / "data" / "interim"))
            except Exception:
                interim_dir = ROOT / "data" / "interim"
            interim_dir.mkdir(parents=True, exist_ok=True)
            out_no_outliers = interim_dir / "cleaned_no_outliers.csv"
            df.to_csv(out_no_outliers, index=False)
            logging.info("Pre-split outlier-removed dataset saved => %s", out_no_outliers)
        except Exception as _save_exc:  # noqa: BLE001
            logging.warning("Could not persist cleaned_no_outliers CSV: %s", _save_exc)
    run_optional("eda", df)

    if skip_train:
        logging.info("--skip-train flag set; stopping before model training.")
        return

    # ---------- model training ----------
    # Geo-POI wiring: allow CLI/env and sensible auto-detect of artifacts/poi_christchurch.csv
    # Set env before importing model_training so it picks up configuration via _geo_env_config.
    try:
        # Prefer config.py values; fall back to auto-detect under artifacts/
        cfg_poi = None
        try:
            import config as _cfg_geo  # type: ignore
            cfg_poi = getattr(_cfg_geo, "GEO_POI_CSV", None)
            cfg_lat = getattr(_cfg_geo, "GEO_LAT_COL", None)
            cfg_lon = getattr(_cfg_geo, "GEO_LON_COL", None)
            cfg_radii = getattr(_cfg_geo, "GEO_RADII_KM", None)
            cfg_decay = getattr(_cfg_geo, "GEO_DECAY_KM", None)
            cfg_maxdecay = getattr(_cfg_geo, "GEO_MAX_DECAY_KM", None)
            cfg_cats = getattr(_cfg_geo, "GEO_CATEGORIES", None)
        except Exception:
            cfg_lat = cfg_lon = cfg_radii = cfg_decay = cfg_maxdecay = cfg_cats = None

        if not os.getenv("GEO_POI_CSV"):
            default_poi = ARTIFACT_DIR / "poi_christchurch.csv"
            chosen = cfg_poi if (cfg_poi and Path(cfg_poi).exists()) else (str(default_poi) if default_poi.exists() else None)
            if chosen:
                os.environ["GEO_POI_CSV"] = str(chosen)
                logging.info("Geo-POI enabled via config/default: %s", chosen)
        # Coordinate and geometry defaults (only if not provided via CLI/ENV)
        if cfg_lat and not os.getenv("LAT_COL"):
            os.environ["LAT_COL"] = str(cfg_lat)
        if cfg_lon and not os.getenv("LON_COL"):
            os.environ["LON_COL"] = str(cfg_lon)
        if cfg_radii and not os.getenv("GEO_RADII_KM"):
            try:
                os.environ["GEO_RADII_KM"] = ",".join(str(float(r)) for r in cfg_radii)
            except Exception:
                pass
        if cfg_decay is not None and not os.getenv("GEO_DECAY_KM"):
            os.environ["GEO_DECAY_KM"] = str(cfg_decay)
        if cfg_maxdecay is not None and not os.getenv("GEO_MAX_DECAY_KM"):
            os.environ["GEO_MAX_DECAY_KM"] = str(cfg_maxdecay)
        if cfg_cats and not os.getenv("GEO_CATEGORIES"):
            try:
                os.environ["GEO_CATEGORIES"] = ",".join(cfg_cats)
            except Exception:
                pass
    except Exception:
        # Best-effort only; training continues without POIs if anything fails here.
        pass
    # Expose suburb feature toggles to training via ENV BEFORE importing training module
    if no_suburb_features:
        os.environ["USE_SUBURB_FEATURES"] = "0"
    if use_suburb_365d:
        os.environ["USE_SUBURB_365D_MEDIAN"] = "1"
    # New fine-grained toggles (default: LOO on, medians off, rank off)
    if use_suburb_medians:
        os.environ["USE_SUBURB_MEDIANS"] = "1"
    if use_suburb_rank:
        os.environ["USE_SUBURB_RANK"] = "1"
    if no_suburb_loo:
        os.environ["USE_SUBURB_LOO"] = "0"
    if walk_cv:
        os.environ["WALK_FORWARD_CV"] = "1"
    if cv_last_k_folds:
        os.environ["CV_LAST_K_FOLDS"] = str(cv_last_k_folds)
    os.environ["GAP_MONTHS"] = str(gap_months)
    os.environ["TEST_HOLDOUT_MONTHS"] = str(test_holdout_months)
    os.environ["EARLY_HOLDOUT_MONTHS"] = str(early_holdout_months)
    os.environ["SEARCH_HALFLIFE"] = "1" if search_halflife else "0"

    # Import after env is set so model_training reads correct flags at import time
    from model_training import train_model, DEFAULT_TRIALS, DEFAULT_TIMEOUT  # type: ignore

    # Dynamically determine column names with alias support (new + legacy)
    def _pick(df_cols, cands):
        s = set(df_cols)
        for c in cands:
            if c in s:
                return c
            for d in s:
                if str(d).lower() == str(c).lower():
                    return d
        return None
    date_col = _pick(df.columns, ["last_rent_date", "last_rental_date", "date", "Last Rental Date", "Date"]) 
    target_col = _pick(df.columns, ["last_rent_price", "last_rental_price", "weekly_rent", "rent", "Last Rental Price", "Rent"]) 
    suburb_col = _pick(df.columns, ["suburb", "locality", "Suburb", "Suburb Name"]) 

    if not date_col or not target_col or not suburb_col:
        missing = [n for n, v in {"date_col": date_col, "target_col": target_col, "suburb_col": suburb_col}.items() if v is None]
        raise ValueError(f"Required columns missing from dataset: {', '.join(missing)}")

    # Walk-forward CV env toggles
    if os.getenv("WALK_FORWARD_CV") is None:
        # Allow CLI to control; default remains off unless flag set
        os.environ["WALK_FORWARD_CV"] = "1" if args.pre_split_outliers and False else "0"  # no-op unless overridden below

    artefacts = train_model(
        df_raw=df,
        target_col=target_col,
        date_col=date_col,
        suburb_col=suburb_col,
        train_frac=train_frac,
        valid_frac=valid_frac,
        trials=trials or DEFAULT_TRIALS,
        max_ohe_card=max_ohe_card,
        use_gpu=use_gpu,
        scale_numeric=not no_scale,
        explain=explain,
        block_cv=block_cv,
        timeout_sec=timeout_sec or DEFAULT_TIMEOUT,
        skip_featurealgo=skip_featurealgo,
        final_fit_on_all=final_fit,
        removed_tracker=removed_tracker,
    )
    removed_tracker.flush()
    if "__row_id" in df.columns:
        df.drop(columns=["__row_id"], inplace=True, errors="ignore")
    logging.info("🏁 Training finished. Artefacts: %s", artefacts)


# ────────────────────────── CLI ────────────────────────── #
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="Rental‑price end‑to‑end pipeline")
    cli_parser.add_argument("--skip-train", action="store_true", help="Stop after EDA")
    cli_parser.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING | ERROR")
    cli_parser.add_argument("--gpu", action="store_true", help="Enable GPU training if CUDA device is available")
    cli_parser.add_argument("--trials", type=int, help="Override number of Optuna trials")
    cli_parser.add_argument("--timeout-sec", type=int, help="Optuna study timeout seconds (default 600)")
    cli_parser.add_argument("--no-scale", action="store_true", help="Disable StandardScaler for numeric features")
    cli_parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction (default 0.7)")
    cli_parser.add_argument("--valid-frac", type=float, default=0.1, help="Validation fraction (default 0.1)")
    cli_parser.add_argument("--max-ohe-card", type=int, default=15, help="Max cardinality for one-hot encoding")
    # Legacy tuning flags (no longer used in v7.x)
    cli_parser.add_argument("--explain", action="store_true", help="Save SHAP summary plot")
    cli_parser.add_argument("--block-cv", action="store_true", help="Use monthly block-aware GroupShuffleSplit for CV")
    cli_parser.add_argument("--skip-featurealgo", action="store_true", help="Bypass transformers in the featurealgo package")
    cli_parser.add_argument("--final-fit", action="store_true", help="Refit final model on train+early with best params/iteration")
    # Suburb feature toggles
    cli_parser.add_argument("--no-suburb-features", action="store_true", help="Disable all suburb-derived features (LOO, medians)")
    cli_parser.add_argument("--use-suburb-365d", action="store_true", help="Enable 12-month suburb median feature (off by default)")
    cli_parser.add_argument("--use-suburb-medians", action="store_true", help="Enable suburb rolling medians (90d, 365d, bed-conditional)")
    cli_parser.add_argument("--use-suburb-rank", action="store_true", help="Enable Rank30dX12m feature (momentum-like ratio)")
    cli_parser.add_argument("--no-suburb-loo", action="store_true", help="Disable OOF LOO encoders for suburb/suburb×month")
    cli_parser.add_argument("--walk-cv", action="store_true", help="Use walk-forward CV (no future leakage) instead of Monte-Carlo block CV")
    cli_parser.add_argument("--cv-last-k-folds", type=int, help="Only use the last K folds before EARLY/TEST for Optuna CV to reduce CV↔TEST drift (default 5)")
    cli_parser.add_argument("--gap-months", type=int, default=0, help="Months gap between training and validation blocks (walk-forward CV)")
    cli_parser.add_argument("--test-holdout-months", type=int, default=12, help="Final TEST size in months")
    cli_parser.add_argument("--early-holdout-months", type=int, default=2, help="EARLY size in months")
    cli_parser.add_argument("--no-search-halflife", action="store_true", help="Disable halflife search in Optuna")
    # Geo-POI CLI (train=serve parity via env consumed by model_training and prediction)
    cli_parser.add_argument("--poi-csv", help="Path to POI CSV (enables geo features)")
    cli_parser.add_argument("--lat-col", default=None, help="Latitude column name in training data (default 'Latitude')")
    cli_parser.add_argument("--lon-col", default=None, help="Longitude column name in training data (default 'Longitude')")
    cli_parser.add_argument(
        "--geo-radii-km", default=None,
        help="Comma/space separated radii in km for POI counts (e.g. '0.5,1.0,2.0')",
    )
    cli_parser.add_argument("--geo-decay-km", type=float, default=None, help="Decay km for accessibility score (default 1.5)")
    cli_parser.add_argument("--geo-max-decay-km", type=float, default=None, help="Max distance in km for accessibility aggregation (default 3.0)")
    cli_parser.add_argument(
        "--geo-categories", default=None,
        help="Optional POI categories to include (comma-separated, e.g. 'SCHOOL_PRIMARY,UNIVERSITY')",
    )
    cli_parser.add_argument("--pre-split-outliers", action="store_true", help="Run outlier removal before train/val/test split (WARNING: can cause leakage; off by default)")
    cli_parser.add_argument("--no-train-outlier-removal", action="store_true", help="Disable training-only outlier removal stage")
    args = cli_parser.parse_args()
    # Bridge Geo-POI CLI -> ENV before entering main() so model_training picks it up
    try:
        if getattr(args, "poi_csv", None):
            os.environ["GEO_POI_CSV"] = str(args.poi_csv)
        if getattr(args, "lat_col", None):
            os.environ["LAT_COL"] = str(args.lat_col)
        if getattr(args, "lon_col", None):
            os.environ["LON_COL"] = str(args.lon_col)
        if getattr(args, "geo_radii_km", None):
            os.environ["GEO_RADII_KM"] = str(args.geo_radii_km)
        if getattr(args, "geo_decay_km", None) is not None:
            os.environ["GEO_DECAY_KM"] = str(args.geo_decay_km)
        if getattr(args, "geo_max_decay_km", None) is not None:
            os.environ["GEO_MAX_DECAY_KM"] = str(args.geo_max_decay_km)
        if getattr(args, "geo_categories", None):
            os.environ["GEO_CATEGORIES"] = str(args.geo_categories)
    except Exception:
        pass
    main(
        skip_train=args.skip_train,
        loglevel=args.log_level,
        use_gpu=args.gpu,
        trials=args.trials,
        timeout_sec=args.timeout_sec,
        no_scale=args.no_scale,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        max_ohe_card=args.max_ohe_card,
        explain=args.explain,
        block_cv=args.block_cv,
        skip_featurealgo=args.skip_featurealgo,
        pre_split_outliers=args.pre_split_outliers,
        no_train_outlier_removal=args.no_train_outlier_removal,
        final_fit=args.final_fit,
        no_suburb_features=args.no_suburb_features,
        use_suburb_365d=args.use_suburb_365d,
        use_suburb_medians=args.use_suburb_medians,
        use_suburb_rank=args.use_suburb_rank,
        no_suburb_loo=args.no_suburb_loo,
        walk_cv=args.walk_cv,
        cv_last_k_folds=args.cv_last_k_folds,
        gap_months=args.gap_months,
        test_holdout_months=args.test_holdout_months,
        early_holdout_months=args.early_holdout_months,
        search_halflife=not args.no_search_halflife,
    )
