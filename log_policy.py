"""Log feature policy shared between training and prediction."""
from __future__ import annotations

import os
import pandas as pd

# Keep a conservative default: exclude size logs unless explicitly allowed via env
# Tests expect log_Floor Size (sqm) to be dropped under defaults.
_DEFAULT_ALLOWED = "Capital Value,Land Value,Improvement Value,Days on Market"
ALLOWED_LOG_FEATURES = tuple(
    s.strip()
    for s in os.getenv("ALLOWED_LOG_FEATURES", _DEFAULT_ALLOWED).split(",")
    if s.strip()
)
ALLOWED_LOG_FEATURES_SET = set(ALLOWED_LOG_FEATURES)


def _enforce_log_feature_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop log_* columns that are not explicitly whitelisted."""
    if df.empty:
        return df
    drop_cols: list[str] = []
    for col in df.columns:
        name = str(col)
        if not name.startswith("log_"):
            continue
        base = name[4:].strip()
        if base not in ALLOWED_LOG_FEATURES_SET:
            drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def apply_log_policy(df: pd.DataFrame, *, drop_raw: bool = False) -> pd.DataFrame:
    """Apply whitelist and optionally drop raw counterparts of allowed logs."""
    df = _enforce_log_feature_policy(df)
    if drop_raw:
        for base in ALLOWED_LOG_FEATURES:
            raw = str(base)
            log_name = f"log_{raw}"
            if raw in df.columns and log_name in df.columns:
                df = df.drop(columns=[raw])
    return df


__all__ = [
    "ALLOWED_LOG_FEATURES",
    "ALLOWED_LOG_FEATURES_SET",
    "_enforce_log_feature_policy",
    "apply_log_policy",
]
