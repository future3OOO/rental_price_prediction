#!/usr/bin/env python3
"""outlier_removal.py • v3.0

Robust, vectorised outlier pruning for residential rentals.
This version replaces the legacy loop-based implementation with a
configuration-driven, fully vectorised routine inspired by best practices
seen in Kaggle competitions.

What's new (v3.0)
─────────────────
✔ True MAD Mahalanobis distance (diagonal)  
✔ Optional robust MinCovDet and Isolation-Forest ensemble  
✔ Auto numeric-column detection (unless explicit list given)  
✔ Bedroom-specific z-threshold override via `main_z_per_bed`  
✔ Helper `ORConfig.to_dict()` for Optuna / JSON  
✔ Artefacts unchanged but include mask; external API still returns only `clean_df` for backward-compatibility.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest

__all__ = [
    "ORConfig",
    "get_outlier_mask",
    "save_outlier_report",
    "sophisticated_outlier_removal",
]


@dataclass(slots=True)
class ORConfig:
    """Hyper-parameters for outlier removal.

    Adjust these to run sweeps (e.g. with Optuna) without editing code.
    """

    main_z: float = 5.0              # |z| ≥ main_z  → candidate
    bed4_low_z: float = -1.0         # bed ≥ 4 & z ≤ bed4_low_z → candidate
    support_z: float = 2.0           # |z| ≥ support_z counts as supporting
    min_support: int = 2             # votes needed from supporting features
    mahal_p: float = 0.999           # χ² quantile for Mahalanobis fence
    iqr_k: float = 3.0               # IQR multiplier for rent fences
    luxury_iqr_k: float = 3.0        # capital value ≥ Q3 + k·IQR → luxury skip

    # bedroom-specific main_z overrides e.g. {1:4, 2:4.5, 3:5}
    main_z_per_bed: Dict[int, float] | None = None

    # ensemble toggles
    use_mcd: bool = True
    use_isoforest: bool = True

    # IsolationForest hyper-parameters
    iso_max_samples: str | int = "auto"
    iso_random_state: int = 42

    # runtime options
    log_dir: str | Path = "output"
    artefact_name: str = "outliers.parquet"

    # helper
    def to_dict(self):
        return asdict(self)


# ───────────────────────── helpers ────────────────────────── #

def _modified_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if pd.isna(mad) or mad == 0:
        mad = 1e-9
    return 0.6745 * (series - med) / mad


def _iqr_bounds(series: pd.Series, k: float) -> Tuple[float, float]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1 or 1e-9
    return q1 - k * iqr, q3 + k * iqr


def _diag_mahalanobis(df: pd.DataFrame) -> pd.Series:
    """Fast diagonal Mahalanobis distance using *true* MAD (median abs-dev)."""
    if df.empty:
        return pd.Series(0.0, index=df.index)

    med = df.median()
    mad = df.apply(lambda s: (s - s.median()).abs().median()).replace(0, 1e-9)
    X = (df - med) / mad
    return (X**2).sum(axis=1)


# ─────────────────────── main routine ─────────────────────── #

def get_outlier_mask(
    df: pd.DataFrame,
    cfg: ORConfig,
    *,
    bed_col: str,
    rent_col: str,
    capval_col: str,
    numeric_cols: Iterable[str] | None,
) -> pd.Series:
    """Return boolean mask (True ⇢ outlier) the same length as *df*."""

    log = logging.getLogger(__name__)

    # -------- sanity checks & defaults -------- #
    if numeric_cols is None:
        num_cols: List[str] = list(
            df.select_dtypes("number")
            .columns.difference({rent_col, capval_col, bed_col})
        )
    else:
        num_cols = list(dict.fromkeys(numeric_cols))

    # -- verify mandatory columns exist --
    for col in (rent_col, capval_col, bed_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in input data")

    # Keep only numeric columns that are actually present in data. Warn once for
    # any missing so users know why certain z-score columns are absent.
    existing_num_cols = [c for c in num_cols if c in df.columns]
    missing_num_cols = sorted(set(num_cols) - set(existing_num_cols))
    if missing_num_cols:
        log.warning("Numeric columns absent in dataset will be skipped: %s", ", ".join(missing_num_cols))
    num_cols = existing_num_cols  # may be empty ⇒ Mahalanobis step skipped

    df = df.copy()

    # ensure numeric types
    df[rent_col] = pd.to_numeric(df[rent_col], errors="coerce")
    df[capval_col] = pd.to_numeric(df[capval_col], errors="coerce")

    # log-scale heavy-tailed targets
    df["_log_rent"] = np.log1p(df[rent_col])
    df["_log_cap"] = np.log1p(df[capval_col])

    # -------- group-wise robust z-scores -------- #
    grp = df.groupby(bed_col, group_keys=False)
    z_rent = grp["_log_rent"].transform(_modified_z)
    z_capv = grp["_log_cap"].transform(_modified_z)
    z_support = grp[num_cols].transform(_modified_z) if num_cols else pd.DataFrame(index=df.index)

    # -------- IQR fences (rent) & luxury skip (cap value) -------- #
    _ , cap_hi_fence = grp[capval_col].transform(lambda s: _iqr_bounds(s, cfg.iqr_k)[0]), grp[capval_col].transform(lambda s: _iqr_bounds(s, cfg.luxury_iqr_k)[1])

    # -------- Mahalanobis leverage in high-D -------- #
    if num_cols:
        maha_d2 = _diag_mahalanobis(df[num_cols])
        maha_thresh = chi2.ppf(cfg.mahal_p, len(num_cols))
        maha_flag = maha_d2 > maha_thresh
    else:
        maha_d2 = pd.Series(0.0, index=df.index)
        maha_flag = pd.Series(False, index=df.index)

    # -------- candidate rules -------- #
    per_bed_thr = df[bed_col].map(cfg.main_z_per_bed or {})
    thr = per_bed_thr.fillna(cfg.main_z)
    main_out = (z_rent.abs() >= thr) | (z_capv.abs() >= thr)
    bed4_low = (df[bed_col] >= 4) & (z_rent <= cfg.bed4_low_z)
    candidate = main_out | bed4_low | maha_flag

    # supporting feature vote
    support_votes = (z_support.abs() >= cfg.support_z).sum(axis=1)
    has_support = support_votes >= cfg.min_support

    # skip luxury high-value properties
    luxury_skip = df[capval_col] > cap_hi_fence

    remove_mask = candidate & has_support & ~luxury_skip

    # Isolation-Forest ensemble veto
    if cfg.use_isoforest and num_cols:
        try:
            X_iso = df[num_cols].fillna(df[num_cols].median())
            iso = IsolationForest(
                contamination="auto",
                max_samples=cfg.iso_max_samples,
                random_state=cfg.iso_random_state,
            ).fit(X_iso)
            iso_out = iso.predict(X_iso) == -1
            remove_mask &= iso_out
        except Exception as err:  # noqa: BLE001
            log.warning("IsolationForest failed (%s) – skipped.", err)

    return remove_mask


def save_outlier_report(
    df: pd.DataFrame,
    mask: pd.Series,
    cfg: ORConfig,
    *,
    bed_col: str,
    z_rent: pd.Series,
    z_capv: pd.Series,
    votes: pd.Series,
    maha: pd.Series,
    reason: pd.Series,
    per_feat_z: dict,
) -> None:
    """Write artefacts (parquet + CSV) to cfg.log_dir."""

    out_dir = Path(cfg.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reason_series = pd.Series(reason, index=df.index, dtype="object")
    per_feat_series = {k: pd.Series(v, index=df.index) for k, v in per_feat_z.items()}
    artefacts = (
        df[mask]
        .assign(
            z_rent=pd.Series(z_rent, index=df.index)[mask],
            z_cap=pd.Series(z_capv, index=df.index)[mask],
            votes=pd.Series(votes, index=df.index)[mask],
            maha=pd.Series(maha, index=df.index)[mask],
            reason=reason_series[mask],
            **{k: series[mask] for k, series in per_feat_series.items()},
        )
        .copy()
    )

    try:
        artefacts.to_parquet(out_dir / cfg.artefact_name, index=False, engine="pyarrow")
    except ImportError:
        logging.getLogger(__name__).warning("pyarrow missing; skipped parquet export for outlier report")
    artefacts.to_csv(out_dir / "removed_outliers.csv", index=False)
    df[~mask].to_csv(out_dir / "cleaned_data.csv", index=False)
    logging.getLogger(__name__).info(
        "Removed %d outliers • artefact → %s", len(artefacts), cfg.artefact_name
    )


# ───────── wrapper – keeps original API ───────── #

def sophisticated_outlier_removal(
    data: pd.DataFrame,
    *,
    config: ORConfig | None = None,
    bed_col: str = "Bed",
    rent_col: str = "Last Rental Price",
    capval_col: str = "Capital Value",
    numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Drop outliers and return cleaned DataFrame (signature unchanged)."""

    cfg = config or ORConfig()

    # build mask
    remove_mask = get_outlier_mask(
        data,
        cfg,
        bed_col=bed_col,
        rent_col=rent_col,
        capval_col=capval_col,
        numeric_cols=numeric_cols,
    )

    # helper z-scores for report
    df = data.copy()
    df["_log_rent"] = np.log1p(df[rent_col])
    df["_log_cap"] = np.log1p(df[capval_col])
    grp = df.groupby(bed_col, group_keys=False)
    z_rent = grp["_log_rent"].transform(_modified_z)
    z_capv = grp["_log_cap"].transform(_modified_z)

    # -------- numeric feature selection -------- #
    if numeric_cols is None:
        num_cols: List[str] = list(
            df.select_dtypes("number")
            .columns.difference({rent_col, capval_col, bed_col})
        )
    else:
        num_cols = list(dict.fromkeys(numeric_cols))

    z_support = (
        grp[num_cols].transform(_modified_z) if num_cols else pd.DataFrame(index=df.index)
    )
    votes = (z_support.abs() >= cfg.support_z).sum(axis=1)

    # fast diagonal distance first
    maha_d2 = _diag_mahalanobis(df[num_cols])
    maha_thresh = chi2.ppf(cfg.mahal_p, len(num_cols))
    maha_flag = maha_d2 > maha_thresh

    # optional full robust covariance (MinCovDet)
    if cfg.use_mcd and len(df) > len(num_cols) + 1:
        try:
            X_fit = df[num_cols].fillna(df[num_cols].median())
            mcd = MinCovDet().fit(X_fit)
            # build per-column Series for NaN imputation to avoid numpy-array error
            meds = pd.Series(mcd.location_, index=num_cols)
            maha_mcd = mcd.mahalanobis(df[num_cols].fillna(meds))
            maha_flag |= maha_mcd > chi2.ppf(cfg.mahal_p, len(num_cols))
        except Exception as err:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "MinCovDet failed (%s) – falling back to diagonal.", err
            )

    # recompute Mahalanobis flag (diag) just for reason label
    if num_cols:
        maha_d2 = _diag_mahalanobis(df[num_cols])
        maha_thresh = chi2.ppf(cfg.mahal_p, len(num_cols))
        maha_flag = maha_d2 > maha_thresh
    else:
        maha_flag = pd.Series(False, index=df.index)

    per_bed_thr = df[bed_col].map(cfg.main_z_per_bed or {}).fillna(cfg.main_z)
    main_out = (z_rent.abs() >= per_bed_thr) | (z_capv.abs() >= per_bed_thr)
    bed4_low = (df[bed_col] >= 4) & (z_rent <= cfg.bed4_low_z)
    reason = np.select(
        [maha_flag, main_out, bed4_low],
        ["mahalanobis", "main_z", "bed4_low"],
        default="mixed",
    )
    maha = maha_d2

    # write artefacts
    save_outlier_report(
        df,
        remove_mask,
        cfg,
        bed_col=bed_col,
        z_rent=z_rent,
        z_capv=z_capv,
        votes=votes,
        maha=maha,
        reason=reason,
        per_feat_z={f"z_{c}": z_support[c] for c in num_cols} if num_cols else {},
    )

    return df[~remove_mask].reset_index(drop=True)


# ───────────── CLI smoke-test ───────────── #
if __name__ == "__main__":  # pragma: no cover
    import argparse, json, sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="Outlier removal quick-test")
    ap.add_argument("csv")
    ap.add_argument("--out", default="output")
    ns = ap.parse_args()

    df_in = pd.read_csv(ns.csv, low_memory=False)
    clean_df = sophisticated_outlier_removal(df_in, config=ORConfig(log_dir=ns.out))
    print(json.dumps({"kept_rows": int(len(clean_df))}, indent=2))
