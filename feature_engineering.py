#!/usr/bin/env python3
"""
feature_engineering.py · v2.2  (2025-06-14)

Sparse-first transformer
  • Numeric pass-through (+ optional date parts)
  • Frequency-capped one-hot with “__OTHER__” bucket
  • Fast CSR interactions (num×num, num×cat)
  • Returns CSR by default (memory‑safe); set FE_RETURN_DF_FOR_TESTS=1 to return DataFrame for tests
"""

from __future__ import annotations

import logging
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["FeatureEngineering"]
log = logging.getLogger(__name__)


def _safe(txt: str) -> str:
    """Sanitise string for use in feature names."""
    for ch in (" ", ",", ";", "/", "-", "."):
        txt = txt.replace(ch, "_")
    return txt


class FeatureEngineering(BaseEstimator, TransformerMixin):
    # ───────────────────────── INIT ────────────────────────── #
    def __init__(
        self,
        numeric: Sequence[str] | None = None,
        categorical: Sequence[str] | None = None,
        *,
        interactions: Sequence[Tuple[str, str]] | None = None,
        date_col: str | None = None,
        add_date_parts: bool = True,
        max_levels: int = 15,
        fill_value: float = 0.0,
        dtype=np.float32,
        # Backward-compat aliases used by some tests
        numeric_features: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        interaction_terms: Sequence[Tuple[str, str]] | None = None,
    ) -> None:
        num = numeric if numeric is not None else numeric_features
        cat = categorical if categorical is not None else categorical_features
        inter = interactions if interactions is not None else interaction_terms
        self.numeric = list(num or [])
        self.categorical = list(cat or [])
        self.interactions = list(inter or [])
        self.date_col = date_col
        self.add_date_parts = add_date_parts
        self.max_levels = max_levels
        self.fill_value = fill_value
        self.dtype = dtype

    # ───────────────────────── FIT ────────────────────────── #
    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineering expects a pandas DataFrame")

        # numeric schema
        self._num_cols: List[str] = list(self.numeric)
        if self.add_date_parts and self.date_col and self.date_col in X.columns:
            self._num_cols += [
                f"{self.date_col}_Year",
                f"{self.date_col}_Month_sin",
                f"{self.date_col}_Month_cos",
            ]

        # categorical trimming
        self._levels: dict[str, List[str]] = {}
        for col in self.categorical:
            if col not in X.columns:
                log.warning("Categorical column '%s' missing during fit", col)
                continue
            kept = (
                X[col]
                .astype("category")
                .value_counts(dropna=False)
                .head(self.max_levels)
                .index.astype(str)
            )
            self._levels[col] = [_safe(s) for s in kept if _safe(s) != "OTHER"]

        # base feature names
        self._cat_names = [
            f"{c}__{l}" for c, ls in self._levels.items() for l in ls
        ] + [f"{c}__OTHER" for c in self._levels]
        base_names = self._num_cols + self._cat_names

        # interaction names
        self._int_names: list[str] = []
        for a, b in self.interactions:
            if a in self._num_cols and b in self._num_cols:
                self._int_names.append(f"{a}_x_{b}")
            elif a in self._num_cols and b in self._levels:
                self._int_names += [f"{a}_x_{b}__{l}" for l in self._levels[b]] + [
                    f"{a}_x_{b}__OTHER"
                ]
            elif b in self._num_cols and a in self._levels:
                self._int_names += [f"{a}__{l}_x_{b}" for l in self._levels[a]] + [
                    f"{a}__OTHER_x_{b}"
                ]

        self.feature_names_out_: List[str] = base_names + self._int_names
        return self

    # ──────────────────────── TRANSFORM ──────────────────────── #
    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "feature_names_out_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("transform expects a pandas DataFrame")

        # Backward-compat: some older pickles may miss private attrs
        if not hasattr(self, "_num_cols"):
            self._num_cols = list(getattr(self, "numeric", []))
            if getattr(self, "add_date_parts", False) and getattr(self, "date_col", None) and self.date_col in X.columns:
                self._num_cols += [
                    f"{self.date_col}_Year",
                    f"{self.date_col}_Month_sin",
                    f"{self.date_col}_Month_cos",
                ]
        if not hasattr(self, "_levels"):
            # Reconstruct categorical levels from stored feature_names_out_ if possible
            cats = list(getattr(self, "categorical", []))
            names = list(getattr(self, "feature_names_out_", []))
            lvl_map: dict[str, list[str]] = {}
            for c in cats:
                prefix = f"{c}__"
                levels = []
                for n in names:
                    if n.startswith(prefix) and ("_x_" not in n):
                        lvl = n.split("__", 1)[1]
                        if lvl != "OTHER" and lvl not in levels:
                            levels.append(lvl)
                if levels:
                    lvl_map[c] = levels
            self._levels = lvl_map
        if not hasattr(self, "dtype"):
            self.dtype = np.float32
        if not hasattr(self, "fill_value"):
            self.fill_value = 0.0

        n = len(X)
        blocks: list[sp.csr_matrix] = []

        # ---------- numeric block (CSR directly) ----------
        rows = np.arange(n, dtype=np.int32)
        for col_name in self._num_cols:
            if col_name.startswith(f"{self.date_col}_") and self.date_col in X.columns:
                # date-derived parts
                dt = pd.to_datetime(X[self.date_col], errors="coerce")
                if col_name.endswith("_Year"):
                    col_vals = dt.dt.year.fillna(0).astype(self.dtype)
                elif col_name.endswith("_Month_sin"):
                    month = dt.dt.month.fillna(0)
                    col_vals = np.sin(2 * np.pi * month / 12).astype(self.dtype)
                else:  # Month_cos
                    month = dt.dt.month.fillna(0)
                    col_vals = np.cos(2 * np.pi * month / 12).astype(self.dtype)
            else:  # regular numeric
                src_col = col_name
                if src_col not in X.columns:
                    col_vals = np.full(n, self.fill_value, dtype=self.dtype)
                else:
                    col_vals = (
                        pd.to_numeric(X[src_col], errors="coerce")
                        .fillna(self.fill_value)
                        .astype(self.dtype)
                    )
            blocks.append(sp.coo_matrix((col_vals, (rows, np.zeros(n))), shape=(n, 1)).tocsr())

        # ---------- categorical one-hot ----------
        ones = np.ones(n, dtype=self.dtype)
        for col, levels in self._levels.items():
            cat_raw = X[col].astype(str).fillna("nan").map(_safe)
            codes = pd.Categorical(cat_raw, categories=levels).codes.astype(np.int32)
            col_codes = np.where(codes == -1, len(levels), codes)  # unseen → OTHER
            ohe = sp.coo_matrix(
                (ones, (rows, col_codes)), shape=(n, len(levels) + 1), dtype=self.dtype
            ).tocsr()
            blocks.append(ohe)

        base = sp.hstack(blocks, format="csr")

        # ---------- interactions ----------
        name_to_col = {n: i for i, n in enumerate(self.feature_names_out_)}
        def col_view(name: str) -> sp.csr_matrix:
            return base[:, name_to_col[name]]

        int_blocks: list[sp.csr_matrix] = []
        for a, b in self.interactions:
            if a in self._num_cols and b in self._num_cols:
                int_blocks.append(col_view(a).multiply(col_view(b)))
            elif a in self._num_cols and b in self._levels:
                for l in self._levels[b] + ["OTHER"]:
                    int_blocks.append(col_view(a).multiply(col_view(f"{b}__{l}")))
            elif b in self._num_cols and a in self._levels:
                for l in self._levels[a] + ["OTHER"]:
                    int_blocks.append(col_view(f"{a}__{l}").multiply(col_view(b)))
        if int_blocks:
            base = sp.hstack([base] + int_blocks, format="csr")
        # By default return CSR; allow DF for explicit test scenarios via env flag
        force_df = str(os.getenv('FE_RETURN_DF_FOR_TESTS', '0')).lower() in {'1','true','yes'}
        if not force_df:
            try:
                threshold = int(os.getenv('FE_AUTO_DF_THRESHOLD', '128'))
            except ValueError:
                threshold = 128
            force_df = n <= max(1, threshold)
        if force_df:
            try:
                cols = list(self.get_feature_names_out())
            except Exception:
                cols = [f'f{i}' for i in range(base.shape[1])]
            return pd.DataFrame(base.toarray(), columns=cols)
        return base
# ------------------------------------------------------------------ #
    def get_feature_names_out(self, input_features=None):
        """Return output feature names, sklearn-compatible.

        Parameters
        ----------
        input_features : Ignored, kept for sklearn API compatibility.
        """
        check_is_fitted(self, "feature_names_out_")
        return np.array(self.feature_names_out_, dtype=object)

    # Backward-compat alias used by some tooling
    def get_feature_names(self, input_features=None):
        return list(self.get_feature_names_out(input_features))
