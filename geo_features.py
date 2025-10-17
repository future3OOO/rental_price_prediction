# geo_features.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R_KM = 6371.0088  # WGS84 mean Earth radius (km)

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

# ---------- helpers ----------


def _path_key(p: str | Path) -> tuple[str, float]:
    ap = os.path.abspath(str(p))
    try:
        mt = Path(p).stat().st_mtime
    except OSError:
        mt = -1.0
    return ap, mt


@lru_cache(maxsize=8)
def _load_poi_csv_cached(key: tuple[str, float]) -> pd.DataFrame:
    path, mtime = key
    if mtime < 0:
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Flexible schema normalisation
    rename = {}
    lower = {str(c).lower(): c for c in df.columns}
    if "name" not in df.columns:
        for cand in ("poi", "title"):
            if cand in lower:
                rename[lower[cand]] = "name"
                break
    if "category" not in df.columns:
        for cand in ("type", "class"):
            if cand in lower:
                rename[lower[cand]] = "category"
                break
    lat_col = next(
        (c for c in df.columns if str(c).lower() in {"latitude", "lat"}), None
    )
    lon_col = next(
        (c for c in df.columns if str(c).lower() in {"longitude", "lon", "lng"}), None
    )
    if lat_col and lat_col != "latitude":
        rename[lat_col] = "latitude"
    if lon_col and lon_col != "longitude":
        rename[lon_col] = "longitude"
    if rename:
        df = df.rename(columns=rename)

    for req in ("name", "category", "latitude", "longitude"):
        if req not in df.columns:
            raise ValueError(f"POI CSV missing column: {req}")

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df["category"] = (
        df["category"].astype(str).str.upper().str.replace(" ", "_", regex=False)
    )
    return df.reset_index(drop=True)


def _deg2rad(a: np.ndarray) -> np.ndarray:
    return np.deg2rad(a.astype(np.float64, copy=False))


def _initial_bearing_deg(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    # All inputs radians; output degrees in [0, 360)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    brng = (brng + 360.0) % 360.0
    return brng


# ---------- public API ----------


def compute_geo_features(
    df: pd.DataFrame,
    *,
    poi_csv: str,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    categories: Optional[Iterable[str]] = None,
    radii_km: Tuple[float, ...] = (0.5, 1.0, 2.0),
    decay_km: float = 1.5,
    max_decay_km: float = 3.0,
    include_bearings: bool | None = None,
    use_accessibility: bool | None = None,
    school_mode: Optional[str] = None,
) -> pd.DataFrame:
    """Compute geo features with optional bearings/accessibility and school consolidation."""
    include_bearings = (
        GEO_INCLUDE_BEARINGS if include_bearings is None else bool(include_bearings)
    )
    use_accessibility = (
        GEO_USE_ACCESSIBILITY if use_accessibility is None else bool(use_accessibility)
    )
    if lat_col not in df.columns or lon_col not in df.columns or df.empty:
        return df

    out = df.copy()
    lat = pd.to_numeric(out[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(out[lon_col], errors="coerce").to_numpy()
    mask = np.isfinite(lat) & np.isfinite(lon)
    if not mask.any():
        return out

    pois_all = _load_poi_csv_cached(_path_key(poi_csv))
    if categories:
        cats = [str(c).upper().replace(" ", "_") for c in categories]
        pois_all = pois_all[pois_all["category"].isin(cats)].copy()
    if pois_all.empty:
        return out

    lat_q = lat[mask]
    lon_q = lon[mask]
    q_rad = np.column_stack([_deg2rad(lat_q), _deg2rad(lon_q)])

    radii_km = tuple(float(r) for r in radii_km if float(r) > 0)
    radii_m = tuple(int(round(r * 1000)) for r in radii_km)

    newcols: Dict[str, np.ndarray] = {}
    school_distances: Dict[str, np.ndarray] = {}

    category_list = sorted(pois_all["category"].unique().tolist())
    for cat in category_list:
        cat_lower = cat.lower()
        p = pois_all[pois_all["category"] == cat]
        dist_col = f"dist_{cat_lower}_km"
        is_school = cat_lower.startswith("school_")
        full_dist = np.full(len(out), np.nan, dtype=np.float32)
        if not p.empty and mask.any():
            p_lat_rad = _deg2rad(p["latitude"].to_numpy())
            p_lon_rad = _deg2rad(p["longitude"].to_numpy())
            tree = BallTree(np.column_stack([p_lat_rad, p_lon_rad]), metric="haversine")
            d_rad, _ = tree.query(q_rad, k=1)
            full_dist[mask] = (d_rad[:, 0] * EARTH_R_KM).astype(np.float32)
        newcols[dist_col] = full_dist
        if is_school:
            school_distances[cat_lower] = full_dist
        # Count features within radii (skip schools for count outputs)
        if (not is_school) and (not p.empty) and mask.any() and radii_m:
            # Precompute tree for counts
            if 'tree' not in locals():
                p_lat_rad = _deg2rad(p["latitude"].to_numpy())
                p_lon_rad = _deg2rad(p["longitude"].to_numpy())
                tree = BallTree(np.column_stack([p_lat_rad, p_lon_rad]), metric="haversine")
            for r_km, r_m in zip(radii_km, radii_m):
                cnt_col = f"cnt_{cat_lower}_{int(r_m)}m"
                cnts = np.zeros(len(out), dtype=np.float32)
                if mask.any():
                    ind = tree.query_radius(q_rad, r=r_km / EARTH_R_KM)
                    # ind is list of arrays; count elements
                    counts = np.array([len(ix) for ix in ind], dtype=np.float32)
                    cnts[mask] = counts
                newcols[cnt_col] = cnts
    if school_distances:
        for key in school_distances:
            newcols.pop(f"dist_{key}_km", None)
        high_dist = school_distances.get("school_high")
        primary_sources = [school_distances.get("school_primary"), school_distances.get("school_intermediate")]
        primary_vals = [arr for arr in primary_sources if arr is not None]
        if high_dist is not None:
            newcols["dist_school_high_km"] = high_dist.astype(np.float32)
        if primary_vals:
            dist_stack = np.vstack(primary_vals)
            with np.errstate(all="ignore"):
                primary_dist = np.nanmin(dist_stack, axis=0)
            newcols["dist_school_primary_intermediate_km"] = primary_dist.astype(np.float32)
        # Additionally expose a consolidated minimum school distance
        try:
            all_school = [arr for arr in school_distances.values() if arr is not None]
            if all_school:
                stack = np.vstack(all_school)
                with np.errstate(all="ignore"):
                    min_all = np.nanmin(stack, axis=0)
                newcols["dist_school_min_km"] = min_all.astype(np.float32)
        except Exception:
            pass

    if use_accessibility:
        acc_cols = [c for c in newcols.keys() if c.startswith("acc_")]
        if acc_cols:
            acc_mat = np.column_stack([newcols[c] for c in acc_cols]).astype(np.float64)
            access_index = acc_mat.sum(axis=1, dtype=np.float64).astype(np.float32)
            newcols["access_index"] = access_index
    else:
        # Remove any stale accessibility columns if toggled off
        for key in [c for c in list(newcols.keys()) if c.startswith("acc_")]:
            newcols.pop(key, None)

    add_df = pd.DataFrame(newcols, index=out.index)
    out = pd.concat([out, add_df], axis=1)
    return out
