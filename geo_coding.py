# geo_coding.py
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any, List, Mapping, NamedTuple

import numpy as np
import pandas as pd
import requests


# ----------------------------- Cache (SQLite) ----------------------------- #

class GeocodeCache:
    """Tiny SQLite cache: query text -> (lat, lon, raw_json, ts)."""

    def __init__(self, path: str | Path = "artifacts/geocode_cache.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS geocode_cache (
                 q TEXT PRIMARY KEY,
                 lat REAL,
                 lon REAL,
                 raw_json TEXT,
                 ts REAL
            )"""
        )
        self._conn.commit()

    def get(self, q: str) -> Optional[Tuple[float, float]]:
        cur = self._conn.execute("SELECT lat, lon FROM geocode_cache WHERE q = ?", (q,))
        row = cur.fetchone()
        return (float(row[0]), float(row[1])) if row else None

    def set(self, q: str, lat: float, lon: float, raw: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO geocode_cache (q, lat, lon, raw_json, ts) VALUES (?, ?, ?, ?, ?)",
            (q, float(lat), float(lon), json.dumps(raw, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# --------------------------- Rate limiter --------------------------- #

class RateLimiter:
    """Enforces a minimum interval between calls (per-process)."""

    def __init__(self, min_interval_sec: float = 1.2) -> None:
        self.min_interval = float(min_interval_sec)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        lag = self.min_interval - (now - self._last)
        if lag > 0:
            time.sleep(lag)
        self._last = time.monotonic()


# ----------------------------- Geocoder ----------------------------- #

@dataclass
class GeocodeResult:
    query: str
    lat: float
    lon: float
    ok: bool
    provider: str
    meta: Dict[str, Any]


def _parse_viewbox_from_env() -> Optional[Tuple[float, float, float, float]]:
    """Return (minlon, minlat, maxlon, maxlat) or None."""
    lat_range = os.getenv("GEO_BBOX_LAT")
    lon_range = os.getenv("GEO_BBOX_LON")
    if not lat_range or not lon_range:
        return None
    try:
        lat1, lat2 = [float(x.strip()) for x in lat_range.split(",")]
        lon1, lon2 = [float(x.strip()) for x in lon_range.split(",")]
        minlat, maxlat = (lat1, lat2) if lat1 <= lat2 else (lat2, lat1)
        minlon, maxlon = (lon1, lon2) if lon1 <= lon2 else (lon2, lon1)
        # basic sanity for Christchurch region
        if not (-44.5 <= minlat <= -42.0 and -44.5 <= maxlat <= -42.0 and 171.0 <= minlon <= 174.0 and 171.0 <= maxlon <= 174.0):
            # If weird box, ignore it rather than break geocoding
            return None
        return (minlon, minlat, maxlon, maxlat)
    except Exception:
        return None


class NominatimGeocoder:
    """
    OpenStreetMap Nominatim geocoder (read-only; be polite).
    Respects rate limits and supports bounded & structured fallbacks.
    """

    def __init__(
        self,
        base_url: str = "https://nominatim.openstreetmap.org/search",
        *,
        email: Optional[str] = None,
        min_interval_sec: Optional[float] = None,
        cache: Optional[GeocodeCache] = None,
        countrycodes: str = "nz",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url
        self.email = email or os.getenv("NOMINATIM_EMAIL") or "sylabis@gmail.com"

        self.countrycodes = countrycodes
        self.cache = cache or GeocodeCache()
        self.session = session or requests.Session()

        if min_interval_sec is None:
            qps_env = os.getenv("GEOCODER_QPS")
            if qps_env:
                try:
                    qps = max(float(qps_env), 1e-6)
                    min_interval_sec = 1.0 / qps
                except Exception:
                    pass
        if min_interval_sec is None:
            try:
                min_interval_sec = float(os.getenv("GEOCODER_MIN_INTERVAL", "1.2"))
            except Exception:
                min_interval_sec = 1.2
        self.ratelimit = RateLimiter(min_interval_sec=min_interval_sec)

        self.viewbox = _parse_viewbox_from_env()
        self.use_bounded_default = os.getenv("GEOCODER_BOUNDED", "1").lower() in {"1", "true", "yes"}

    def _headers(self) -> Dict[str, str]:
        return {
            "User-Agent": f"RentalModel-Geocoder/1.0 ({self.email})",
            "Accept": "application/json",
        }

    def _request(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.ratelimit.wait()
        resp = self.session.get(self.base_url, params=params, headers=self._headers(), timeout=20)
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _search_free(self, q: str, *, city_bias: Optional[str], bounded: bool) -> List[Dict[str, Any]]:
        params = {
            "format": "json",
            "limit": 1,
            "countrycodes": self.countrycodes,
            "addressdetails": 0,
            "q": f"{q}, {city_bias}" if city_bias else q,
        }
        if bounded and self.viewbox is not None:
            minlon, minlat, maxlon, maxlat = self.viewbox
            params["viewbox"] = f"{minlon},{minlat},{maxlon},{maxlat}"
            params["bounded"] = 1
        return self._request(params)

    def _search_structured(
        self,
        *,
        street: Optional[str],
        city: Optional[str],
        postcode: Optional[str],
        county: Optional[str],
        state: Optional[str],
        bounded: bool,
    ) -> List[Dict[str, Any]]:
        params = {
            "format": "json",
            "limit": 1,
            "countrycodes": self.countrycodes,
            "addressdetails": 0,
        }
        if street:   params["street"]   = street
        if city:     params["city"]     = city
        if postcode: params["postalcode"] = postcode
        if county:   params["county"]   = county
        if state:    params["state"]    = state
        if bounded and self.viewbox is not None:
            minlon, minlat, maxlon, maxlat = self.viewbox
            params["viewbox"] = f"{minlon},{minlat},{maxlon},{maxlat}"
            params["bounded"] = 1
        return self._request(params)

    def geocode_structured(
        self,
        *,
        housenumber: str | None,
        road: str | None,
        suburb: str | None = None,
        city: str | None = None,
        county: str | None = None,
        state: str | None = None,
        postalcode: str | None = None,
        country: str = "New Zealand",
    ) -> GeocodeResult:
        """Structured-only lookup. Uses same endpoint but with structured params.

        Returns cached result if available (cache key: deterministic param string).
        """
        params: Dict[str, Any] = {
            "format": "json",
            "limit": 1,
            "countrycodes": self.countrycodes,
            "addressdetails": 0,
        }
        if country:
            params["country"] = country
        if postalcode:
            params["postalcode"] = str(postalcode).strip()
        if housenumber or road:
            street = " ".join([p for p in [housenumber, road] if p]).strip()
            if street:
                params["street"] = street
        if suburb:
            params["suburb"] = suburb
        if city:
            params["city"] = city
        if county:
            params["county"] = county
        if state:
            params["state"] = state
        # Bound to viewbox if available
        if self.viewbox is not None:
            minlon, minlat, maxlon, maxlat = self.viewbox
            params["viewbox"] = f"{minlon},{minlat},{maxlon},{maxlat}"
            params["bounded"] = 1

        # Deterministic cache key
        key_parts: List[str] = ["struct"]
        for k in ("street", "suburb", "city", "county", "state", "postalcode", "country"):
            v = str(params.get(k, "")) if params.get(k) is not None else ""
            key_parts.append(f"{k}={v}")
        cache_key = "|".join(key_parts)

        cached = self.cache.get(cache_key)
        if cached:
            return GeocodeResult(cache_key, cached[0], cached[1], True, "cache", {})

        data = self._request(params)
        if not data:
            return GeocodeResult(cache_key, float("nan"), float("nan"), False, "nominatim", {})
        lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
        self.cache.set(cache_key, lat, lon, data[0])
        return GeocodeResult(cache_key, lat, lon, True, "nominatim", {"raw": data[0]})

    def geocode(self, q: str, city_bias: Optional[str] = None) -> GeocodeResult:
        q_norm = " ".join(str(q).strip().split())
        if not q_norm:
            return GeocodeResult(q_norm, float("nan"), float("nan"), False, "nominatim", {})

        cached = self.cache.get(q_norm)
        if cached:
            return GeocodeResult(q_norm, cached[0], cached[1], True, "cache", {})

        # 1) Free-text bounded → free-text unbounded
        for bounded in (self.use_bounded_default, False):
            data = self._search_free(q_norm, city_bias=city_bias, bounded=bounded)
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                self.cache.set(q_norm, lat, lon, data[0])
                return GeocodeResult(q_norm, lat, lon, True, "nominatim", {"bounded": bounded})

        # 2) Structured fallback (NZ-friendly)
        street, city, postcode = _parse_addr_for_structured(q_norm)
        county = "Christchurch City" if not city_bias else None
        state  = "Canterbury"
        for bounded in (self.use_bounded_default, False):
            data = self._search_structured(
                street=street, city=city or "Christchurch", postcode=postcode, county=county, state=state, bounded=bounded
            )
            if data:
                lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
                self.cache.set(q_norm, lat, lon, data[0])
                return GeocodeResult(q_norm, lat, lon, True, "nominatim", {"structured": True, "bounded": bounded})

        return GeocodeResult(q_norm, float("nan"), float("nan"), False, "nominatim", {})

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass
        self.cache.close()


# ----------------------- Query building & normalisation ----------------------- #

_GEO_QUERY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "Street Address": ("Street Address", "street_address", "address"),
    "Suburb": ("Suburb", "suburb"),
    "City": ("City", "city"),
    "Postcode": ("Postcode", "postcode", "postal_code", "post_code"),
}

# --- NZ-specific normalization helpers for structured lookups ---
_UNIT_PREFIX_NZ = re.compile(r"^(unit|flat|suite|apt|apartment|level)\s+\S+\s*", re.I)
_UNIT_SLASH = re.compile(r"^(\d{1,4})/(\d{1,6})\s+(.*)$")  # e.g., 6/17 Foo Street
_NUM_ROAD = re.compile(r"^(\d{1,6}[A-Za-z]?)\s+(.*)$")      # e.g., 17A Foo Street
_PARENS = re.compile(r"\(.*?\)")
_SPACES = re.compile(r"\s+")

_ROAD_ABBR = {
    "ST": "Street", "RD": "Road", "AVE": "Avenue", "AV": "Avenue",
    "DR": "Drive", "PL": "Place", "CRES": "Crescent", "CT": "Court",
    "HWY": "Highway", "MT": "Mount",
}

_SUBURB_FIX = {
    "ST ALBANS": "St Albans",
    "UPPER RICCARTON": "Upper Riccarton",
    "PHILLIPSTOWN": "Phillipstown",
    "RUSLEY": "Russley",
}

def _title_road(road: str) -> str:
    s = _PARENS.sub("", road or "").strip()
    s = _SPACES.sub(" ", s)
    parts = s.split(" ") if s else []
    if not parts:
        return s
    last = parts[-1].strip(".").upper()
    parts[-1] = _ROAD_ABBR.get(last, parts[-1].title())
    for i in range(len(parts) - 1):
        parts[i] = parts[i].title()
    return " ".join(parts).strip()

def _canon_suburb(s: str | None) -> str | None:
    if not s:
        return None
    u = str(s).strip()
    if not u:
        return None
    fix = _SUBURB_FIX.get(u.upper())
    return fix or u.title()

class StreetParts(NamedTuple):
    unit: str | None
    housenumber: str | None
    road: str | None

def _parse_street(street: str | None) -> StreetParts:
    if not street:
        return StreetParts(None, None, None)
    s = re.sub(r",+$", "", str(street).strip())
    s = _UNIT_PREFIX_NZ.sub("", s).strip()
    m = _UNIT_SLASH.match(s)
    if m:
        unit, house, road = m.group(1), m.group(2), m.group(3)
        return StreetParts(unit, house, _title_road(road))
    m = _NUM_ROAD.match(s)
    if m:
        house, road = m.group(1), m.group(2)
        return StreetParts(None, house, _title_road(road))
    return StreetParts(None, None, _title_road(s))

_ABBREV = {
    r"\bSt\b": "Street",
    r"\bSt\.\b": "Street",
    r"\bRd\b": "Road",
    r"\bRd\.\b": "Road",
    r"\bAve\b": "Avenue",
    r"\bAve\.\b": "Avenue",
    r"\bHwy\b": "Highway",
    r"\bMt\b": "Mount",
    r"\bSq\b": "Square",
    r"\bPl\b": "Place",
    r"\bPl\.\b": "Place",
    r"\bCt\b": "Court",
    r"\bLn\b": "Lane",
    r"\bTer\b": "Terrace",
}

_UNIT_PREFIX = re.compile(r"^(unit|flat|suite|apt|apartment|level)\s+\S+\s+", re.I)

def _expand_abbrev(s: str) -> str:
    out = s
    for pat, repl in _ABBREV.items():
        out = re.sub(pat, repl, out, flags=re.I)
    return out

def _clean_component_value(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().strip('"').strip("'")
    if not s:
        return ""
    if re.fullmatch(r"-?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except Exception:
            pass
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",+$", "", s)
    return s.strip()

def _normalise_component_series(values: pd.Series) -> pd.Series:
    return values.apply(_clean_component_value).astype("string")

def _normalise_query_text(text: str) -> str:
    s = _clean_component_value(text)
    if not s:
        return ""
    if os.getenv("GEO_EXPAND_ABBREV", "0").lower() in {"1", "true", "yes"}:
        s = _expand_abbrev(s)
    s = _UNIT_PREFIX.sub("", s)
    s = re.sub(r",\s*,", ", ", s)
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"P\.?O\.?\s*Box.*$", "", s, flags=re.I)
    s = re.sub(r",\s*$", "", s)
    return s.strip()

def make_geo_query_series(
    df: pd.DataFrame,
    *,
    column_aliases: Mapping[str, Iterable[str]] | None = None,
    fallback_city: str | None = None,
    fallback_country: str | None = None,
) -> pd.Series:
    if column_aliases is None:
        column_aliases = _GEO_QUERY_ALIASES
    fallback_city = (fallback_city or os.getenv("GEO_FALLBACK_CITY") or "Christchurch").strip()
    fallback_country = (fallback_country or os.getenv("GEO_FALLBACK_COUNTRY") or "New Zealand").strip()

    if df.empty:
        return pd.Series([], index=df.index, dtype="string")

    components: Dict[str, pd.Series] = {}
    for canonical, aliases in column_aliases.items():
        series = None
        for cand in aliases:
            matches = [col for col in df.columns if str(col).lower() == str(cand).lower()]
            if matches:
                series = _normalise_component_series(df[matches[0]])
                break
        if series is None:
            series = pd.Series(["" for _ in range(len(df))], index=df.index, dtype="string")
        components[canonical] = series

    comp_df = pd.DataFrame(components, index=df.index)

    def _compose(row: pd.Series) -> str:
        street = _normalise_query_text(row.get("Street Address", ""))
        suburb = _clean_component_value(row.get("Suburb", ""))
        city   = _clean_component_value(row.get("City", "")) or fallback_city
        pc     = _clean_component_value(row.get("Postcode", ""))
        parts: List[str] = []
        if street: parts.append(street)
        if suburb: parts.append(suburb)
        if city:   parts.append(city)
        if pc:     parts.append(pc)
        parts.append(fallback_city)
        parts.append(fallback_country)
        return ", ".join(parts)

    query = comp_df.apply(_compose, axis=1).astype("string")
    return query

def assign_geo_queries(
    df: pd.DataFrame,
    *,
    query_col: str = "__geo_query__",
    column_aliases: Mapping[str, Iterable[str]] | None = None,
    fallback_city: str | None = None,
    fallback_country: str | None = None,
) -> pd.DataFrame:
    work = df.copy()
    work[query_col] = make_geo_query_series(
        work, column_aliases=column_aliases, fallback_city=fallback_city, fallback_country=fallback_country
    )
    return work


# ----------------------- Reading precomputed results ----------------------- #

def _geocode_results_path_key(path: str | Path) -> Tuple[str, float]:
    p = Path(path)
    abs_path = os.path.abspath(str(p))
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = -1.0
    return abs_path, mtime

@lru_cache(maxsize=8)
def _load_geocode_results_cached(path_key: Tuple[str, float]) -> pd.DataFrame:
    path, mtime = path_key
    if mtime < 0:
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rename_map = {}
    if "original_query" in df.columns and "query" not in df.columns:
        rename_map["original_query"] = "query"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "query" not in df.columns:
        raise ValueError("geocode results missing 'query' column")
    lat_col = next((c for c in df.columns if str(c).lower() in {"latitude", "lat"}), "latitude")
    lon_col = next((c for c in df.columns if str(c).lower() in {"longitude", "lon", "lng"}), "longitude")
    df = df.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    df["query"] = df["query"].fillna("").astype(str).map(_normalise_query_text)
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["query", "latitude", "longitude"])
    df = df.drop_duplicates(subset=["query"], keep="last").reset_index(drop=True)
    return df

def merge_geocode_results(
    df: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    query_col: str = "__geo_query__",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or query_col not in out.columns or geo_df.empty:
        return out
    geo_norm = geo_df.copy()
    if "query" not in geo_norm.columns:
        return out
    geo_norm["query_norm"] = geo_norm["query"].fillna("").astype(str).map(_normalise_query_text)
    geo_norm["latitude"]  = pd.to_numeric(geo_norm["latitude"], errors="coerce")
    geo_norm["longitude"] = pd.to_numeric(geo_norm["longitude"], errors="coerce")
    geo_norm = geo_norm.dropna(subset=["query_norm", "latitude", "longitude"])
    if geo_norm.empty:
        return out
    geo_norm = geo_norm.drop_duplicates(subset=["query_norm"], keep="last")
    mapping = geo_norm.set_index("query_norm")
    queries = out[query_col].fillna("").astype(str).map(_normalise_query_text)
    lat_match = queries.map(mapping["latitude"])
    lon_match = queries.map(mapping["longitude"])
    if lat_col not in out.columns:
        out[lat_col] = np.nan
    if lon_col not in out.columns:
        out[lon_col] = np.nan
    lat_mask = out[lat_col].isna() & lat_match.notna()
    lon_mask = out[lon_col].isna() & lon_match.notna()
    if lat_mask.any(): out.loc[lat_mask, lat_col] = lat_match[lat_mask].astype(float)
    if lon_mask.any(): out.loc[lon_mask, lon_col] = lon_match[lon_mask].astype(float)
    return out

def enrich_with_geocodes(
    df: pd.DataFrame,
    *,
    query_col: str = "__geo_query__",
    results_path: str | Path | None = None,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or query_col not in out.columns:
        return out
    path = Path(results_path) if results_path is not None else Path(os.getenv("GEOCODE_RESULTS_CSV", "artifacts/geocode_query_results.csv"))
    key = _geocode_results_path_key(path)
    if key[1] < 0:
        return out
    try:
        geo_df = _load_geocode_results_cached(key)
    except Exception:
        return out
    if geo_df.empty:
        return out
    return merge_geocode_results(out, geo_df, query_col=query_col, lat_col=lat_col, lon_col=lon_col)


# ----------------------- Property batch geocoding ----------------------- #

def _parse_addr_for_structured(q: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (street, city, postcode) approximations from free text."""
    parts = [p.strip() for p in q.split(",") if p.strip()]
    # postcode: last numeric token of length 4
    postcode = next((p for p in parts if re.fullmatch(r"\d{4}", p)), None)
    # city: prefer Christchurch/Lyttelton/Diamond Harbour etc.
    city = next((p for p in parts if re.search(r"christchurch|lyttelton|kaiapoi|rangiora|ashburton", p, re.I)), None)
    # street: first part, strip unit markers and expand abbreviations
    street = parts[0] if parts else None
    if street:
        street = _UNIT_PREFIX.sub("", street)
        street = _expand_abbrev(street)
    return street, city, postcode

def compose_address(
    row: pd.Series,
    *,
    street_cols: Iterable[str] = ("street_address",),
    suburb_col: str = "suburb",
    city_col: str = "city",
    postcode_col: str = "postcode",
    country: str = "New Zealand",
) -> str:
    parts: List[str] = []
    for c in street_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(str(row[c]).strip())
    for c in (suburb_col, city_col, postcode_col):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            s = str(row[c]).strip()
            if s.endswith(".0"): s = s[:-2]
            parts.append(s)
    parts.append(country)
    # de-dup adjacent
    seen, out = set(), []
    for p in parts:
        pl = p.lower()
        if pl in seen: continue
        seen.add(pl)
        out.append(p)
    return ", ".join(out)

def _address_candidates_minimal(
    row: pd.Series,
    *,
    street_cols: Iterable[str],
    suburb_col: str,
    city_col: str,
    postcode_col: str,
) -> tuple[dict, list[str]]:
    """Return structured dict for primary lookup and at most two q-fallbacks."""
    street_raw = None
    for c in street_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            street_raw = str(row[c]).strip()
            break

    unit, housenumber, road = _parse_street(street_raw)
    suburb = _canon_suburb(str(row.get(suburb_col, "")).strip() or None)
    city = "Christchurch"
    postal = None
    if postcode_col in row and pd.notna(row[postcode_col]) and str(row[postcode_col]).strip():
        p = str(row[postcode_col]).strip()
        postal = p[:-2] if p.endswith(".0") else p

    struct = {
        "housenumber": housenumber,
        "road": road,
        "suburb": suburb,
        "city": city,
        "county": "Christchurch City",
        "state": "Canterbury",
        "postalcode": postal,
        "country": "New Zealand",
    }

    q_fallbacks: list[str] = []
    if housenumber and road and suburb:
        q_fallbacks.append(f"{housenumber} {road}, {suburb}, Christchurch, New Zealand")
    elif housenumber and road:
        q_fallbacks.append(f"{housenumber} {road}, Christchurch, New Zealand")
    if road and suburb:
        q_fallbacks.append(f"{road}, {suburb}, Christchurch, New Zealand")

    seen: set[str] = set()
    q_fallbacks = [x for x in q_fallbacks if not (x in seen or seen.add(x))]
    return struct, q_fallbacks[:2]

def geocode_properties(
    df: pd.DataFrame,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    street_cols: Iterable[str] = ("street_address",),
    suburb_col: str = "suburb",
    city_col: str = "city",
    postcode_col: str = "postcode",
    city_bias: str = "Christchurch, Canterbury",
    geocoder: Optional[NominatimGeocoder] = None,
    centroid_fallback: Optional[pd.DataFrame] = None,  # optional: columns suburb|postcode|Latitude|Longitude
) -> pd.DataFrame:
    """Fill missing Latitude/Longitude via cache-first geocoding with robust fallbacks."""
    g = geocoder or NominatimGeocoder()
    work = df.copy()
    if lat_col not in work.columns:
        work[lat_col] = np.nan
    if lon_col not in work.columns:
        work[lon_col] = np.nan

    need = ~(work[lat_col].notna() & work[lon_col].notna())
    if not need.any():
        return work

    for idx, row in work.loc[need].iterrows():
        struct, q_fallbacks = _address_candidates_minimal(
            row,
            street_cols=street_cols,
            suburb_col=suburb_col,
            city_col=city_col,
            postcode_col=postcode_col,
        )
        # 1) Structured first
        try:
            res = g.geocode_structured(**struct)
            if res.ok and np.isfinite(res.lat) and np.isfinite(res.lon):
                work.at[idx, lat_col] = res.lat
                work.at[idx, lon_col] = res.lon
                continue
        except Exception:
            pass

        # 2) Minimal q fallbacks (<=2)
        success = False
        for q in q_fallbacks:
            try:
                res = g.geocode(q, city_bias=None)
                if res.ok and np.isfinite(res.lat) and np.isfinite(res.lon):
                    work.at[idx, lat_col] = res.lat
                    work.at[idx, lon_col] = res.lon
                    success = True
                    break
            except Exception:
                continue
        if success:
            continue

        # 3) Last resort: full composed free-text with bias
        try:
            q = compose_address(
                row,
                street_cols=street_cols,
                suburb_col=suburb_col,
                city_col=city_col,
                postcode_col=postcode_col,
                country="New Zealand",
            )
            res = g.geocode(q, city_bias=city_bias)
            if res.ok and np.isfinite(res.lat) and np.isfinite(res.lon):
                work.at[idx, lat_col] = res.lat
                work.at[idx, lon_col] = res.lon
        except Exception:
            pass

    # Optional centroid fallback for final misses
    if centroid_fallback is not None:
        remaining = work[lat_col].isna() | work[lon_col].isna()
        if remaining.any():
            cf = centroid_fallback.copy()
            # choose suburb first, then postcode
            if "suburb" in work.columns and "suburb" in cf.columns:
                m = work.loc[remaining, "suburb"].astype(str).str.lower().map(
                    cf.set_index(cf["suburb"].astype(str).str.lower())[["Latitude", "Longitude"]].to_dict(orient="index")
                )
                mm = m.notna()
                if mm.any():
                    work.loc[remaining[remaining].index[mm], lat_col] = [m[i]["Latitude"] for i in m[mm].index]
                    work.loc[remaining[remaining].index[mm], lon_col] = [m[i]["Longitude"] for i in m[mm].index]
            remaining = work[lat_col].isna() | work[lon_col].isna()
            if remaining.any() and "postcode" in work.columns and "postcode" in cf.columns:
                m = work.loc[remaining, "postcode"].astype(str).map(
                    cf.set_index(cf["postcode"].astype(str))[["Latitude", "Longitude"]].to_dict(orient="index")
                )
                mm = m.notna()
                if mm.any():
                    work.loc[remaining[remaining].index[mm], lat_col] = [m[i]["Latitude"] for i in m[mm].index]
                    work.loc[remaining[remaining].index[mm], lon_col] = [m[i]["Longitude"] for i in m[mm].index]

    return work


# ----------------------- Christchurch landmarks → CSV ----------------------- #

DEFAULT_LANDMARKS: Dict[str, Iterable[str]] = {
    "CBD": ["Cathedral Square", "Te Pae Christchurch Convention Centre"],
    "UNIVERSITY": ["University of Canterbury", "Ara Institute of Canterbury City Campus"],
    "HOSPITAL": ["Christchurch Hospital", "Burwood Hospital"],
    "MALL": ["Westfield Riccarton", "Northlands Mall", "The Palms", "Eastgate Mall", "Merivale Mall"],
    "PARK": ["Hagley Park", "Christchurch Botanic Gardens", "Victoria Park"],
    "BUS": ["Christchurch Bus Interchange"],
    "TRAIN": ["Christchurch Railway Station Addington"],
    "AIRPORT": ["Christchurch International Airport"],
    "BEACH": ["New Brighton Pier", "Sumner Beach"],
    "WILDLIFE_PARK": ["Orana Wildlife Park", "Willowbank Wildlife Reserve"],
    "STADIUM": ["Apollo Projects Stadium"],  # Rugby Park
    "MUSEUM": ["Canterbury Museum"],
    "ART_GALLERY": ["Christchurch Art Gallery Te Puna o Waiwhetū"],
    "GONDOLA": ["Christchurch Gondola Base Station"],
    "LIBRARY": ["Tūranga Central Library"],
    # Schools – top high schools/intermediates/primaries (sample starters; extend freely)
    "SCHOOL_HIGH": [
        "Christchurch Boys' High School",
        "Christchurch Girls' High School",
        "Burnside High School",
        "Riccarton High School",
        "Shirley Boys' High School",
        "Avonside Girls' High School",
        "Middleton Grange School",
        "St Andrew's College",
        "St Thomas of Canterbury College",
        "Hillmorton High School",
        "Cashmere High School",
        "Papanui High School",
        "St Bede's College",
        "Rangi Ruru Girls' School",
    ],
    "SCHOOL_INTERMEDIATE": [
        "Cobham Intermediate School",
        "Casebrook Intermediate School",
        "Kirkwood Intermediate School",
        "Chisnallwood Intermediate School",
        "Breens Intermediate",
        "Heaton Normal Intermediate",
    ],
    "SCHOOL_PRIMARY": [
        "Fendalton Open Air School",
        "Ilam School",
        "Merrin School",
        "Paparoa Street School",
        "Redwood School",
        "Somerset Crescent School",  # update/curate as needed
        "Thorrington School",
        "Spreydon School",
        "Westburn School",
        "West Spreydon School",
        "Riccarton Primary School",
        "Burnside Primary School",
        "Halswell School",
    ],
}

def build_landmarks_csv(
    out_csv: str | Path,
    *,
    landmarks: Optional[Dict[str, Iterable[str]]] = None,
    seed_csv: str | Path | None = None,
    extra_pois: Optional[Iterable[Dict[str, Any]]] = None,
    city_bias: str = "Christchurch, Canterbury",
    geocoder: Optional[NominatimGeocoder] = None,
) -> pd.DataFrame:
    """Geocode POIs (landmarks) and write CSV: name, category, latitude, longitude, source."""
    g = geocoder or NominatimGeocoder()

    def _cat(cat: str) -> str:
        return str(cat).strip().upper().replace(" ", "_")

    poi_specs: List[Dict[str, Any]] = []

    if seed_csv:
        seed_path = Path(seed_csv)
        if not seed_path.exists():
            raise FileNotFoundError(seed_path)
        seed_df = pd.read_csv(seed_path)
        for _, row in seed_df.iterrows():
            name = str(row.get("name", "")).strip()
            category = str(row.get("category", "")).strip()
            if not name or not category:
                continue
            poi_specs.append({
                "name": name,
                "category": _cat(category),
                "query": (str(row.get("query", "")).strip() or None),
                "city": (str(row.get("city", "")).strip() or None),
                "country": (str(row.get("country", "")).strip() or None),
            })

    if extra_pois:
        for item in extra_pois:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            category = str(item.get("category", "")).strip()
            if not name or not category:
                continue
            poi_specs.append({
                "name": name,
                "category": _cat(category),
                "query": (str(item.get("query", "")).strip() or None),
                "city": (str(item.get("city", "")).strip() or None),
                "country": (str(item.get("country", "")).strip() or None),
            })

    if landmarks is not None:
        lm_iter = landmarks.items()
    elif seed_csv is None:
        lm_iter = DEFAULT_LANDMARKS.items()
    else:
        lm_iter = []
    for cat, names in lm_iter:
        for name in names:
            poi_specs.append({"name": str(name).strip(), "category": _cat(cat), "query": None, "city": None, "country": None})

    # Deduplicate (category, name)
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for spec in poi_specs:
        k = (spec["category"], spec["name"])
        dedup[k] = spec

    results: List[Dict[str, Any]] = []
    for spec in dedup.values():
        name, category = spec["name"], spec["category"]
        query = spec.get("query")
        city  = spec.get("city") or city_bias
        country = spec.get("country") or "New Zealand"

        def _strip_paren(x: str) -> str: return re.sub(r"\s*\([^)]*\)", "", x or "").strip()

        candidates: List[Tuple[str, Optional[str]]] = []
        if query:
            candidates.append((query, None))
        candidates.append((", ".join([name, city, country]), None))
        simple = _strip_paren(name)
        if simple and simple != name:
            candidates.append((", ".join([simple, city, country]), None))
            candidates.append((simple, city))
        candidates.append((name, city))

        seen: set[str] = set()
        for q, bias in candidates:
            qn = q.strip()
            if not qn or qn in seen:
                continue
            seen.add(qn)
            try:
                res = g.geocode(qn, city_bias=bias)
                if res.ok:
                    results.append({"name": name, "category": category, "latitude": res.lat, "longitude": res.lon, "source": res.provider})
                    break
            except Exception:
                continue

    df = pd.DataFrame(results, columns=["name", "category", "latitude", "longitude", "source"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df




