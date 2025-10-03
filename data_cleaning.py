# data_cleaning.py

import pandas as pd
import numpy as np
import logging
import traceback
from config import CLEANED_DATA_PATH  # ensure your config.py has the correct path
import re
import os
from geo_coding import assign_geo_queries, enrich_with_geocodes


_ALIAS_MAP = {
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
    "pets": "Pets",
    "carparks": "Car",
    "garage": "garage",
    "car": "Car",
    "latitude": "Latitude",
    "lat": "Latitude",
    "gps_lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "lng": "Longitude",
    "gps_lon": "Longitude",
}



_FURNISHINGS_CANONICAL = {
    # Lowercase to align with training artifacts
    "unfurnished": "unfurnished",
    "furnished": "furnished",
    "partially furnished": "partially furnished",
    "partial": "partially furnished",
    "fully furnished": "furnished",
    "full": "furnished",
}


def _canonicalize_furnishings(value: object) -> str:
    """Return lowercase categories matching training ('unfurnished'|'furnished'|'partially furnished')."""
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


def _normalize_furnishings_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Furnishings", "furnishings") if c in df.columns]
    if not cols:
        return df
    work = df.copy()
    for col in cols:
        work[col] = work[col].apply(_canonicalize_furnishings)
    if "Furnishings" in cols and "furnishings" not in cols:
        work["furnishings"] = work["Furnishings"]
    if "furnishings" in cols and "Furnishings" not in cols:
        work["Furnishings"] = work["furnishings"]
    return work


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


def _normalize_pets_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Pets", "pets") if c in df.columns]
    if not cols:
        return df
    work = df.copy()
    for col in cols:
        work[col] = work[col].apply(_canonicalize_pets)
    if "Pets" in cols and "pets" not in cols:
        work["pets"] = work["Pets"]
    if "pets" in cols and "Pets" not in cols:
        work["Pets"] = work["pets"]
    return work


_GARAGE_CANONICAL = {
    "no": "No Garage",
    "none": "No Garage",
    "nil": "No Garage",
    "n": "No Garage",
    "0": "No Garage",
    "yes": "Yes",
    "y": "Yes",
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
    return text_val.title()


def _normalize_garage_column(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("Garage", "garage") if c in df.columns]
    if not cols:
        return df
    work = df.copy()
    for col in cols:
        work[col] = work[col].apply(_canonicalize_garage)
    if "Garage" in cols and "garage" not in cols:
        work["garage"] = work["Garage"]
    if "garage" in cols and "Garage" not in cols:
        work["Garage"] = work["garage"]
    return work


def _apply_alias_mapping(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    """Rename alias columns to canonical names and merge duplicate information."""
    work = df.copy()
    for alias_lower, canonical in alias_map.items():
        matching = [col for col in work.columns if col.lower() == alias_lower]
        for col in matching:
            if col == canonical:
                continue
            if canonical in work.columns:
                work[canonical] = work[canonical].combine_first(work[col])
                work.drop(columns=[col], inplace=True)
            else:
                work.rename(columns={col: canonical}, inplace=True)
    return work


def _normalize_aliases_and_ranges(df_in: pd.DataFrame) -> pd.DataFrame:
    """Map flexible CSV schemas to canonical columns and apply clamps."""
    df = _apply_alias_mapping(df_in, _ALIAS_MAP)
    df = _normalize_furnishings_column(df)
    df = _normalize_pets_column(df)
    df = _normalize_garage_column(df)

    lower_cols = {c.lower(): c for c in df.columns}

    # Date parse (dayfirst)
    if "Last Rental Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["Last Rental Date"]
    ):
        df["Last Rental Date"] = pd.to_datetime(
            df["Last Rental Date"], dayfirst=True, errors="coerce"
        )

    # Postcode as 4-digit string
    if "Postcode" in df.columns:
        df["Postcode"] = (
            pd.to_numeric(df["Postcode"], errors="coerce")
            .astype("Int64")
            .astype("string")
            .str.zfill(4)
        )

    # Latitude/Longitude numeric coercion
    if "Latitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    if "Longitude" in df.columns:
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    # Derive Car from components when present
    if "Car" not in df.columns:
        g_col = lower_cols.get("garage_parks")
        o_col = lower_cols.get("offstreet_parks")
        g = (
            pd.to_numeric(df[g_col], errors="coerce").fillna(0)
            if g_col is not None
            else 0
        )
        o = (
            pd.to_numeric(df[o_col], errors="coerce").fillna(0)
            if o_col is not None
            else 0
        )
        if isinstance(g, (pd.Series, np.ndarray)) or isinstance(
            o, (pd.Series, np.ndarray)
        ):
            df["Car"] = g + o

    # Numeric type coercions and clamps
    if "Last Rental Price" in df.columns:
        df["Last Rental Price"] = pd.to_numeric(
            df["Last Rental Price"], errors="coerce"
        ).clip(lower=150, upper=2500)
    if "Bed" in df.columns:
        df["Bed"] = pd.to_numeric(df["Bed"], errors="coerce").clip(lower=0, upper=8)
    if "Bath" in df.columns:
        df["Bath"] = pd.to_numeric(df["Bath"], errors="coerce").clip(lower=0, upper=6)
    if "Year Built" in df.columns:
        df["Year Built"] = pd.to_numeric(df["Year Built"], errors="coerce").clip(
            lower=1900, upper=2025
        )

    # Canonical string normalization
    for c in (
        "Suburb",
        "Agency",
        "Agent",
        "Land Use",
        "Development Zone",
        "Property Type",
    ):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()

    return df


def data_cleaning(data):
    """
    Cleans the dataset by:
      - Normalizing suburb strings.
      - Renaming columns to ensure consistent naming of 'sqm'.
      - Parsing numeric columns, including Land/Capital Value columns that have
        $ signs or commas.
      - Converting 'Last Rental Date' to datetime with a chosen format.
      - Excluding properties with bed count > 5.
      - (Optionally) dropping rows missing 'Bed' or leaving them as NaN.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logging.info("Starting data cleaning...")

    try:
        # 0) Canonicalize schema for downstream steps
        data = _normalize_aliases_and_ranges(data)
        try:
            data = assign_geo_queries(data)
            before_missing = None
            if {"Latitude", "Longitude"}.issubset(data.columns):
                mask_missing = data["Latitude"].isna() | data["Longitude"].isna()
                before_missing = int(mask_missing.sum())
            data = enrich_with_geocodes(data)
            if before_missing is not None and {"Latitude", "Longitude"}.issubset(
                data.columns
            ):
                mask_after = data["Latitude"].isna() | data["Longitude"].isna()
                filled = before_missing - int(mask_after.sum())
                if filled > 0:
                    logging.info("Geo enrichment filled %d coordinate rows", filled)
        except Exception as geo_exc:
            logging.warning("Geo enrichment skipped: %s", geo_exc)
        #######################
        # 1. Normalize Suburb
        #######################
        if "Suburb" in data.columns:
            data["Suburb"] = data["Suburb"].str.strip().str.upper()
            logging.info("Normalized 'Suburb' column.")
        else:
            logging.warning("'Suburb' column not found; skipping normalization.")

        #######################
        # 2. Rename columns
        #######################
        # Ensures consistent naming of 'sqm'
        rename_map = {
            "Land Size (m²)": "Land Size (sqm)",
            "Floor Size (m²)": "Floor Size (sqm)",
        }
        data.rename(columns=rename_map, inplace=True)
        logging.info(f"Columns after renaming: {list(data.columns)}")

        ###############################
        # 2b. Handle First-Rental fallback for "Not Disclosed" rows
        ###############################
        if {"First Rental Price", "First Rental Date"}.issubset(data.columns):
            # Normalise helper columns: empty strings -> NaN
            for col in ["First Rental Price", "First Rental Date"]:
                data[col].replace(r"^\s*$", pd.NA, regex=True, inplace=True)

            # Identify rows where last rental price explicitly says "Not Disclosed"
            mask_not_disclosed = (
                data["Last Rental Price"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("not disclosed")
            )

            # First Rental Price must be numeric
            first_price_numeric = pd.to_numeric(
                data["First Rental Price"], errors="coerce"
            )
            mask_first_price_ok = first_price_numeric.notna()

            # First Rental Date must parse to datetime (we'll parse later but require non-empty string here)
            mask_first_date_ok = data["First Rental Date"].notna()

            mask_use_fallback = (
                mask_not_disclosed & mask_first_price_ok & mask_first_date_ok
            )

            logging.info(
                "Fallback stats – not_disclosed=%d, first_price_num=%d, first_date_ok=%d, rows_to_replace=%d",
                mask_not_disclosed.sum(),
                mask_first_price_ok.sum(),
                mask_first_date_ok.sum(),
                mask_use_fallback.sum(),
            )

            if mask_use_fallback.any():
                data.loc[mask_use_fallback, "Last Rental Price"] = first_price_numeric[
                    mask_use_fallback
                ]
                data.loc[mask_use_fallback, "Last Rental Date"] = data.loc[
                    mask_use_fallback, "First Rental Date"
                ]

            # After replacement, drop auxiliary columns
            data.drop(
                columns=["First Rental Price", "First Rental Date"],
                inplace=True,
                errors="ignore",
            )
        else:
            missing_aux = {"First Rental Price", "First Rental Date"} - set(
                data.columns
            )
            logging.info(
                f"Auxiliary first-rental columns not found or incomplete: {missing_aux}. Skipping fallback replacement."
            )

        ###############################
        # 3. Handle 'Capital Value' & 'Land Value'
        ###############################
        # If they contain $ or commas, remove them:
        def clean_currency_column(df, colname):
            if colname in df.columns:
                df[colname] = (
                    df[colname]
                    .astype(str)
                    .replace(r"[^0-9\.\-]+", "", regex=True)  # remove $, commas, etc.
                )
                df[colname] = pd.to_numeric(df[colname], errors="coerce")
                logging.info(f"Cleaned currency column '{colname}'.")
            else:
                logging.warning(
                    f"Column '{colname}' not found; skipping currency cleanup."
                )

        clean_currency_column(data, "Capital Value")
        clean_currency_column(data, "Land Value")

        #######################
        # 4. Numeric columns
        #######################
        # Price parser without auto-annualization: parse numeric value as-is
        def parse_price(val):
            if pd.isnull(val):
                return np.nan
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            val_str = str(val).strip()
            cleaned = re.sub(r"[^0-9\.\-]+", "", val_str)
            if cleaned in ("", "-", "."):
                return np.nan
            try:
                return float(cleaned)
            except ValueError:
                return np.nan

        # Generic helper to coerce mixed-format numeric strings to float
        def parse_numeric(val):
            """Return float value from mixed-format numeric string or NaN."""
            if pd.isnull(val):
                return np.nan
            # Already numeric types remain unchanged
            if isinstance(val, (int, float, np.integer, np.floating)):
                return val
            val_str = str(val)
            # Remove everything except digits, sign and decimal point
            cleaned = re.sub(r"[^0-9\.\-]+", "", val_str)
            # Guard against empty or non-numeric result
            if cleaned in ("", "-", "."):
                return np.nan
            try:
                return float(cleaned)
            except ValueError:
                return np.nan

        # Specialised parser for Bath – extract first integer digit only
        def parse_bath(val):
            if pd.isnull(val):
                return np.nan
            m = re.search(r"(\d+)", str(val))
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return np.nan
            return np.nan

        # Adjust list to match your data layout
        numeric_cols = [
            "Bed",
            "Bath",
            "Car",
            "Land Size (sqm)",
            "Floor Size (sqm)",
            "Year Built",
            "Last Rental Price",
            "Days on Market",
            "Capital Value",
            "Land Value",
        ]

        price_cols = {"Last Rental Price", "First Rental Price"}

        for col in numeric_cols:
            if col not in data.columns:
                logging.warning(f"Numeric column '{col}' not found in the data.")
                continue

            if col == "Bath":
                data[col] = data[col].apply(parse_bath)
                parser_type = "bath-special"
            elif col in price_cols:
                data[col] = data[col].apply(parse_price)
                parser_type = "price"
            else:
                data[col] = data[col].apply(parse_numeric)
                parser_type = "generic"

            logging.info(
                f"Converted '{col}' to numeric ({parser_type} parser). "
                f"NaN count: {data[col].isnull().sum()}"
            )

        #######################
        # 5. Fill Missing Numeric
        #######################
        # For numeric columns other than 'Bed' and 'Last Rental Price', fill with median
        fill_with_median_features = [
            "Land Size (sqm)",
            "Floor Size (sqm)",
            "Year Built",
            "Days on Market",
            "Capital Value",
            "Land Value",
        ]
        for col in fill_with_median_features:
            if col in data.columns:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
                logging.info(f"Filled missing '{col}' with median = {median_val:.2f}.")
            else:
                logging.warning(f"Column '{col}' not found for missing-value fill.")

        # Special rule for 'Car': treat missing or '-' as 0 (do NOT fill with median)
        if "Car" in data.columns:
            before_nulls = data["Car"].isnull().sum()
            data["Car"].fillna(0, inplace=True)
            after_nulls = data["Car"].isnull().sum()
            logging.info(
                f"Car column: replaced {before_nulls - after_nulls} NaNs with 0. Remaining NaNs: {after_nulls}."
            )

        # For 'Last Rental Price', drop rows where it's NaN (after coercion)
        if "Last Rental Price" in data.columns:
            initial_rows = len(data)
            if data["Last Rental Price"].isnull().any():
                logging.warning(
                    f"NaNs found in 'Last Rental Price' before explicit drop: {data['Last Rental Price'].isnull().sum()}."
                )
                data.dropna(subset=["Last Rental Price"], inplace=True)
                rows_dropped = initial_rows - len(data)
                logging.info(
                    f"Dropped {rows_dropped} rows due to NaN in 'Last Rental Price'. New shape: {data.shape}"
                )
            else:
                logging.info("No NaNs found in 'Last Rental Price' to drop.")
        else:
            logging.warning(
                "'Last Rental Price' column not found; cannot drop NaNs for it."
            )

        #######################
        # 6. Handle missing 'Bed'
        #######################
        if "Bed" in data.columns:
            bed_missing = data["Bed"].isnull().sum()
            if bed_missing:
                data["Bed"].fillna(0, inplace=True)
                logging.info(f"Replaced {bed_missing} missing 'Bed' values with 0.")
        else:
            logging.warning("'Bed' column not found; cannot process missing values.")

        #######################
        # 7. Parse 'Last Rental Date'
        #######################
        if "Last Rental Date" in data.columns:
            sample_dates = data["Last Rental Date"].dropna().head(5).tolist()
            logging.info(f"Sample 'Last Rental Date' before parsing: {sample_dates}")

            # Robust multi-format parsing. First attempt: pandas auto-detect with dayfirst.
            data["Last Rental Date"] = pd.to_datetime(
                data["Last Rental Date"], errors="coerce", dayfirst=True
            )

            # Second pass for still-NaT rows using dateutil (handles fuzzy strings like "Mar-23").
            try:
                from dateutil import parser as _date_parser

                mask_nat = data["Last Rental Date"].isna()
                if mask_nat.any():
                    logging.info(
                        f"Attempting dateutil parse for {mask_nat.sum()} unparsed 'Last Rental Date' rows…"
                    )
                    data.loc[mask_nat, "Last Rental Date"] = (
                        data.loc[mask_nat, "Last Rental Date"]
                        .astype(str)
                        .apply(
                            lambda x: _date_parser.parse(x, dayfirst=True, fuzzy=True)
                        )
                    )
            except Exception as dt_exc:
                logging.warning(f"dateutil fallback parsing failed: {dt_exc}")

            # Final check – abort cleaning if any NaT remain, because downstream steps require valid dates.
            nat_cnt = data["Last Rental Date"].isna().sum()
            if nat_cnt:
                before = len(data)
                data = data.dropna(subset=["Last Rental Date"]).reset_index(drop=True)
                logging.warning(
                    "Dropped %d rows with invalid 'Last Rental Date' after parsing (rows before=%d, after=%d)",
                    nat_cnt,
                    before,
                    len(data),
                )

            valid_dates = data["Last Rental Date"].notnull().sum()
            missing_dates = data["Last Rental Date"].isnull().sum()
            logging.info(
                f"After parsing, valid 'Last Rental Date': {valid_dates}, missing: {missing_dates}"
            )

            # If missing still exist, fill with median date or drop rows
            if missing_dates > 0:
                date_median = data["Last Rental Date"].median()
                if pd.isnull(date_median):
                    logging.warning(
                        "Cannot fill missing 'Last Rental Date'; median is NaT."
                    )
                else:
                    data["Last Rental Date"].fillna(date_median, inplace=True)
                    logging.info(
                        f"Filled missing 'Last Rental Date' with median = {date_median}"
                    )
        else:
            logging.warning(
                "'Last Rental Date' column not found in data. Skipping date parsing."
            )

        #######################
        # 8a. Strict domain filters (hard pre-split rules)
        #######################
        def _apply_domain_filters(df: pd.DataFrame) -> pd.DataFrame:
            now_year = pd.Timestamp.now().year
            tmin = float(os.getenv("CLEAN_TARGET_MIN", "0.0"))
            tmax = float(os.getenv("CLEAN_TARGET_MAX", "2000.0"))
            fsmin = float(os.getenv("CLEAN_FLOOR_MIN", "1.0"))
            fsmax = float(os.getenv("CLEAN_FLOOR_MAX", "100000.0"))
            lsmin = float(os.getenv("CLEAN_LAND_MIN", "1.0"))
            lsmax = float(os.getenv("CLEAN_LAND_MAX", "200000.0"))
            bedmax = int(os.getenv("CLEAN_BED_MAX", "15"))
            bathmax = int(os.getenv("CLEAN_BATH_MAX", "15"))
            carmax = int(os.getenv("CLEAN_CAR_MAX", "20"))
            yom_min = int(os.getenv("CLEAN_YEAR_MIN", "1800"))
            yom_max = int(os.getenv("CLEAN_YEAR_MAX", str(now_year + 1)))
            dom_max = int(os.getenv("CLEAN_DOM_MAX", "3650"))
            capmin = float(os.getenv("CLEAN_CAP_MIN", "1000.0"))
            capmax = float(os.getenv("CLEAN_CAP_MAX", "10000000000.0"))

            conds = []
            if "Last Rental Price" in df.columns:
                conds.append(df["Last Rental Price"].between(tmin, tmax))
            if "Floor Size (sqm)" in df.columns:
                conds.append(df["Floor Size (sqm)"].between(fsmin, fsmax))
            if "Land Size (sqm)" in df.columns:
                conds.append(df["Land Size (sqm)"].between(lsmin, lsmax))
            if "Bed" in df.columns:
                conds.append(df["Bed"].between(0, bedmax))
            if "Bath" in df.columns:
                conds.append(df["Bath"].between(0, bathmax))
            if "Car" in df.columns:
                conds.append(df["Car"].between(0, carmax))
            if "Year Built" in df.columns:
                conds.append(df["Year Built"].between(yom_min, yom_max))
            if "Days on Market" in df.columns:
                conds.append(df["Days on Market"].between(0, dom_max))
            if "Capital Value" in df.columns:
                conds.append(df["Capital Value"].between(capmin, capmax))

            mask = pd.Series(True, index=df.index)
            for c in conds:
                mask &= c.fillna(False)

            before = len(df)
            df = df[mask].copy()
            logging.info("Domain filters removed %d rows.", before - len(df))

            # Guard: drop non-positive floor size to avoid invalid PPS
            if "Floor Size (sqm)" in df.columns:
                pre = len(df)
                df = df[df["Floor Size (sqm)"] > 0]
                logging.info(
                    "Removed %d rows with non-positive 'Floor Size (sqm)'.",
                    pre - len(df),
                )

            return df

        data = _apply_domain_filters(data)

        #######################
        # 8. Remove bed > 5
        #######################
        if "Bed" in data.columns:
            prev_shape = data.shape
            # Keep rows where Bed is missing (NaN) or <=5; drop only where Bed >5
            bed_cond = (data["Bed"].isna()) | (data["Bed"] <= 5)
            data = data[bed_cond]
            new_shape = data.shape
            logging.info(
                f"Removed rows with Bed > 5 while retaining unknowns. Shape changed from {prev_shape} to {new_shape}."
            )

        #######################
        # 9. Chronological sort (oldest → newest)
        #######################
        if "Last Rental Date" in data.columns:
            before_order_first = data["Last Rental Date"].iloc[0]
            data = data.sort_values("Last Rental Date").reset_index(drop=True)
            after_order_first = data["Last Rental Date"].iloc[0]
            logging.info(
                "Chronological sort applied on 'Last Rental Date' (%s → %s).",
                before_order_first,
                after_order_first,
            )
        else:
            logging.warning(
                "'Last Rental Date' column missing – skipping chronological sort."
            )

        #######################
        # 10. Final Logging
        #######################
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {list(data.columns)}")

        # Save
        data.to_csv(CLEANED_DATA_PATH, index=False)
        logging.info(f"Cleaned data saved to '{CLEANED_DATA_PATH}'.")

        # Treat missing 'Bath' similar to 'Car' – replace NaN with 0
        if "Bath" in data.columns:
            before_nulls_bath = data["Bath"].isnull().sum()
            data["Bath"].fillna(0, inplace=True)
            # Cast to int for clarity
            data["Bath"] = data["Bath"].astype(int)
            after_nulls_bath = data["Bath"].isnull().sum()
            logging.info(
                f"Bath column: replaced {before_nulls_bath - after_nulls_bath} NaNs with 0. Remaining NaNs: {after_nulls_bath}."
            )
        else:
            logging.warning("'Bath' column not found; cannot process missing values.")

        return data

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        logging.error(traceback.format_exc())
        raise


#
# Additional Note on Predicting 'Bed' for missing rows
#
# If you'd like to *predict* Bed for missing rows (rather than drop them or leave them as NaN),
# you could train a small classifier/regressor using other columns (e.g. Land Size, Floor Size, Year Built, etc.)
# This would be done *after* or as part of data cleaning. The approach might be:
#  1) Separate rows that have a known 'Bed'.
#  2) Train a model to predict 'Bed' from your numeric/categorical features.
#  3) Apply that model to rows missing 'Bed', fill them in, and proceed.
#
# This is more advanced but can sometimes boost overall data coverage and performance.
# It's recommended to validate carefully to ensure you don't introduce too much noise.
