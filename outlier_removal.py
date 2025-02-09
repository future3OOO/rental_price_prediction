import os
import logging
import traceback
import numpy as np
import pandas as pd

def sophisticated_outlier_removal(
    data: pd.DataFrame,
    property_type_column: str = 'Bed',
    numeric_cols: list = None,
    output_dir: str = 'output',
    rental_price_col: str = 'Last Rental Price',
    capital_value_col: str = 'Capital Value'
) -> (pd.DataFrame, int):
    """
    Enhanced outlier removal that:
      1) Picks a main 'target' per row (Rental Price vs. Capital Value) based on which
         has the larger absolute z-score within its property-type group.
      2) Applies supporting numeric features to confirm outlier status.
      3) For bed≥4, checks if the main target's z-score is suspiciously low
         (below a negative threshold) so it can still be flagged even if |z| is
         under the main threshold.
      4) Excludes Bath, Car, *and* Days on Market from suspicious checks, 
         even if they appear in numeric_cols externally.

    *Generates two CSVs in `output_dir`*:
      - `removed_outliers.csv`: details for each removed outlier row
      - `remaining_data_zscores.csv`: z-scores for every row (kept or removed)

    Args:
      data (pd.DataFrame): Input dataset.
      property_type_column (str): Column used to group data (e.g. 'Bed').
      numeric_cols (list): Numeric features for computing z-scores. If None, defaults are used.
      output_dir (str): Folder to save CSV logs.
      rental_price_col (str): Rental price column name.
      capital_value_col (str): Capital value column name.

    Returns:
      (cleaned_df, num_removed): (The cleaned DataFrame, number of outliers removed)
    """

    try:
        ########################################################################
        # 1) Provide default numeric_cols if None was passed
        ########################################################################
        if numeric_cols is None:
            numeric_cols = [
                'Land Size (sqm)',
                'Floor Size (sqm)',
                'Year Built',
                'Capital Value',
                'Time_Index'
            ]

        # Force-exclude 'Days on Market', 'Bath', 'Car' from suspicious checks:
        exclude_set = {'Days on Market', 'Bath', 'Car'}
        numeric_cols = [c for c in numeric_cols if c not in exclude_set]

        # Check required columns
        required_cols = {rental_price_col, capital_value_col, property_type_column} | set(numeric_cols)
        missing_req = required_cols - set(data.columns)
        if missing_req:
            raise ValueError(f"Missing required columns in data: {missing_req}")

        logging.info("Starting sophisticated outlier removal...")

        ########################################################################
        # 2) Key thresholds
        ########################################################################
        MAIN_TARGET_ZSCORE = 2.0           # row is potential outlier if |z|≥2.0 for the main target
        LOW_VALUE_ZSCORE_THRESHOLD = 1.0   # bed≥4 => if main-target z < -1 => suspicious
        SUPPORTING_ZSCORE = 1.0            # supporting feats with |z|≥1 => suspicious
        MIN_SUPPORTING_FEATURES = 2        # how many supporting feats needed
        LUXURY_SKIP_CAP_ZSCORE = 3.5       # if capital-value z‐score is extremely high => skip removal

        ########################################################################
        # 3) Group data by property_type_column => compute modified z-scores
        ########################################################################
        grouped = data.groupby(property_type_column, group_keys=False)

        def modified_zscore(series: pd.Series) -> pd.Series:
            """Compute modified z-score for a single numeric column within a group."""
            med = series.median()
            mad = (series - med).abs().median()
            if pd.isnull(mad) or mad == 0:
                mad = 1e-9
            return 0.6745 * (series - med) / mad

        # Z-scores for main target columns (rent vs. capital):
        z_rent_df = grouped[[rental_price_col]].transform(modified_zscore)
        z_capv_df = grouped[[capital_value_col]].transform(modified_zscore)

        z_rent = z_rent_df[rental_price_col]        # Series of z-scores for rental price
        z_capv = z_capv_df[capital_value_col]       # Series of z-scores for capital value

        # Z-scores for supporting numeric features:
        supp_z_df = grouped[numeric_cols].transform(modified_zscore)

        ########################################################################
        # 4) pick_main_target => row-level deciding if rent or capital is the main
        ########################################################################
        def pick_main_target(ix) -> str:
            zr = abs(z_rent.at[ix]) if ix in z_rent.index else 0
            zc = abs(z_capv.at[ix]) if ix in z_capv.index else 0
            return rental_price_col if zr >= zc else capital_value_col

        def row_is_potential_outlier_main(ix) -> bool:
            """If row's max(|zRent|,|zCapVal|) ≥ MAIN_TARGET_ZSCORE => potential outlier."""
            zr = abs(z_rent.at[ix]) if ix in z_rent.index else 0
            zc = abs(z_capv.at[ix]) if ix in z_capv.index else 0
            return max(zr, zc) >= MAIN_TARGET_ZSCORE

        ########################################################################
        # 5) Potential outliers based on main threshold
        ########################################################################
        potential_idx_main = [i for i in data.index if row_is_potential_outlier_main(i)]

        # bed≥4 => low-value check
        potential_idx_bed4 = []
        for i in data.index:
            bed_val = data.at[i, property_type_column]
            if pd.isnull(bed_val):
                continue
            if bed_val >= 4:
                # main_t
                main_t = pick_main_target(i)
                if i in supp_z_df.index and main_t in supp_z_df.columns:
                    val_z = supp_z_df.at[i, main_t]
                    if pd.notnull(val_z) and val_z < 0 and abs(val_z) >= LOW_VALUE_ZSCORE_THRESHOLD:
                        potential_idx_bed4.append(i)

        # Combine
        potential_idxs = list(set(potential_idx_main + potential_idx_bed4))
        logging.info(f"Potential main outliers: {len(potential_idx_main)}; "
                     f"bed≥4 low outliers: {len(potential_idx_bed4)}; total unique: {len(potential_idxs)}")

        outliers_removed = []
        detail_list = []

        ########################################################################
        # 6) Evaluate each potential row
        ########################################################################
        for ix in potential_idxs:
            # skip-luxury if capital-value zscore is extremely high
            cap_z = abs(z_capv.at[ix]) if ix in z_capv.index else 0
            if cap_z > LUXURY_SKIP_CAP_ZSCORE:
                # skip removal
                continue

            bed_val = data.at[ix, property_type_column]
            main_t = pick_main_target(ix)

            # main zscore
            main_z = supp_z_df.at[ix, main_t] if (ix in supp_z_df.index and main_t in supp_z_df.columns) else 0

            # Condition #1 => abs(main_z)≥ MAIN_TARGET_ZSCORE
            # Condition #2 => bed≥4 negative outlier
            cond1 = (abs(main_z) >= MAIN_TARGET_ZSCORE)
            cond2 = False
            if pd.notnull(bed_val) and bed_val >= 4 and pd.notnull(main_z):
                if main_z < 0 and abs(main_z) >= LOW_VALUE_ZSCORE_THRESHOLD:
                    cond2 = True

            # if neither condition => skip
            if not (cond1 or cond2):
                continue

            # The “other target” is also a supporting feature
            alt_target = capital_value_col if main_t == rental_price_col else rental_price_col

            suspicious_count = 0
            suspicious_feats = []

            # alt target check:
            if alt_target in supp_z_df.columns:
                alt_z = supp_z_df.at[ix, alt_target]
                if pd.notnull(alt_z) and abs(alt_z) >= SUPPORTING_ZSCORE:
                    suspicious_count += 1
                    suspicious_feats.append(f"{alt_target}[z={alt_z:.2f}]")

            # other numeric feats
            for feat in numeric_cols:
                if feat not in supp_z_df.columns:
                    continue
                if feat in [main_t, alt_target]:
                    continue
                fz = supp_z_df.at[ix, feat]
                if pd.notnull(fz) and abs(fz) >= SUPPORTING_ZSCORE:
                    suspicious_count += 1
                    suspicious_feats.append(f"{feat}[z={fz:.2f}]")

            # if bed≥4 negative => maybe allow fewer supporting feats
            needed = MIN_SUPPORTING_FEATURES
            if cond2:  # bed≥4 negative outlier
                needed = 1

            if suspicious_count >= needed:
                outliers_removed.append(ix)
                row_info = {
                    'Index': ix,
                    property_type_column: bed_val,
                    'mainTarget': main_t,
                    'mainZ': main_z,
                    'zCapVal': z_capv.at[ix] if ix in z_capv.index else np.nan,
                    'zRent': z_rent.at[ix] if ix in z_rent.index else np.nan,
                    'SuspiciousFeats': ";".join(suspicious_feats)
                }
                for feat in numeric_cols:
                    if feat in data.columns:
                        row_info[f"{feat}_val"] = data.at[ix, feat]
                    if feat in supp_z_df.columns:
                        row_info[f"{feat}_z"] = supp_z_df.at[ix, feat]

                detail_list.append(row_info)

        cleaned_df = data.drop(index=outliers_removed)
        num_removed = len(outliers_removed)
        logging.info(f"Removed {num_removed} outliers. Cleaned shape: {cleaned_df.shape}")

        ########################################################################
        # 7) CSV logs
        ########################################################################
        os.makedirs(output_dir, exist_ok=True)

        if num_removed > 0:
            rm_df = pd.DataFrame(detail_list)
            removed_csv = os.path.join(output_dir, "removed_outliers.csv")
            rm_df.to_csv(removed_csv, index=False)
            logging.info(f"Removed outlier details => {removed_csv}")

        # Dump z-scores for all rows
        zscore_entries = []
        for i in data.index:
            row_d = {
                'Index': i,
                property_type_column: data.at[i, property_type_column] if property_type_column in data.columns else np.nan,
                'zRent': z_rent.at[i] if i in z_rent.index else np.nan,
                'zCapVal': z_capv.at[i] if i in z_capv.index else np.nan,
                'isRemoved': (i in outliers_removed),
            }
            for feat in numeric_cols:
                row_d[f"{feat}_val"] = data.at[i, feat] if feat in data.columns else np.nan
                if feat in supp_z_df.columns:
                    row_d[f"{feat}_z"] = supp_z_df.at[i, feat]
            zscore_entries.append(row_d)

        out_zscore_path = os.path.join(output_dir, "remaining_data_zscores.csv")
        pd.DataFrame(zscore_entries).to_csv(out_zscore_path, index=False)
        logging.info(f"remaining_data_zscores => {out_zscore_path}")

        return cleaned_df, num_removed

    except Exception as e:
        logging.error(f"Error in sophisticated_outlier_removal: {str(e)}")
        logging.error(traceback.format_exc())
        raise
