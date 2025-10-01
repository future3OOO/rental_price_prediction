# data_preparation.py

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from scipy import stats
import traceback

# Retain your original plotting imports:
from plotting import (
    plot_vwap_interactive,
    plot_global_vwap_interactive,
    plot_global_median_interactive,
    plot_eq_interactive_beds_1_and_2,
    plot_global_vwap_interactive_eq
)

# ------------------------------------------------------------------
# Global VWAP plotting flag (shared with plotting.py). Default False.
# ------------------------------------------------------------------
ENABLE_VWAP_PLOTS: bool = os.getenv("ENABLE_VWAP_PLOTS", "False").strip().lower() in {"1", "true", "yes"}

def _vwap_noop(*args, **kwargs):
    if not ENABLE_VWAP_PLOTS:
        logging.info("VWAP plots disabled via ENABLE_VWAP_PLOTS – skipping plot generation.")

if not ENABLE_VWAP_PLOTS:
    # Replace local VWAP plotting helpers with no-ops
    def plot_vwap(*args, **kwargs):
        _vwap_noop()

    def plot_global_vwap(*args, **kwargs):
        _vwap_noop()

    # Any interactive wrappers imported from plotting are already stubbed.

def remove_outliers(group):
    """
    Example outlier removal function (if you need it).
    Adjust the threshold or remove entirely if not required.
    """
    z_scores = np.abs(stats.zscore(group['Last Rental Price']))
    return group[z_scores < 4]

def plot_moving_medians(data, bed_type, plot_dir):
    """
    Plots rolling medians (3M and 12M) for a given bed type.
    Mirrors your original logic.
    """
    try:
        bed_data = data[data['Bed'] == bed_type].copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(
            bed_data['Last Rental Date'],
            bed_data['Rolling_Median_3M'],
            label='3-Month Moving Median'
        )
        plt.plot(
            bed_data['Last Rental Date'],
            bed_data['Rolling_Median_2Y_bed'],
            label='2-Year Moving Median'
        )

        plt.scatter(
            bed_data['Last Rental Date'],
            bed_data['Last Rental Price'],
            label='Actual Price', alpha=0.1, s=5, color='gray'
        )

        plt.title(f'Moving Medians for {bed_type}-Bedroom Properties')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f'moving_medians_{bed_type}bed.png'
        path = os.path.join(plot_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        logging.info(f"Moving medians plot saved: {path}")

    except Exception as e:
        logging.error(f"Error in plot_moving_medians for bed type {bed_type}: {str(e)}")
        raise

def plot_vwap(data, bed_type, plot_dir):
    """
    Plots VWAP (3M, 12M) for a given bed type in a standard Matplotlib chart.
    Prioritizes EQUAL-WEIGHTED VWAP, shows dollar-weighted as secondary if available.
    """
    try:
        bed_data = data[data['Bed'] == bed_type].copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))

        # Equal-weighted lines (Primary)
        if 'VWAP_3M_eq' in bed_data.columns and 'VWAP_2Y_eq' in bed_data.columns:
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_3M_eq'],
                label='3M VWAP (Equal-Weighted)'
            )
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_2Y_eq'],
                label='2Y VWAP (Equal-Weighted)'
            )
            title_suffix = '(Equal-Weighted Priority)'
        else:
            logging.warning(f"Equal-weighted VWAP columns (VWAP_3M_eq, VWAP_2Y_eq) not found for bed type {bed_type}. Cannot plot primary lines.")
            title_suffix = '(Equal-Weighted Not Found)'

        # Dollar-weighted lines (Secondary, if available)
        if 'VWAP_3M' in bed_data.columns and 'VWAP_2Y' in bed_data.columns:
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_3M'],
                label='3M VWAP (Dollar-Weighted)',
                linestyle='--', alpha=0.7
            )
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_2Y'],
                label='2Y VWAP (Dollar-Weighted)',
                linestyle='--', alpha=0.7
            )

        # 1-Year Exponential Moving Average (equal-weighted) trend for additional smoothing
        try:
            daily_eq = (
                bed_data.set_index('Last Rental Date')['Last Rental Price']
                .resample('D').mean()
            )
            ema_1y = daily_eq.ewm(span=365, adjust=False, min_periods=30).mean()
            plt.plot(ema_1y.index, ema_1y.values, label='EMA 1Y (Equal-Weighted)', color='green', linewidth=2.2)
        except Exception as ema_err:
            logging.warning(f"Could not compute EMA_1Y for bed={bed_type}: {ema_err}")

        plt.scatter(
            bed_data['Last Rental Date'],
            bed_data['Last Rental Price'],
            label='Actual Price', alpha=0.1, s=5, color='gray'
        )

        plt.title(f'VWAP for {bed_type}-Bedroom Properties {title_suffix}')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f'vwap_{bed_type}bed.png'
        path = os.path.join(plot_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        logging.info(f"VWAP plot saved: {path}")

    except Exception as e:
        logging.error(f"Error in plot_vwap for bed type {bed_type}: {str(e)}")
        raise

def plot_global_vwap(data, plot_dir):
    """
    Plots the global (all-bed) VWAP lines (3M/12M) plus actual prices.
    Prioritizes EQUAL-WEIGHTED VWAP, shows dollar-weighted as secondary if available.
    """
    try:
        global_data = data.copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))

        # Equal-weighted global lines (Primary)
        if 'VWAP_3M_all_eq' in global_data.columns and 'VWAP_2Y_all_eq' in global_data.columns:
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_3M_all_eq'],
                label='3M VWAP (All Beds, Equal-Weighted)'
            )
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_2Y_all_eq'],
                label='2Y VWAP (All Beds, Equal-Weighted)'
            )
            title_suffix = '(Equal-Weighted Priority)'
        else:
            logging.warning("Global equal-weighted VWAP columns (VWAP_3M_all_eq, VWAP_2Y_all_eq) not found. Cannot plot primary lines.")
            title_suffix = '(Equal-Weighted Not Found)'

        # Dollar-weighted global lines (Secondary, if available)
        if 'VWAP_3M_all' in global_data.columns and 'VWAP_2Y_all' in global_data.columns:
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_3M_all'],
                label='3M VWAP (All Beds, Dollar-Weighted)',
                linestyle='--', alpha=0.7
            )
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_2Y_all'],
                label='2Y VWAP (All Beds, Dollar-Weighted)',
                linestyle='--', alpha=0.7
            )

        # 1-Year Exponential Moving Average (equal-weighted) trend for global data
        try:
            global_daily_eq = (
                global_data.set_index('Last Rental Date')['Last Rental Price']
                .resample('D').mean()
            )
            ema_global = global_daily_eq.ewm(span=365, adjust=False, min_periods=30).mean()
            plt.plot(ema_global.index, ema_global.values, label='EMA 1Y (Eq All)', color='green', linewidth=2.2)
        except Exception as ema_err:
            logging.warning(f"Could not compute global EMA_1Y: {ema_err}")

        plt.scatter(
            global_data['Last Rental Date'],
            global_data['Last Rental Price'],
            label='Actual Price', alpha=0.1, s=5, color='gray'
        )

        plt.title(f'VWAP for All-Bedroom Properties Combined {title_suffix}')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = 'vwap_all_beds.png'
        path = os.path.join(plot_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        logging.info(f"Global VWAP plot saved: {path}")

    except Exception as e:
        logging.error(f"Error in plot_global_vwap: {str(e)}")
        raise

def prepare_data(data, plot_dir):
    """
    Prepares data by:
      1. Sorting by date,
      2. Calculating both DOLLAR-weighted and EQUAL-weighted VWAP for each bed type (3M/12M),
      3. Calculating global (all-beds) VWAP in both styles,
      4. Calculating percentage differences,
      5. Plotting bed-specific and global VWAP,
      6. Logging raw VWAP columns to CSV for verification,
      7. Keeping 1-bed & 5+ bed data if it meets minimal rolling-window criteria,
      8. Creates an additional combined interactive eq plot for beds=1 & 2 (single chart).
    """
    try:
        logging.info("Starting data preparation...")

        # If VWAP plots are disabled globally, exit early to avoid heavy computations
        if not ENABLE_VWAP_PLOTS:
            logging.info("ENABLE_VWAP_PLOTS is False – skipping VWAP calculations & plots in prepare_data().")
            return data

        # ------------------------------------------------------------------
        # 0) Ensure the incoming dataframe has no duplicated columns.  If the
        #    script is rerun in the same session the global variable `data`
        #    may already contain VWAP columns; duplicated columns silently
        #    break downstream logic and CSV exports.  We proactively collapse
        #    to the first occurrence, log what happened, and guarantee column
        #    uniqueness for the remainder of the function.
        # ------------------------------------------------------------------
        if data.columns.duplicated().any():
            dupes = data.columns[data.columns.duplicated()].unique().tolist()
            logging.warning(
                "Incoming DataFrame had duplicated columns – keeping first occurrence and dropping %d duplicates: %s",
                len(dupes), dupes
            )
            data = data.loc[:, ~data.columns.duplicated()].copy()

        data = data.copy() # Work on a copy

        # Ensure 'Last Rental Date' is datetime, especially if cleaning is bypassed
        if 'Last Rental Date' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['Last Rental Date']):
                logging.info("Coercing 'Last Rental Date' to datetime in prepare_data as it's not already datetime type.")
                original_nulls = data['Last Rental Date'].isnull().sum()
                data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'], format='%d-%b-%y', errors='coerce')
                coerced_nulls = data['Last Rental Date'].isnull().sum()
                newly_null = coerced_nulls - original_nulls
                if newly_null > 0:
                    logging.warning(f"{newly_null} new NaT values in 'Last Rental Date' after coercion.")
                
                if data['Last Rental Date'].isnull().any():
                    median_date = data['Last Rental Date'].median()
                    if pd.notnull(median_date):
                        data['Last Rental Date'].fillna(median_date, inplace=True)
                        logging.info(f"Filled NaT values in 'Last Rental Date' with median date: {median_date}.")
                    else:
                        logging.warning("Could not fill NaT in 'Last Rental Date' as median date is also NaT. Rows with NaT might be dropped or cause issues.")
            
            # Sort by date early, after ensuring it's datetime
            data.sort_values('Last Rental Date', inplace=True)
            logging.info("'Last Rental Date' is datetime. Data sorted by date.")
        else:
            logging.error("'Last Rental Date' column not found. Cannot proceed with time-series operations.")
            raise ValueError("'Last Rental Date' column is required for data preparation.")

        # Ensure 'Last Rental Price' is numeric, coercing errors, 
        # especially if data_cleaning was bypassed.
        if 'Last Rental Price' in data.columns and data['Last Rental Price'].dtype == 'object':
            logging.info("Coercing 'Last Rental Price' to numeric in prepare_data as it appears to be non-numeric.")
            # First, remove common non-numeric characters like '$', ',', then convert
            # A more robust regex might be needed if other characters like '-' are common and not part of valid numbers
            data['Last Rental Price'] = (
                data['Last Rental Price']
                .astype(str)
                .str.replace(r'[$,]', '', regex=True) # Remove $ and commas
            )
            # Attempt to convert to numeric, coercing actual problematic strings (like '-') to NaN
            data['Last Rental Price'] = pd.to_numeric(data['Last Rental Price'], errors='coerce')
            # It's good practice to fill NaNs created by coercion if subsequent code assumes no NaNs.
            # For 'Last Rental Price', the data_cleaning script fills with median. 
            # If bypassed, we might need a strategy here or ensure downstream handles it.
            # For now, let's assume subsequent rolling calculations can handle NaNs or they get dropped.
            if data['Last Rental Price'].isnull().any():
                logging.warning(f"NaNs introduced in 'Last Rental Price' during coercion in prepare_data: {data['Last Rental Price'].isnull().sum()}")

        # Ensure 'Bed' column is numeric if cleaning was bypassed, coercing errors
        if 'Bed' in data.columns and data['Bed'].dtype == 'object':
            logging.info("Coercing 'Bed' to numeric in prepare_data as it appears to be non-numeric.")
            data['Bed'] = pd.to_numeric(data['Bed'], errors='coerce')
            if data['Bed'].isnull().any():
                logging.warning(f"NaNs introduced in 'Bed' during coercion in prepare_data: {data['Bed'].isnull().sum()}")

        # 1) Sort data by date
        data = data.copy()
        data.sort_values('Last Rental Date', inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Columns for weighting
        data['Dollar_Volume'] = data['Last Rental Price'].astype(float)
        data['Volume'] = np.nan
        data['Volume_eq'] = 1.0

        bed_values_raw = data['Bed'].unique()
        # Filter out NaN bed_values that may have resulted from coercion if cleaning was bypassed
        bed_values = [b for b in bed_values_raw if pd.notnull(b)]
        logging.info(f"Calculating VWAP for VALID bed types: {bed_values}")

        # Rolling constraints (matching your original logic)
        THREE_MONTH_MIN = 3
        TWELVE_MONTH_MIN = 1

        # 2) Per-bed computations
        for b_type in bed_values:
            subset = data[data['Bed'] == b_type].copy()
            subset.set_index('Last Rental Date', inplace=True)
            subset.sort_index(inplace=True)

            # (A) DOLLAR-weighted
            subset['Volume'] = subset['Dollar_Volume']
            subset['Rolling_PV_3M'] = (subset['Last Rental Price'] * subset['Volume']).rolling(
                window='90D', min_periods=THREE_MONTH_MIN).sum()
            subset['Rolling_V_3M'] = subset['Volume'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

            subset['Rolling_PV_2Y'] = (subset['Last Rental Price'] * subset['Volume']).rolling(
                window='730D', min_periods=TWELVE_MONTH_MIN).sum()
            subset['Rolling_V_2Y'] = subset['Volume'].rolling(window='730D', min_periods=TWELVE_MONTH_MIN).sum()

            subset['VWAP_3M'] = (subset['Rolling_PV_3M'] / subset['Rolling_V_3M']).shift(1)
            subset['VWAP_2Y'] = (subset['Rolling_PV_2Y'] / subset['Rolling_V_2Y']).shift(1)

            # (B) EQUAL-weighted
            subset['Volume_eq'] = 1.0
            subset['Rolling_PV_3M_eq'] = (subset['Last Rental Price'] * subset['Volume_eq']).rolling(
                window='90D', min_periods=THREE_MONTH_MIN).sum()
            subset['Rolling_V_3M_eq'] = subset['Volume_eq'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

            subset['Rolling_PV_2Y_eq'] = (subset['Last Rental Price'] * subset['Volume_eq']).rolling(
                window='730D', min_periods=TWELVE_MONTH_MIN).sum()
            subset['Rolling_V_2Y_eq'] = subset['Volume_eq'].rolling(window='730D', min_periods=TWELVE_MONTH_MIN).sum()

            subset['VWAP_3M_eq'] = (subset['Rolling_PV_3M_eq'] / subset['Rolling_V_3M_eq']).shift(1)
            subset['VWAP_2Y_eq'] = (subset['Rolling_PV_2Y_eq'] / subset['Rolling_V_2Y_eq']).shift(1)

            # (C) Per-bed Rolling Medians (New)
            # Calculate Rolling Medians for the current bed type subset
            subset['Rolling_Median_3M_bed'] = subset['Last Rental Price'].rolling(
                window='90D', min_periods=THREE_MONTH_MIN
            ).median().shift(1) # Shift to prevent data leakage
            subset['Rolling_Median_2Y_bed'] = subset['Last Rental Price'].rolling(
                window='730D', min_periods=TWELVE_MONTH_MIN
            ).median().shift(1) # Shift to prevent data leakage

            # Move results back
            mask = (data['Bed'] == b_type)
            data.loc[mask, 'Volume'] = subset['Volume'].values
            data.loc[mask, 'VWAP_3M'] = subset['VWAP_3M'].values
            data.loc[mask, 'VWAP_2Y'] = subset['VWAP_2Y'].values
            data.loc[mask, 'Volume_eq'] = subset['Volume_eq'].values
            data.loc[mask, 'VWAP_3M_eq'] = subset['VWAP_3M_eq'].values
            data.loc[mask, 'VWAP_2Y_eq'] = subset['VWAP_2Y_eq'].values
            data.loc[mask, 'Rolling_Median_3M_bed'] = subset['Rolling_Median_3M_bed'].values
            data.loc[mask, 'Rolling_Median_2Y_bed'] = subset['Rolling_Median_2Y_bed'].values

        # 3) Global Weighted (Dollar + Eq)
        data.set_index('Last Rental Date', inplace=True)
        data.sort_index(inplace=True)

        # (A) $ Weighted
        data['Volume_all'] = data['Dollar_Volume']
        data['Rolling_PV_3M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='90D', min_periods=THREE_MONTH_MIN).sum()
        data['Rolling_V_3M_all'] = data['Volume_all'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

        data['Rolling_PV_2Y_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='730D', min_periods=TWELVE_MONTH_MIN).sum()
        data['Rolling_V_2Y_all'] = data['Volume_all'].rolling(window='730D', min_periods=TWELVE_MONTH_MIN).sum()

        data['VWAP_3M_all'] = (data['Rolling_PV_3M_all'] / data['Rolling_V_3M_all']).shift(1)
        data['VWAP_2Y_all'] = (data['Rolling_PV_2Y_all'] / data['Rolling_V_2Y_all']).shift(1)

        # (B) EQ Weighted
        data['Volume_all_eq'] = 1.0
        data['Rolling_PV_3M_all_eq'] = (data['Last Rental Price'] * data['Volume_all_eq']).rolling(
            window='90D', min_periods=THREE_MONTH_MIN).sum()
        data['Rolling_V_3M_all_eq'] = data['Volume_all_eq'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

        data['Rolling_PV_2Y_all_eq'] = (data['Last Rental Price'] * data['Volume_all_eq']).rolling(
            window='730D', min_periods=TWELVE_MONTH_MIN).sum()
        data['Rolling_V_2Y_all_eq'] = data['Volume_all_eq'].rolling(window='730D', min_periods=TWELVE_MONTH_MIN).sum()

        data['VWAP_3M_all_eq'] = (data['Rolling_PV_3M_all_eq'] / data['Rolling_V_3M_all_eq']).shift(1)
        data['VWAP_2Y_all_eq'] = (data['Rolling_PV_2Y_all_eq'] / data['Rolling_V_2Y_all_eq']).shift(1)

        # Calculate Rolling Median of VWAP_3M_all_eq
        if 'VWAP_3M_all_eq' in data.columns:
            data['RollMed_VWAP_3M_all_eq_30D'] = data['VWAP_3M_all_eq'].rolling(window='30D', min_periods=1).median()
            logging.info("Calculated 30-day rolling median of VWAP_3M_all_eq (shifted by base VWAP only).")
        else:
            logging.warning("VWAP_3M_all_eq column not found, cannot calculate its 30-day rolling median.")
            data['RollMed_VWAP_3M_all_eq_30D'] = np.nan

        # Calculate Rolling Median of VWAP_2Y_all_eq (730-day)
        if 'VWAP_2Y_all_eq' in data.columns:
            data['RollMed_VWAP_2Y_all_eq_730D'] = data['VWAP_2Y_all_eq'].rolling(window='730D', min_periods=1).median()
            logging.info("Calculated 730-day rolling median of VWAP_2Y_all_eq (shifted by base VWAP only).")
        else:
            logging.warning("VWAP_2Y_all_eq column not found, cannot calculate its 730-day rolling median.")
            data['RollMed_VWAP_2Y_all_eq_730D'] = np.nan

        data.reset_index(inplace=True)

        # ------------------------------------------------------------------
        # End-of-function integrity check – assert no duplicate columns remain.
        # If duplicates slipped through they indicate a bug in the feature
        # engineering logic; fail fast so production never sees corrupted
        # schema.
        # ------------------------------------------------------------------
        if data.columns.duplicated().any():
            dupes_final = data.columns[data.columns.duplicated()].unique().tolist()
            raise RuntimeError(
                f"prepare_data produced duplicated columns: {dupes_final}. "
                "This is a logic error and must be fixed before continuing."
            )

        # Log NaN counts before the main VWAP dropna
        logging.info(f"Shape before critical VWAP dropna: {data.shape}")
        vwap_cols_to_check = ['VWAP_3M_all', 'VWAP_2Y_all', 'VWAP_3M_all_eq', 'VWAP_2Y_all_eq',
                              'VWAP_3M', 'VWAP_2Y', 'VWAP_3M_eq', 'VWAP_2Y_eq']
        for col in vwap_cols_to_check:
            if col in data.columns:
                # Check for inf as well, as dropna might not handle them unless they are first converted to NaN
                is_inf_sum = np.isinf(data[col]).sum()
                is_nan_sum = data[col].isnull().sum()
                logging.info(f"NaNs in {col} BEFORE dropna: {is_nan_sum}, Infs in {col} BEFORE dropna: {is_inf_sum}")
            else:
                logging.warning(f"Column {col} not found BEFORE dropna - this might be expected for per-bed if only one bed type exists.")

        # 4) Drop missing VWAP columns
        cols_for_dropna = [
            'VWAP_3M','VWAP_2Y','VWAP_3M_eq','VWAP_2Y_eq',
            'VWAP_3M_all','VWAP_2Y_all','VWAP_3M_all_eq','VWAP_2Y_all_eq',
            'Rolling_Median_3M_bed', 'Rolling_Median_2Y_bed',
            'RollMed_VWAP_3M_all_eq_30D',
            'RollMed_VWAP_2Y_all_eq_730D'
        ]
        # Ensure only existing columns are used in dropna subset to avoid errors if a per-bed VWAP column wasn't created (e.g. only 1 bed type)
        actual_cols_for_dropna = [col for col in cols_for_dropna if col in data.columns]
        
        # Replace inf with NaN before dropping, to be thorough
        if actual_cols_for_dropna: # only proceed if there are columns to process
            data[actual_cols_for_dropna] = data[actual_cols_for_dropna].replace([np.inf, -np.inf], np.nan)
            logging.info(f"Replaced inf with NaN in {actual_cols_for_dropna} before dropna.")

        data.dropna(subset=actual_cols_for_dropna, inplace=True)
        logging.info(f"Shape AFTER critical VWAP dropna: {data.shape}")

        # Log NaN counts after the main VWAP dropna
        for col in vwap_cols_to_check: # Check the same broader list
            if col in data.columns:
                is_inf_sum = np.isinf(data[col]).sum()
                is_nan_sum = data[col].isnull().sum()
                logging.info(f"NaNs in {col} AFTER dropna: {is_nan_sum}, Infs in {col} AFTER dropna: {is_inf_sum}")
            else:
                # This might happen if a column was all NaN (or inf converted to NaN) and then effectively dropped or not present 
                # if it wasn't in actual_cols_for_dropna due to not existing initially.
                logging.warning(f"Column {col} may not be present or fully populated AFTER dropna.")

        # 5) Calculate percentage diffs
        data['Percentage_Diff'] = ((data['VWAP_3M'] - data['VWAP_2Y']) / data['VWAP_2Y']) * 100
        data['Percentage_Diff'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Percentage_Diff'].fillna(0, inplace=True)

        data['Percentage_Diff_eq'] = ((data['VWAP_3M_eq'] - data['VWAP_2Y_eq']) / data['VWAP_2Y_eq']) * 100
        data['Percentage_Diff_eq'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Percentage_Diff_eq'].fillna(0, inplace=True)

        data['Percentage_Diff_all'] = ((data['VWAP_3M_all'] - data['VWAP_2Y_all']) / data['VWAP_2Y_all']) * 100
        data['Percentage_Diff_all'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Percentage_Diff_all'].fillna(0, inplace=True)

        data['Percentage_Diff_all_eq'] = ((data['VWAP_3M_all_eq'] - data['VWAP_2Y_all_eq']) / data['VWAP_2Y_all_eq']) * 100
        data['Percentage_Diff_all_eq'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Percentage_Diff_all_eq'].fillna(0, inplace=True)

        # 6) Plot bed-specific VWAP
        bed_values = data['Bed'].unique()
        for b_type in bed_values:
            sub_b = data[data['Bed'] == b_type].copy()
            if sub_b.empty:
                logging.warning(f"No data for bed={b_type} after VWAP calculation; skipping plots.")
                continue

            # Standard VWAP plot
            logging.info(f"Plotting standard VWAP for {b_type}-bed ...")
            plot_vwap(sub_b, b_type, plot_dir)

            # Interactive VWAP (dollar-based)
            logging.info(f"Plotting interactive VWAP (dollar-based) for {b_type}-bed ...")
            plot_vwap_interactive(sub_b, b_type, plot_dir)

        # 7) Global VWAP
        logging.info("Plotting global VWAP (both $ and eq if present)...")
        plot_global_vwap(data, plot_dir)

        # 8) Combined eq plot for bed=1 & bed=2
        logging.info("Plotting combined eq interactive for bed=1 & 2 ...")
        plot_eq_interactive_beds_1_and_2(data, plot_dir)

        # 9) Rolling medians for all-beds
        data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'])
        data.sort_values('Last Rental Date', inplace=True)
        data.set_index('Last Rental Date', inplace=True)

        # Align min_periods to 1 for both 1M and 12M global medians for maximum data retention
        data['Rolling_Median_1M_all'] = data['Last Rental Price'].rolling(window='30D', min_periods=1).median().shift(1)
        data['Rolling_Median_2Y_all'] = data['Last Rental Price'].rolling(window='730D', min_periods=1).median().shift(1)
        logging.info("Calculated SHIFTED Rolling_Median_1M_all and Rolling_Median_2Y_all.")

        data.dropna(subset=['Rolling_Median_1M_all','Rolling_Median_2Y_all'], inplace=True)
        data.reset_index(inplace=True)

        # Log data state before plot_global_median_interactive (for rolling_median_all_beds_interactive context)
        logging.info(f"Data shape just BEFORE plot_global_median_interactive: {data.shape}")
        if 'Rolling_Median_1M_all' in data.columns:
            logging.info(f"NaNs in Rolling_Median_1M_all for global_median_interactive: {data['Rolling_Median_1M_all'].isnull().sum()}, Infs: {np.isinf(data['Rolling_Median_1M_all']).sum()}")
        if 'Rolling_Median_2Y_all' in data.columns:
            logging.info(f"NaNs in Rolling_Median_2Y_all for global_median_interactive: {data['Rolling_Median_2Y_all'].isnull().sum()}, Infs: {np.isinf(data['Rolling_Median_2Y_all']).sum()}")

        logging.info("Plotting interactive global rolling median (explicit call)...")
        plot_global_median_interactive(data, plot_dir)

        # NOW CALL THE WRAPPER FUNCTION plot_global_vwap_interactive
        # This should prioritize global medians if available, otherwise fallback.
        logging.info("Calling plot_global_vwap_interactive (should prioritize medians if available)...")
        plot_global_vwap_interactive(data, plot_dir) 

        # Explicitly call plot_global_vwap_interactive_eq to ensure vwap_all_beds_eq_only.html is updated
        # with the new RollMed_VWAP_3M_all_eq_30D line.
        logging.info("Explicitly calling plot_global_vwap_interactive_eq to update vwap_all_beds_eq_only.html...")
        plot_global_vwap_interactive_eq(data, plot_dir) # Make sure this is imported in plotting.py if not already

        # 10) Dump columns for verification
        verify_cols = [
            'Last Rental Date','Bed','Last Rental Price',
            'Dollar_Volume','Volume','VWAP_3M','VWAP_2Y',
            'Volume_eq','VWAP_3M_eq','VWAP_2Y_eq',
            'Rolling_Median_3M_bed', 'Rolling_Median_2Y_bed',
            'Volume_all','VWAP_3M_all','VWAP_2Y_all',
            'Volume_all_eq','VWAP_3M_all_eq','VWAP_2Y_all_eq',
            'Percentage_Diff','Percentage_Diff_eq',
            'Percentage_Diff_all','Percentage_Diff_all_eq',
            'RollMed_VWAP_3M_all_eq_30D',
            'RollMed_VWAP_2Y_all_eq_730D'
        ]
        log_df = data.copy()
        missing_cols = [c for c in verify_cols if c not in log_df.columns]
        for c in missing_cols:
            log_df[c] = np.nan

        raw_csv = os.path.join(plot_dir, 'vwap_raw_verification.csv')
        log_df[verify_cols].to_csv(raw_csv, index=False)
        logging.info(f"Dumped raw VWAP columns => {raw_csv}")

        logging.info("Data preparation completed successfully.")
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {data.columns.tolist()}")
        return data

    except Exception as e:
        logging.error(f"Error in prepare_data: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# ------------------------------------------------------------------
# Final safeguard: ensure that even if earlier definitions re-enabled the
# static Matplotlib VWAP helpers, they are no-ops when the flag is off.
# ------------------------------------------------------------------
if not ENABLE_VWAP_PLOTS:
    plot_vwap = _vwap_noop  # type: ignore
    plot_global_vwap = _vwap_noop  # type: ignore
