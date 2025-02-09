import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from scipy import stats
import traceback

from plotting import (
    plot_vwap_interactive,
    plot_global_vwap_interactive,
    plot_global_median_interactive,
    plot_eq_interactive_beds_1_and_2  # <-- Import your new combined function here
)

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
            bed_data['Rolling_Median_12M'],
            label='12-Month Moving Median'
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
    Plots *dollar-weighted* VWAP (3M, 12M) for a given bed type in a standard Matplotlib chart.
    (If you want to also plot the equal-weighted lines, you can do so similarly.)
    """
    try:
        bed_data = data[data['Bed'] == bed_type].copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))

        # Dollar-weighted lines
        plt.plot(
            bed_data['Last Rental Date'],
            bed_data['VWAP_3M'],
            label='3M VWAP (Dollar-Weighted)'
        )
        plt.plot(
            bed_data['Last Rental Date'],
            bed_data['VWAP_12M'],
            label='12M VWAP (Dollar-Weighted)'
        )

        # Optionally, plot equal-weighted lines if they exist
        if 'VWAP_3M_eq' in bed_data.columns and 'VWAP_12M_eq' in bed_data.columns:
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_3M_eq'],
                label='3M VWAP (Equal-Weighted)',
                linestyle='--'
            )
            plt.plot(
                bed_data['Last Rental Date'],
                bed_data['VWAP_12M_eq'],
                label='12M VWAP (Equal-Weighted)',
                linestyle='--'
            )

        plt.scatter(
            bed_data['Last Rental Date'],
            bed_data['Last Rental Price'],
            label='Actual Price', alpha=0.1, s=5, color='gray'
        )

        plt.title(f'VWAP for {bed_type}-Bedroom Properties')
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
    Plots the *dollar-weighted* global (all-bed) VWAP lines (3M/12M) plus actual prices.
    Same approach if you want to also show the equal-weighted lines.
    """
    try:
        global_data = data.copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(
            global_data['Last Rental Date'],
            global_data['VWAP_3M_all'],
            label='3M VWAP (All Beds, $-Weighted)'
        )
        plt.plot(
            global_data['Last Rental Date'],
            global_data['VWAP_12M_all'],
            label='12M VWAP (All Beds, $-Weighted)'
        )

        # If equal-weighted lines exist, plot them in a different style
        if 'VWAP_3M_all_eq' in global_data.columns and 'VWAP_12M_all_eq' in global_data.columns:
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_3M_all_eq'],
                label='3M VWAP (All Beds, Eq)',
                linestyle='--'
            )
            plt.plot(
                global_data['Last Rental Date'],
                global_data['VWAP_12M_all_eq'],
                label='12M VWAP (All Beds, Eq)',
                linestyle='--'
            )

        plt.scatter(
            global_data['Last Rental Date'],
            global_data['Last Rental Price'],
            label='Actual Price', alpha=0.1, s=5, color='gray'
        )

        plt.title('VWAP for All-Bedroom Properties Combined')
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
      7. Keeping 1-bed & 5+ bed data if it meets the minimal rolling-window criteria,
      8. Creates an additional combined interactive eq plot for beds=1 & 2 (single chart).
    """
    try:
        logging.info("Starting data preparation...")

        # 1) Sort
        data = data.copy()
        data.sort_values('Last Rental Date', inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Columns for weighting
        data['Dollar_Volume'] = data['Last Rental Price'].astype(float)
        data['Volume'] = np.nan       # bed-specific volume ($-weighted)
        data['Volume_eq'] = 1.0       # bed-specific volume eq

        bed_values = data['Bed'].unique()
        logging.info(f"Calculating VWAP for bed types: {bed_values}")

        # Rolling constraints
        THREE_MONTH_MIN = 3
        TWELVE_MONTH_MIN = 1

        # 2) Per-bed computations
        for b_type in bed_values:
            subset = data[data['Bed'] == b_type].copy()
            subset.set_index('Last Rental Date', inplace=True)
            subset.sort_index(inplace=True)

            # (A) $ Weighted
            subset['Volume'] = subset['Dollar_Volume']
            subset['Rolling_PV_3M'] = (subset['Last Rental Price'] * subset['Volume']).rolling(
                window='90D', min_periods=THREE_MONTH_MIN).sum()
            subset['Rolling_V_3M'] = subset['Volume'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

            subset['Rolling_PV_12M'] = (subset['Last Rental Price'] * subset['Volume']).rolling(
                window='365D', min_periods=TWELVE_MONTH_MIN).sum()
            subset['Rolling_V_12M'] = subset['Volume'].rolling(window='365D', min_periods=TWELVE_MONTH_MIN).sum()

            subset['VWAP_3M']  = (subset['Rolling_PV_3M']  / subset['Rolling_V_3M']).shift(1)
            subset['VWAP_12M'] = (subset['Rolling_PV_12M'] / subset['Rolling_V_12M']).shift(1)

            # (B) EQ Weighted
            subset['Volume_eq'] = 1.0
            subset['Rolling_PV_3M_eq'] = (subset['Last Rental Price'] * subset['Volume_eq']).rolling(
                window='90D', min_periods=THREE_MONTH_MIN).sum()
            subset['Rolling_V_3M_eq'] = subset['Volume_eq'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

            subset['Rolling_PV_12M_eq'] = (subset['Last Rental Price'] * subset['Volume_eq']).rolling(
                window='365D', min_periods=TWELVE_MONTH_MIN).sum()
            subset['Rolling_V_12M_eq'] = subset['Volume_eq'].rolling(window='365D', min_periods=TWELVE_MONTH_MIN).sum()

            subset['VWAP_3M_eq']  = (subset['Rolling_PV_3M_eq']  / subset['Rolling_V_3M_eq']).shift(1)
            subset['VWAP_12M_eq'] = (subset['Rolling_PV_12M_eq'] / subset['Rolling_V_12M_eq']).shift(1)

            # Move results back
            mask = (data['Bed'] == b_type)
            data.loc[mask, 'Volume']        = subset['Volume'].values
            data.loc[mask, 'VWAP_3M']       = subset['VWAP_3M'].values
            data.loc[mask, 'VWAP_12M']      = subset['VWAP_12M'].values
            data.loc[mask, 'Volume_eq']     = subset['Volume_eq'].values
            data.loc[mask, 'VWAP_3M_eq']    = subset['VWAP_3M_eq'].values
            data.loc[mask, 'VWAP_12M_eq']   = subset['VWAP_12M_eq'].values

        # 3) Global Weighted
        data.set_index('Last Rental Date', inplace=True)
        data.sort_index(inplace=True)

        # (A) Dollar Weighted
        data['Volume_all'] = data['Dollar_Volume']
        data['Rolling_PV_3M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='90D', min_periods=THREE_MONTH_MIN).sum()
        data['Rolling_V_3M_all'] = data['Volume_all'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

        data['Rolling_PV_12M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='365D', min_periods=TWELVE_MONTH_MIN).sum()
        data['Rolling_V_12M_all'] = data['Volume_all'].rolling(window='365D', min_periods=TWELVE_MONTH_MIN).sum()

        data['VWAP_3M_all']  = (data['Rolling_PV_3M_all']  / data['Rolling_V_3M_all']).shift(1)
        data['VWAP_12M_all'] = (data['Rolling_PV_12M_all'] / data['Rolling_V_12M_all']).shift(1)

        # (B) Equal Weighted
        data['Volume_all_eq'] = 1.0
        data['Rolling_PV_3M_all_eq'] = (data['Last Rental Price'] * data['Volume_all_eq']).rolling(
            window='90D', min_periods=THREE_MONTH_MIN).sum()
        data['Rolling_V_3M_all_eq'] = data['Volume_all_eq'].rolling(window='90D', min_periods=THREE_MONTH_MIN).sum()

        data['Rolling_PV_12M_all_eq'] = (data['Last Rental Price'] * data['Volume_all_eq']).rolling(
            window='365D', min_periods=TWELVE_MONTH_MIN).sum()
        data['Rolling_V_12M_all_eq'] = data['Volume_all_eq'].rolling(window='365D', min_periods=TWELVE_MONTH_MIN).sum()

        data['VWAP_3M_all_eq']  = (data['Rolling_PV_3M_all_eq']  / data['Rolling_V_3M_all_eq']).shift(1)
        data['VWAP_12M_all_eq'] = (data['Rolling_PV_12M_all_eq'] / data['Rolling_V_12M_all_eq']).shift(1)

        data.reset_index(inplace=True)

        # 4) Drop missing VWAP columns
        data.dropna(subset=[
            'VWAP_3M', 'VWAP_12M',
            'VWAP_3M_eq', 'VWAP_12M_eq',
            'VWAP_3M_all', 'VWAP_12M_all',
            'VWAP_3M_all_eq', 'VWAP_12M_all_eq'
        ], inplace=True)

        # 5) Calculate percentage diffs if needed
        data['Percentage_Diff'] = ((data['VWAP_3M'] - data['VWAP_12M']) / data['VWAP_12M']) * 100
        data['Percentage_Diff'] = data['Percentage_Diff'].replace([np.inf, -np.inf], np.nan).fillna(0)

        data['Percentage_Diff_eq'] = ((data['VWAP_3M_eq'] - data['VWAP_12M_eq']) / data['VWAP_12M_eq']) * 100
        data['Percentage_Diff_eq'] = data['Percentage_Diff_eq'].replace([np.inf, -np.inf], np.nan).fillna(0)

        data['Percentage_Diff_all'] = ((data['VWAP_3M_all'] - data['VWAP_12M_all']) / data['VWAP_12M_all']) * 100
        data['Percentage_Diff_all'] = data['Percentage_Diff_all'].replace([np.inf, -np.inf], np.nan).fillna(0)

        data['Percentage_Diff_all_eq'] = ((data['VWAP_3M_all_eq'] - data['VWAP_12M_all_eq']) / data['VWAP_12M_all_eq']) * 100
        data['Percentage_Diff_all_eq'] = data['Percentage_Diff_all_eq'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 6) Plot bed-specific VWAP
        bed_values = data['Bed'].unique()
        for b_type in bed_values:
            sub_b = data[data['Bed'] == b_type].copy()
            if sub_b.empty:
                logging.warning(f"No data for bed={b_type} after VWAP calculation; skipping plots.")
                continue

            logging.info(f"Plotting standard VWAP for {b_type}-bed (including eq lines if present).")
            plot_vwap(sub_b, b_type, plot_dir)

            logging.info(f"Plotting interactive global VWAP (dollar-based) for {b_type}-bed.")
            plot_vwap_interactive(sub_b, b_type, plot_dir)

        # 7) Global VWAP
        logging.info("Plotting global VWAP (matplotlib) for both $ and eq if present.")
        plot_global_vwap(data, plot_dir)

        logging.info("Plotting interactive global VWAP (plotly, dollar-weighted).")
        plot_global_vwap_interactive(data, plot_dir)

        # 8) Combined interactive eq plot for bed=1 & bed=2 (single aggregated line)
        logging.info("Plotting combined eq interactive for bed=1 & 2 ...")
        plot_eq_interactive_beds_1_and_2(data, plot_dir)

        # 9) Rolling medians for all-beds
        data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'])
        data.sort_values('Last Rental Date', inplace=True)
        data.set_index('Last Rental Date', inplace=True)

        data['Rolling_Median_1M_all'] = data['Last Rental Price'].rolling(window='30D', min_periods=3).median()
        data['Rolling_Median_12M_all'] = data['Last Rental Price'].rolling(window='365D', min_periods=10).median()

        data.dropna(subset=['Rolling_Median_1M_all', 'Rolling_Median_12M_all'], inplace=True)
        data.reset_index(inplace=True)

        logging.info("Plotting interactive global rolling median ...")
        plot_global_median_interactive(data, plot_dir)

        # 10) Dump columns for verification
        verify_cols = [
            'Last Rental Date', 'Bed', 'Last Rental Price',
            'Dollar_Volume', 'Volume', 'VWAP_3M', 'VWAP_12M',
            'Volume_eq', 'VWAP_3M_eq', 'VWAP_12M_eq',
            'Volume_all', 'VWAP_3M_all', 'VWAP_12M_all',
            'Volume_all_eq', 'VWAP_3M_all_eq', 'VWAP_12M_all_eq',
            'Percentage_Diff', 'Percentage_Diff_eq',
            'Percentage_Diff_all', 'Percentage_Diff_all_eq'
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
