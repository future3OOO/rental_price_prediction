# time_series_analysis.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import traceback

def time_based_analysis(data, plot_dir):
    """Perform time-based analysis and forecasting based on bedroom type."""
    logging.info("Performing time-based analysis...")

    try:
        # Ensure 'Last Rental Date' is in datetime format
        data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'], errors='coerce')
        logging.info("'Last Rental Date' converted to datetime format.")

        # Remove rows with missing dates, prices, or bedroom types
        initial_shape = data.shape
        data = data.dropna(subset=['Last Rental Date', 'Last Rental Price', 'Bed'])
        logging.info(f"Dropped rows with missing 'Last Rental Date', 'Last Rental Price', or 'Bed'.\n"
                     f"Shape before: {initial_shape}, after: {data.shape}")

        # Create a quarterly period column
        data['Quarterly Period'] = data['Last Rental Date'].dt.to_period('Q')
        logging.info("Created 'Quarterly Period' column.")

        # Get unique bedroom types
        bed_types = data['Bed'].unique()
        logging.info(f"Unique bedroom types found: {bed_types}")

        if len(bed_types) == 0:
            logging.warning("No bedroom types found after data cleaning. Skipping time-based analysis.")
            return

        for bed_type in bed_types:
            bed_data = data[data['Bed'] == bed_type].copy()

            # Proceed only if enough data is available
            if len(bed_data) < 10:
                logging.info(f"Not enough data for bed type: {bed_type}. Skipping.")
                continue

            logging.info(f"Processing bed type: {bed_type}")

            # Aggregate data by Quarterly Period
            period_median = bed_data.groupby('Quarterly Period')['Last Rental Price'].median().reset_index()
            period_median['Quarterly Period Timestamp'] = period_median['Quarterly Period'].dt.to_timestamp()

            logging.info(f"Prepared data for Prophet for bed type: {bed_type}")

            # Prepare DataFrame for Prophet
            prophet_df = period_median.rename(columns={'Quarterly Period Timestamp': 'ds', 'Last Rental Price': 'y'})
            logging.info(f"Prepared DataFrame for Prophet forecasting for bed type: {bed_type}")

            try:
                # Initialize Prophet model with quarterly seasonality
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(prophet_df)
                logging.info(f"Prophet model fitted for bed type: {bed_type}")

                # Create future dataframe for forecasting
                future = model.make_future_dataframe(periods=4, freq='Q')  # Forecasting next 4 quarters
                forecast = model.predict(future)
                logging.info(f"Forecasting completed for bed type: {bed_type}")

                # Plot the forecast
                fig = model.plot(forecast)
                plt.title(f'Rental Price Forecast for {int(bed_type)}-Bedroom Properties')
                plt.xlabel('Date')
                plt.ylabel('Rental Price (NZD per week)')
                plt.tight_layout()

                # Ensure the plot directory exists
                os.makedirs(plot_dir, exist_ok=True)
                plot_filename = f'prophet_forecast_{int(bed_type)}_bed.png'
                plot_path = os.path.join(plot_dir, plot_filename)
                fig.savefig(plot_path)
                plt.close(fig)
                logging.info(f"Forecast plot saved: {plot_path}")

            except Exception as e:
                logging.error(f"Error forecasting for bed type {bed_type}: {e}")
                logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"Error in time-based analysis: {e}")
        logging.error(traceback.format_exc())
