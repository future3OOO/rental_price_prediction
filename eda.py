# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
import os
from plotting import (
    plot_average_price_per_month,
    plot_cumulative_change,
    plot_vwap_interactive
)

def perform_eda(data, plot_dir):
    """
    Perform exploratory data analysis, including plotting median rental prices,
    advanced statistical tests, and interactive visualizations.
    """
    try:
        logging.info("Starting exploratory data analysis...")

        # Ensure the plot directory exists
        os.makedirs(plot_dir, exist_ok=True)

        # Plot median rental prices per suburb
        plot_median_rental_price_per_suburb(data, plot_dir)

        # Add the new plotting functions here
        date_column = 'Last Rental Date'
        price_column = 'Last Rental Price'
        
        # Plot average price per month
        logging.info("Generating average price per month plot...")
        plot_average_price_per_month(data, date_column, price_column, plot_dir)
        
        # Plot cumulative change
        logging.info("Generating cumulative change plot...")
        plot_cumulative_change(data, date_column, price_column, plot_dir)

        # Perform advanced statistical tests
        advanced_statistical_tests(data, plot_dir)

        # Create interactive visualizations
        interactive_visualizations(data, plot_dir)

        logging.info("Exploratory data analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in perform_eda: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_median_rental_price_per_suburb(data, plot_dir):
    """Plot median rental prices for each suburb."""
    logging.info("Plotting median rental prices per suburb...")
    try:
        if 'Suburb' not in data.columns or 'Last Rental Price' not in data.columns:
            raise KeyError("'Suburb' or 'Last Rental Price' column is missing.")

        suburb_median = data.groupby('Suburb')['Last Rental Price'].median().reset_index()
        suburb_median = suburb_median.rename(columns={'Last Rental Price': 'Median Rental Price'})

        # Sort suburbs by median rental price
        suburb_median = suburb_median.sort_values(by='Median Rental Price', ascending=False)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=suburb_median, x='Suburb', y='Median Rental Price', palette='viridis')
        plt.xlabel('Suburb')
        plt.ylabel('Median Rental Price (NZD per week)')
        plt.title('Median Rental Prices by Suburb')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'median_rental_prices_by_suburb.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting median rental prices per suburb: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream

def advanced_statistical_tests(data, plot_dir):
    """Perform advanced statistical analysis and generate plots."""
    logging.info("Performing advanced statistical tests...")
    try:
        import numpy as np  # Ensure numpy is imported

        # Ensure the plot directory exists
        os.makedirs(plot_dir, exist_ok=True)

        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # Calculate correlation for numeric columns
        numeric_correlation = data[numeric_cols].corr()

        # Plot correlation heatmap for numeric columns
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'numeric_correlation_heatmap.png'))
        plt.close()
        logging.info("Numeric correlation heatmap saved.")

        # Analyze categorical columns
        for cat_col in categorical_cols:
            if cat_col not in ['Street Address', 'Last Rental Date']:  # Exclude columns not needed
                plt.figure(figsize=(10, 6))
                avg_price = data.groupby(cat_col)['Last Rental Price'].mean().sort_values(ascending=False)

                if avg_price.empty:
                    logging.info(f"No data for categorical column '{cat_col}'. Skipping.")
                    continue

                avg_price.plot(kind='bar')
                plt.title(f'Average Rental Price by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel('Average Rental Price')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                filename = f"avg_price_by_{cat_col.replace(' ', '_').lower()}.png"
                plt.savefig(os.path.join(plot_dir, filename))
                plt.close()
                logging.info(f"Plot saved: {filename}")

        logging.info("Advanced statistical tests completed successfully.")

    except Exception as e:
        logging.error(f"Error performing advanced statistical tests: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream

def interactive_visualizations(data, plot_dir):
    """Create interactive visualizations."""
    logging.info("Creating interactive visualizations...")
    try:
        import plotly.express as px
        os.makedirs(plot_dir, exist_ok=True)

        # Interactive scatter plot
        fig = px.scatter(
            data_frame=data,
            x='Floor Size (sqm)',
            y='Last Rental Price',
            color='Suburb',
            title='Floor Size vs. Rental Price by Suburb'
        )
        plot_path = os.path.join(plot_dir, 'interactive_scatter.html')
        fig.write_html(plot_path)
        logging.info(f"Interactive scatter plot saved: {plot_path}")

        # Rental Price Distribution
        fig_dist = px.histogram(
            data_frame=data,
            x='Last Rental Price',
            nbins=30,
            title='Distribution of Rental Prices'
        )
        dist_plot_path = os.path.join(plot_dir, 'rental_price_distribution.html')
        fig_dist.write_html(dist_plot_path)
        logging.info(f"Rental price distribution plot saved: {dist_plot_path}")

        # Rental Price vs Number of Bedrooms
        fig_bed = px.box(
            data_frame=data,
            x='Bed',
            y='Last Rental Price',
            title='Rental Price vs Number of Bedrooms'
        )
        bed_plot_path = os.path.join(plot_dir, 'rental_price_vs_bedrooms.html')
        fig_bed.write_html(bed_plot_path)
        logging.info(f"Rental price vs bedrooms plot saved: {bed_plot_path}")

        # Rental Price vs Suburb (Box Plot)
        fig_suburb = px.box(
            data_frame=data,
            x='Suburb',
            y='Last Rental Price',
            title='Rental Price Distribution by Suburb'
        )
        fig_suburb.update_xaxes(tickangle=45)
        suburb_plot_path = os.path.join(plot_dir, 'rental_price_by_suburb.html')
        fig_suburb.write_html(suburb_plot_path)
        logging.info(f"Rental price by suburb plot saved: {suburb_plot_path}")

    except Exception as e:
        logging.error(f"Error creating interactive visualizations: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream
