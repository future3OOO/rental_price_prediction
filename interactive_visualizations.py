# interactive_visualizations.py

import logging
import plotly.express as px
import pandas as pd
import os
from utils import save_plotly_fig

def interactive_visualizations(data, plot_dir):
    """Create interactive visualizations using Plotly."""
    logging.info("Creating interactive visualizations...")
    try:
        # Example: Interactive scatter plot
        fig = px.scatter(
            data,
            x='Living Space (sqm)',
            y='Last Rental Price',
            color='Suburb',
            title='Living Space vs. Last Rental Price',
            hover_data=['Bed', 'Bath', 'Car']
        )
        # Save the figure using save_plotly_fig
        save_plotly_fig(fig, 'interactive_scatter_plot', plot_dir)

        # ---------------------------------------------------------------
        # New: Interactive bar chart of median price per sqm by suburb
        # ---------------------------------------------------------------
        required_cols = {'Last Rental Price', 'Floor Size (sqm)', 'Suburb'}
        if required_cols.issubset(data.columns):
            logging.info("Creating price-per-sqm interactive plot by suburb ...")
            # Ensure numeric division safety
            tmp = data.copy()
            tmp = tmp[tmp['Floor Size (sqm)'] > 0]
            tmp['Price_per_sqm'] = tmp['Last Rental Price'] / tmp['Floor Size (sqm)']

            agg_df = (
                tmp.groupby('Suburb', as_index=False)['Price_per_sqm']
                .median()
                .sort_values('Price_per_sqm', ascending=False)
            )

            fig_ppsqm = px.bar(
                agg_df,
                x='Suburb',
                y='Price_per_sqm',
                title='Median Annual Rental Price per sqm by Suburb',
                hover_data={'Price_per_sqm': ':.2f'},
            )
            fig_ppsqm.update_layout(
                xaxis_title='Suburb',
                yaxis_title='Price per sqm (annual)',
                xaxis_tickangle=-45
            )

            save_plotly_fig(fig_ppsqm, 'price_per_sqm_by_suburb', plot_dir)
            logging.info("Price-per-sqm interactive plot created.")
        else:
            missing = required_cols - set(data.columns)
            logging.warning(
                f"Cannot create price-per-sqm plot; missing columns: {missing}."
            )

        logging.info("Interactive visualizations created successfully.")
    except Exception as e:
        logging.error(f"Error creating interactive visualizations: {e}")
