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
        logging.info("Interactive visualizations created successfully.")
    except Exception as e:
        logging.error(f"Error creating interactive visualizations: {e}")
