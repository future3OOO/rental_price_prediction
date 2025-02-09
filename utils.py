# utils.py

import os
import logging
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PLOT_DIR

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PLOT_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def save_plot(fig, filename, plot_dir):
    """Save Matplotlib or Plotly figures."""
    try:
        file_path = os.path.join(plot_dir, f"{filename}.png")
        if hasattr(fig, 'write_image'):
            # For Plotly figures
            fig.write_image(file_path)
        else:
            # For Matplotlib figures
            fig.savefig(file_path, bbox_inches='tight')
        logging.info(f"Plot saved: {file_path}")
    except Exception as e:
        logging.error(f"Error saving plot '{filename}': {e}")

def save_plotly_fig(fig, filename, plot_dir):
    """Save Plotly figure to an HTML file."""
    try:
        file_path = os.path.join(plot_dir, f"{filename}.html")
        fig.write_html(file_path)
        logging.info(f"Plotly figure saved: {file_path}")
    except Exception as e:
        logging.error(f"Error saving Plotly figure '{filename}': {e}")

def encode_plot_to_base64(fig):
    """Encode Matplotlib figure to base64 string."""
    try:
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        img_data = base64.b64encode(img_buf.getvalue()).decode()
        plt.close(fig)
        return img_data
    except Exception as e:
        logging.error(f"Error encoding plot to base64: {e}")
        return None
