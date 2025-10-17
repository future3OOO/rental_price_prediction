# geospatial_analysis.py

import logging
import folium
from folium.plugins import HeatMap
import os

def geospatial_analysis(data, plot_dir):
    """Perform geospatial analysis and create a heatmap."""
    logging.info("Performing geospatial analysis...")

    if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
        logging.warning("Latitude and Longitude are required for geospatial analysis.")
        return

    try:
        # Create map
        map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        # Heatmap
        heat_data = data[['Latitude', 'Longitude', 'Last Rental Price']].values.tolist()
        HeatMap(heat_data).add_to(m)

        map_path = os.path.join(plot_dir, 'rental_price_heatmap.html')
        m.save(map_path)
        logging.info(f"Heatmap saved as '{map_path}'")

    except Exception as e:
        logging.error(f"Error during geospatial analysis: {e}")
