# statistical_tests.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import save_plot

def advanced_statistical_tests(data, plot_dir):
    """Perform advanced statistical tests."""
    logging.info("Performing advanced statistical tests...")

    try:
        # ANOVA test
        groups = [group['Last Rental Price'].values for name, group in data.groupby('Bath')]
        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
            logging.info(f"ANOVA test F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")

        # Correlation matrix
        numeric_cols = data.select_dtypes(include='number').columns
        corr_matrix = data[numeric_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        save_plot(plt.gcf(), 'correlation_matrix', plot_dir)
        plt.close()

    except Exception as e:
        logging.error(f"Error in advanced statistical tests: {e}")
