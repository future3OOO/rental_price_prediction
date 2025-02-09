# reproducibility.py

import logging
import os

def reproducibility_guidelines(data, plot_dir):
    """Save cleaned data and requirements for reproducibility."""
    logging.info("Saving cleaned data and requirements for reproducibility...")

    try:
        # Save cleaned data
        cleaned_data_path = os.path.join(plot_dir, 'cleaned_rental_data.csv')
        data.to_csv(cleaned_data_path, index=False)
        logging.info(f"Cleaned data saved at '{cleaned_data_path}'")

        # Save requirements
        requirements_path = os.path.join(plot_dir, 'requirements.txt')
        with open(requirements_path, 'w') as f:
            import subprocess
            subprocess.run(['pip', 'freeze'], stdout=f, check=True)
        logging.info(f"Requirements saved at '{requirements_path}'")

    except Exception as e:
        logging.error(f"Error in reproducibility guidelines: {e}")
