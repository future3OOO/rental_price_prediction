# reproducibility.py

import logging
import os

def reproducibility_guidelines(data, plot_dir):
    """
    Save cleaned data and a requirements.txt for environment reproducibility.
    """
    try:
        cleaned_csv = os.path.join(plot_dir, 'final_cleaned_data.csv')
        data.to_csv(cleaned_csv, index=False)
        logging.info(f"Final cleaned data => {cleaned_csv}")

        req_path = os.path.join(plot_dir, 'requirements.txt')
        with open(req_path,'w') as f:
            import subprocess
            subprocess.run(['pip','freeze'], stdout=f, check=True)
        logging.info(f"requirements.txt saved => {req_path}")
    except Exception as e:
        logging.error(f"Error in reproducibility_guidelines: {str(e)}")
