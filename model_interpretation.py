# model_interpretation.py

import shap
import matplotlib.pyplot as plt
import os
import logging
import pickle
import pandas as pd
import numpy as np

def model_interpretation_with_shap(
    model, 
    X_train_transformed, 
    feature_names, 
    plot_dir, 
    model_name='Model', 
    save_shap_values=False,
    cat_features=None
):
    """Generate SHAP plots for model interpretation."""
    try:
        os.makedirs(plot_dir, exist_ok=True)

        # Ensure X_train_transformed is a DataFrame
        if not isinstance(X_train_transformed, pd.DataFrame):
            X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)

        # Prepare data for SHAP (encode categorical features)
        X_shap = X_train_transformed.copy()
        if cat_features:
            for col in cat_features:
                if col in X_shap.columns:
                    X_shap[col] = X_shap[col].astype('category').cat.codes

        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # Save SHAP values if required
        if save_shap_values:
            shap_values_path = os.path.join(plot_dir, f'shap_values_{model_name}.pkl')
            with open(shap_values_path, 'wb') as f:
                pickle.dump(shap_values, f)
            logging.info(f"SHAP values saved at '{shap_values_path}'")

        # Summary plot
        shap.summary_plot(
            shap_values, 
            X_shap, 
            plot_type='dot', 
            show=False
        )
        # Get the current figure and axes
        fig = plt.gcf()
        # Save the figure
        summary_plot_path = os.path.join(plot_dir, f'shap_summary_{model_name}.png')
        fig.savefig(summary_plot_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"SHAP summary plot saved at '{summary_plot_path}'")

        # Feature importance plot
        shap.summary_plot(
            shap_values, 
            X_shap, 
            plot_type='bar', 
            show=False
        )
        fig = plt.gcf()
        importance_plot_path = os.path.join(plot_dir, f'shap_feature_importance_{model_name}.png')
        fig.savefig(importance_plot_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"SHAP feature importance plot saved at '{importance_plot_path}'")

        # Identify top features
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_indices = np.argsort(feature_importance)[-5:][::-1]
        top_features = [X_shap.columns[idx] for idx in top_features_indices]

        # Dependence plots for top features
        for feature in top_features:
            shap.dependence_plot(
                feature, 
                shap_values, 
                X_shap, 
                interaction_index=None,
                show=False
            )
            fig = plt.gcf()
            dependence_plot_path = os.path.join(plot_dir, f'shap_dependence_{feature}_{model_name}.png')
            fig.savefig(dependence_plot_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"SHAP dependence plot for {feature} saved.")

    except Exception as e:
        logging.error(f"Error in model_interpretation_with_shap: {e}")
        raise