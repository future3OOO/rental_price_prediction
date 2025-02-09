import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import traceback

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Enhanced feature engineering transformer that:
    - Dynamically includes date-derived features (Month_sin, Month_cos, Time_Index) in numeric features if present.
    - Creates interaction terms (if needed).
    - No longer scales here; scaling is handled by MinMaxScaler in the main pipeline.
    """

    def __init__(self, numeric_features, categorical_features, interaction_terms=None, fill_value=0):
        self.numeric_features = numeric_features.copy()  # Copy to avoid mutating the original list
        self.categorical_features = categorical_features
        self.interaction_terms = interaction_terms if interaction_terms is not None else []
        self.fill_value = fill_value
        self.feature_names_out_ = []
        self.interaction_feature_names_ = []
        self.category_levels_ = {}

        # Potential date-derived features added by DateTransformer:
        self.date_derived_features = ['Month_sin', 'Month_cos', 'Time_Index']

    def fit(self, X, y=None):
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be a pandas DataFrame.")

            # Check for date-derived features and add them to numeric_features if present
            for df in self.date_derived_features:
                if df in X.columns and df not in self.numeric_features:
                    self.numeric_features.append(df)

            # Start feature_names_out_ with numeric and categorical features
            self.feature_names_out_ = self.numeric_features + self.categorical_features

            # Prepare interaction feature names
            interaction_feature_names = []
            for feat1, feat2 in self.interaction_terms:
                if feat1 in self.numeric_features and feat2 in self.numeric_features:
                    # Numeric-numeric interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    interaction_feature_names.append(interaction_name)
                elif feat1 in self.numeric_features and feat2 in self.categorical_features:
                    # Numeric-categorical interaction
                    categories = X[feat2].fillna('Unknown').astype(str).unique()
                    self.category_levels_[feat2] = categories
                    for cat in categories:
                        safe_cat = cat.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_x_{feat2}_{safe_cat}"
                        interaction_feature_names.append(interaction_name)
                elif feat1 in self.categorical_features and feat2 in self.numeric_features:
                    # Categorical-numeric interaction
                    categories = X[feat1].fillna('Unknown').astype(str).unique()
                    self.category_levels_[feat1] = categories
                    for cat in categories:
                        safe_cat = cat.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_{safe_cat}_x_{feat2}"
                        interaction_feature_names.append(interaction_name)

            # Add interaction features to feature_names_out_
            self.feature_names_out_.extend(interaction_feature_names)
            self.interaction_feature_names_ = interaction_feature_names

            # Ensure uniqueness of feature names (in case of duplicates)
            self.feature_names_out_ = list(dict.fromkeys(self.feature_names_out_))

            logging.info(f"Feature engineering initialized with {len(self.feature_names_out_)} features")
            logging.info(f"Features: {self.feature_names_out_}")

            return self
        except Exception as e:
            logging.error(f"Error in FeatureEngineering.fit: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X):
        try:
            X_transformed = pd.DataFrame(index=X.index)

            # Numeric features: already determined in fit(), including date-derived
            X_numeric = X[self.numeric_features].apply(pd.to_numeric, errors='coerce').fillna(self.fill_value)
            # Ensure float32 for memory efficiency
            X_numeric = X_numeric.astype(np.float32)

            # Add numeric features to transformed DataFrame
            X_transformed = pd.concat([X_transformed, X_numeric], axis=1)

            # Categorical features
            for col in self.categorical_features:
                X_transformed[col] = X[col].fillna('Unknown').astype(str)

            # Create interaction features
            for feat1, feat2 in self.interaction_terms:
                if feat1 in self.numeric_features and feat2 in self.numeric_features:
                    interaction_name = f"{feat1}_x_{feat2}"
                    X_transformed[interaction_name] = (X_numeric[feat1] * X_numeric[feat2]).astype(np.float32)
                elif feat1 in self.numeric_features and feat2 in self.categorical_features:
                    categories = self.category_levels_.get(feat2, [])
                    for cat in categories:
                        safe_cat = cat.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_x_{feat2}_{safe_cat}"
                        cat_indicator = (X[feat2].fillna('Unknown').astype(str) == cat).astype(np.float32)
                        X_transformed[interaction_name] = (X_numeric[feat1] * cat_indicator).astype(np.float32)
                elif feat1 in self.categorical_features and feat2 in self.numeric_features:
                    categories = self.category_levels_.get(feat1, [])
                    for cat in categories:
                        safe_cat = cat.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_{safe_cat}_x_{feat2}"
                        cat_indicator = (X[feat1].fillna('Unknown').astype(str) == cat).astype(np.float32)
                        X_transformed[interaction_name] = (cat_indicator * X_numeric[feat2]).astype(np.float32)

            # Ensure columns match feature_names_out_
            X_transformed = X_transformed[self.feature_names_out_]

            return X_transformed
        except Exception as e:
            logging.error(f"Error in FeatureEngineering.transform: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
