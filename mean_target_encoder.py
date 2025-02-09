# mean_target_encoder.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, columns=None, default_value=np.nan):
        self.target_column = target_column
        self.columns = columns  # Columns to encode
        self.default_value = default_value
        self.encoding_dict_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        # For each column, compute the mean target value
        for col in self.columns:
            mean_encoded = pd.concat([X[col], y], axis=1).groupby(col)[self.target_column].mean()
            self.encoding_dict_[col] = mean_encoded
        # Compute overall mean to replace missing values
        self.global_mean_ = y.mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col + '_mean_encoded'] = X_transformed[col].map(self.encoding_dict_[col])
            # Replace NaN values with global mean
            X_transformed[col + '_mean_encoded'] = X_transformed[col + '_mean_encoded'].fillna(self.global_mean_)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.columns
        output_features = [f"{col}_mean_encoded" for col in input_features]
        return output_features