) here are the main source code files


# model_training.py

import logging
import traceback
import joblib
import pandas as pd
import numpy as np
import optuna
import json
import os
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from transformers import DateTransformer
from feature_engineering import FeatureEngineering
from data_cleaning import data_cleaning
from config import MODEL_DIR, PLOT_DIR
from plotting import (
    plot_feature_importance,
    plot_learning_curve,
    plot_residuals,
    plot_actual_vs_predicted
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import time
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib
import matplotlib.pyplot as plt

class PreprocessingWithFeatureNames(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessing_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.feature_names = None

    def fit(self, X, y=None):
        self.preprocessing_pipeline.fit(X, y)
        self.feature_names = self.get_feature_names(self.preprocessing_pipeline)
        return self

    def transform(self, X):
        X_transformed = self.preprocessing_pipeline.transform(X)
        if isinstance(X_transformed, np.ndarray):
            X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        else:
            X_transformed.columns = self.feature_names
        return X_transformed

    def get_feature_names(self, column_transformer):
        output_features = []

        for name, transformer, features in column_transformer.transformers_:
            if transformer == 'drop':
                continue
            elif transformer == 'passthrough':
                if features == 'remainder':
                    remainder_features = column_transformer._feature_names_in_
                    output_features.extend(remainder_features)
                else:
                    output_features.extend(features)
            else:
                if hasattr(transformer, 'named_steps'):
                    transformer = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out()
                    output_features.extend(names)
                else:
                    output_features.extend(features)
        return output_features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names

def clip_outliers(data, column, lower_percentile=0.05, upper_percentile=0.95):
    original_count = len(data[column])
    lower_bound = data[column].quantile(lower_percentile)
    upper_bound = data[column].quantile(upper_percentile)

    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

def calculate_vif(X_numeric, feature_names):
    try:
        df = pd.DataFrame(X_numeric, columns=feature_names)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.median())

        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(df.values, i) for i in range(df.shape[1])
        ]
        return vif_data
    except Exception as e:
        logging.error(f"Error calculating VIF: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    if not np.any(nonzero_indices):
        return np.inf
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

def make_prediction_with_range(full_pipeline, X_new, mae):
    """
    Make a prediction with the model and provide an estimated prediction interval using MAE.
    """
    try:
        # Make prediction using the full pipeline
        prediction = full_pipeline.predict(X_new)[0]

        # Calculate margins using MAE
        margin = mae

        lower_bound = max(prediction - margin, 0)  # Ensure lower bound is not negative
        upper_bound = prediction + margin

        logging.info(f"Prediction: {prediction}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        return prediction, lower_bound, upper_bound
    except Exception as e:
        logging.error(f"Error in make_prediction_with_range: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def train_model(data):
    try:
        logging.info("Starting model training with hyperparameter tuning...")
        logging.info(f"Received data shape: {data.shape}")

        # Define the target column
        target_column = 'Last Rental Price'

        # 1. Define Features
        base_numeric_features = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Capital Value', 'VWAP_3M'
        ]
        base_categorical_features = ['Active Listing', 'Suburb', 'Bed']

        date_column = 'Last Rental Date'
        required_features = base_numeric_features + base_categorical_features + [date_column]
        interaction_terms = [
            ('Suburb', 'VWAP_3M'),
            ('Floor Size (sqm)', 'Capital Value'),
            ('Bed', 'Capital Value'),
            ('Bed', 'Bath'),
            ('Year Built', 'Floor Size (sqm)'),
            ('Bed', 'Suburb'),
            ('Bath', 'Floor Size (sqm)'),
            ('Car', 'Land Size (sqm)')
        ]

        # 2. Initial datetime conversion
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

        # 3. Sort by date
        data = data.sort_values(date_column).reset_index(drop=True)

        # 4. Clean and transform full dataset
        data = data_cleaning(data)

        # 5. Ensure numeric features are properly converted
        for col in base_numeric_features + [target_column]:
            if col in data.columns:
                data[col] = data[col].replace({'False': np.nan, 'True': np.nan})
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = np.nan
                data[col] = data[col].fillna(data[col].median())

        # Handle missing values in categorical features
        for col in base_categorical_features:
            if col in data.columns:
                data[col] = data[col].fillna('Unknown')
            else:
                data[col] = 'Unknown'

        # Ensure categorical features are strings
        for col in base_categorical_features:
            data[col] = data[col].astype(str)

        # 6. Handle outliers
        columns_to_clip = ['Land Size (sqm)', 'Floor Size (sqm)', 'Capital Value', 'Last Rental Price']
        for column in columns_to_clip:
            if column in data.columns:
                data = clip_outliers(
                    data, column,
                    lower_percentile=0.05,
                    upper_percentile=0.95
                )

        # 7. Prepare X and y
        X = data[required_features].copy()
        y = data[target_column]

        # 8. Split into train/test using time-based split
        dates = pd.to_datetime(X[date_column])
        cutoff_date = dates.sort_values().iloc[int(len(dates) * 0.8)]

        train_mask = dates <= cutoff_date
        X_train = X[train_mask].copy()
        X_test = X[~train_mask].copy()
        y_train = y[train_mask]
        y_test = y[~train_mask]

        numeric_features = base_numeric_features
        categorical_features = base_categorical_features

        # 9. Build preprocessing pipeline
        date_transformer = DateTransformer(date_column=date_column, drop_original=True)
        feature_engineering = FeatureEngineering(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            interaction_terms=interaction_terms
        )

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])

        # Target Encoding for 'Suburb'
        from category_encoders import TargetEncoder
        target_encoder = TargetEncoder(cols=['Suburb'])

        # Combine preprocessing steps
        preprocessing_pipeline = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features + feature_engineering.interaction_feature_names_),
            ('cat', 'passthrough', [col for col in categorical_features if col != 'Suburb'])
        ])

        # Create full preprocessing pipeline
        full_preprocessing_pipeline = Pipeline(steps=[
            ('date_transformer', date_transformer),
            ('feature_engineering', feature_engineering),
            ('target_encoder', target_encoder),
            ('preprocessing_with_feature_names', PreprocessingWithFeatureNames(preprocessing_pipeline))
        ])

        # 10. Fit and transform training data
        X_train_transformed = full_preprocessing_pipeline.fit_transform(X_train, y_train)
        feature_names = X_train_transformed.columns.tolist()

        # 11. Transform test data
        X_test_transformed = full_preprocessing_pipeline.transform(X_test)

        # Ensure that categorical features are of string type after preprocessing
        for df in [X_train_transformed, X_test_transformed]:
            for col in categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype(str)

        # 12. Get categorical feature names
        cat_features = [col for col in categorical_features if col != 'Suburb']

        # 13. Check for missing values and fill them
        for df in [X_train_transformed, X_test_transformed]:
            if df.isnull().any().any():
                df.fillna(df.median(), inplace=True)

        # 14. Proceed with VIF calculation (on training data only)
        numeric_feature_names = [col for col in X_train_transformed.columns if col not in cat_features]
        X_numeric = X_train_transformed[numeric_feature_names]
        X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
        X_numeric = X_numeric.dropna(axis=1, how='all')

        if not X_numeric.empty:
            vif_data = calculate_vif(X_numeric, X_numeric.columns.tolist())
            if vif_data is not None:
                high_vif_threshold = 10
                high_vif_features = vif_data[vif_data['VIF'] > high_vif_threshold]['Feature'].tolist()
                if high_vif_features:
                    for feature in high_vif_features:
                        if feature in ['Capital Value', 'VWAP_3M']:
                            continue
                        else:
                            X_train_transformed.drop(columns=[feature], inplace=True)
                            X_test_transformed.drop(columns=[feature], inplace=True)
                            numeric_feature_names.remove(feature)
                    X_numeric = X_train_transformed[numeric_feature_names]
                    vif_data = calculate_vif(X_numeric, numeric_feature_names)
            else:
                raise Exception("VIF calculation failed. Please check the data and transformations.")
        else:
            raise Exception("VIF calculation failed due to absence of numeric features.")

        # Update feature names after dropping features
        feature_names = X_train_transformed.columns.tolist()

        # 15. Split training data into training and validation sets for hyperparameter tuning
        X_train_tune, X_valid_tune, y_train_tune, y_valid_tune = train_test_split(
            X_train_transformed, y_train, test_size=0.2, shuffle=False
        )

        # 16. Hyperparameter tuning with Optuna
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 1000, 5000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 12),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'eval_metric': 'MAE',
                'od_type': 'Iter',
                'od_wait': 50,
                'task_type': 'GPU',
                'devices': '0'
            }
            model = CatBoostRegressor(
                **params,
                loss_function='MAE',
                cat_features=cat_features,
                random_seed=42,
                verbose=0
            )
            model.fit(
                X_train_tune,
                y_train_tune,
                eval_set=(X_valid_tune, y_valid_tune),
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=0
            )
            y_pred = model.predict(X_valid_tune)
            mae = mean_absolute_error(y_valid_tune, y_pred)
            return mae

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        best_params = study.best_params

        # Remove parameters not needed for final training
        for param in ['eval_metric', 'od_type', 'od_wait']:
            best_params.pop(param, None)

        # 17. Combine training and validation sets for final training
        X_train_full = pd.concat([X_train_tune, X_valid_tune])
        y_train_full = pd.concat([y_train_tune, y_valid_tune])

        # 18. Train final model with best hyperparameters
        model = CatBoostRegressor(
            **best_params,
            loss_function='MAE',
            cat_features=cat_features,
            random_seed=42,
            verbose=100
        )

        model.fit(
            X_train_full,
            y_train_full,
            eval_set=(X_test_transformed, y_test),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=100
        )

        # 19. Make predictions on the test set
        y_pred = model.predict(X_test_transformed)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)

        # 20. Cross-Validation after model training
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
            X_fold_train = X_train_full.iloc[train_idx]
            X_fold_val = X_train_full.iloc[val_idx]
            y_fold_train = y_train_full.iloc[train_idx]
            y_fold_val = y_train_full.iloc[val_idx]

            model_fold = CatBoostRegressor(
                **best_params,
                loss_function='MAE',
                cat_features=cat_features,
                random_seed=42,
                verbose=0
            )
            model_fold.fit(
                X_fold_train,
                y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=0
            )

            y_pred_fold = model_fold.predict(X_fold_val)
            mae_fold = mean_absolute_error(y_fold_val, y_pred_fold)
            cv_scores.append(mae_fold)

        mean_cv_score = np.mean(cv_scores)

        # 21. Plotting
        plot_learning_curve(
            estimator=model,
            X=X_train_full,
            y=y_train_full,
            plot_dir=PLOT_DIR,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            fit_params={'early_stopping_rounds': 100, 'verbose': False, 'cat_features': cat_features}
        )

        plot_feature_importance(
            model=model,
            feature_names=feature_names,
            plot_dir=PLOT_DIR,
            importance_type='PredictionValuesChange'
        )

        plot_residuals(y_test, y_pred, PLOT_DIR)
        plot_actual_vs_predicted(y_test, y_pred, PLOT_DIR)

        # Compute and Plot SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_full)

        # SHAP summary plot
        try:
            shap.summary_plot(shap_values, X_train_full, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(PLOT_DIR, 'shap_summary.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting SHAP summary: {e}")
            logging.error(traceback.format_exc())

        # Dependence Plots for top features
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.get_feature_importance()
        }).sort_values(by='Importance', ascending=False)

        top_features = importance_df['Feature'].head(5).tolist()
        for feature in top_features:
            shap.dependence_plot(feature, shap_values, X_train_full, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(PLOT_DIR, f'shap_dependence_{feature}.png'), bbox_inches='tight')
            plt.close()

        # 22. Save the CatBoost model separately
        os.makedirs(MODEL_DIR, exist_ok=True)
        catboost_model_path = os.path.join(MODEL_DIR, 'catboost_model.cbm')
        model.save_model(catboost_model_path)

        # 23. Save the preprocessing pipeline separately
        preprocessing_pipeline_path = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')
        joblib.dump(full_preprocessing_pipeline, preprocessing_pipeline_path)

        # 24. Create the full pipeline
        full_pipeline_with_model = Pipeline([
            ('preprocessor', full_preprocessing_pipeline),
            ('model', model)
        ])

        # 25. Save the full pipeline
        full_pipeline_path = os.path.join(MODEL_DIR, 'full_pipeline.joblib')
        joblib.dump(full_pipeline_with_model, full_pipeline_path)

        # 26. Save the model metadata
        metadata = {
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape,
            'feature_names': feature_names,
            'best_params': best_params,
            'cat_features': cat_features,
            'date_column': date_column,
            'numeric_features': numeric_features,
        }
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info("Model training and evaluation completed successfully.")

        # Return necessary outputs
        return {
            'model': model,
            'preprocessing_pipeline': full_preprocessing_pipeline,
            'full_pipeline': full_pipeline_with_model,
            'metadata': metadata,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'feature_names': feature_names,
            'date_column': date_column,
            'cat_features': cat_features,
        }

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        logging.error(traceback.format_exc())
        raise


# feature_engineering.py

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import traceback
from sklearn.preprocessing import StandardScaler

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering transformer that handles both numeric and categorical features
    while preserving categorical features for CatBoost's native handling.
    """

    def __init__(self, numeric_features, categorical_features, interaction_terms=None, fill_value=0):
        """
        Initialize the transformer.

        Parameters:
            numeric_features (list): List of numeric feature names
            categorical_features (list): List of categorical feature names
            interaction_terms (list of tuples): List of feature pairs for interactions
            fill_value (float): Value to fill missing numeric data
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.interaction_terms = interaction_terms if interaction_terms is not None else []
        self.fill_value = fill_value
        self.feature_names_out_ = []
        self.numeric_scaler = StandardScaler()
        self.interaction_feature_names_ = []
        self.category_levels_ = {}

    def fit(self, X, y=None):
        """Fit the transformer, scaling numeric features while preserving categorical ones."""
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input X must be a pandas DataFrame.")

            # Handle numeric features
            numeric_data = X[self.numeric_features].fillna(self.fill_value)
            # Center numeric features to reduce multicollinearity
            self.numeric_scaler.fit(numeric_data)

            # Initialize feature names list
            self.feature_names_out_ = []

            # Add numeric features
            self.feature_names_out_.extend(self.numeric_features)

            # Add categorical features
            self.feature_names_out_.extend(self.categorical_features)

            # Prepare to collect interaction feature names
            interaction_feature_names = []

            # Collect unique categories for categorical features involved in interactions
            for feat1, feat2 in self.interaction_terms:
                if feat1 in self.numeric_features and feat2 in self.categorical_features:
                    categories = X[feat2].fillna('Unknown').astype(str).unique()
                    self.category_levels_[feat2] = categories

                    for category in categories:
                        safe_category = category.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_x_{feat2}_{safe_category}"
                        interaction_feature_names.append(interaction_name)
                elif feat1 in self.categorical_features and feat2 in self.numeric_features:
                    categories = X[feat1].fillna('Unknown').astype(str).unique()
                    self.category_levels_[feat1] = categories
                    for category in categories:
                        safe_category = category.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                        interaction_name = f"{feat1}_{safe_category}_x_{feat2}"
                        interaction_feature_names.append(interaction_name)
                elif feat1 in self.numeric_features and feat2 in self.numeric_features:
                    interaction_name = f"{feat1}_x_{feat2}"
                    interaction_feature_names.append(interaction_name)
                elif feat1 in self.categorical_features and feat2 in self.categorical_features:
                    # Optionally handle interactions between two categorical features
                    pass  # Skipping for now

            # Add interaction feature names to feature_names_out_
            self.feature_names_out_.extend(interaction_feature_names)
            self.interaction_feature_names_ = interaction_feature_names

            # Ensure feature_names_out_ is unique to prevent duplicate labels
            self.feature_names_out_ = list(dict.fromkeys(self.feature_names_out_))

            logging.info(f"Feature engineering initialized with {len(self.feature_names_out_)} features")
            logging.info(f"Features: {self.feature_names_out_}")

            return self

        except Exception as e:
            logging.error(f"Error in FeatureEngineering.fit: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X):
        """Transform features while preserving categorical data for CatBoost."""
        try:
            # Initialize DataFrame to store transformed features
            X_transformed = pd.DataFrame(index=X.index)

            # Ensure correct data types for numeric features
            X_numeric = X[self.numeric_features].apply(pd.to_numeric, errors='coerce').fillna(self.fill_value)

            # Scale numeric features
            X_numeric_scaled = pd.DataFrame(
                self.numeric_scaler.transform(X_numeric),
                columns=self.numeric_features,
                index=X.index
            )
            # Add scaled numeric features to X_transformed
            X_transformed = pd.concat([X_transformed, X_numeric_scaled], axis=1)

            # Handle categorical features (keep as object/string type)
            for col in self.categorical_features:
                if col not in X.columns:
                    X_transformed[col] = 'Unknown'
                else:
                    X_transformed[col] = X[col].fillna('Unknown').astype(str)

            # Create interaction features
            for feat1, feat2 in self.interaction_terms:
                if feat1 in X.columns and feat2 in X.columns:
                    if feat1 in self.numeric_features and feat2 in self.numeric_features:
                        interaction_name = f"{feat1}_x_{feat2}"
                        X_transformed[interaction_name] = (
                            X_numeric[feat1] * X_numeric[feat2]
                        )
                    elif feat1 in self.numeric_features and feat2 in self.categorical_features:
                        categories = self.category_levels_[feat2]
                        for category in categories:
                            safe_category = category.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                            interaction_name = f"{feat1}_x_{feat2}_{safe_category}"
                            cat_indicator = (X[feat2].fillna('Unknown').astype(str) == category).astype(float)
                            X_transformed[interaction_name] = (
                                X_numeric[feat1] * cat_indicator
                            )
                    elif feat1 in self.categorical_features and feat2 in self.numeric_features:
                        categories = self.category_levels_[feat1]
                        for category in categories:
                            safe_category = category.replace(' ', '_').replace(',', '').replace(';', '').replace('/', '_')
                            interaction_name = f"{feat1}_{safe_category}_x_{feat2}"
                            cat_indicator = (X[feat1].fillna('Unknown').astype(str) == category).astype(float)
                            X_transformed[interaction_name] = (
                                cat_indicator * X_numeric[feat2]
                            )
                    elif feat1 in self.categorical_features and feat2 in self.categorical_features:
                        # Optionally handle interactions between two categorical features
                        pass  # Skipping for now

            # Ensure columns are in the same order as feature_names_out_
            X_transformed = X_transformed[self.feature_names_out_]

            # Check for non-finite values in numeric columns
            numeric_columns = self.numeric_features + self.interaction_feature_names_
            if not np.isfinite(X_transformed[numeric_columns].values.astype(float)).all():
                logging.warning("Non-finite values detected in numeric columns")
                X_transformed[numeric_columns] = X_transformed[numeric_columns].replace([np.inf, -np.inf], np.nan)
                X_transformed[numeric_columns] = X_transformed[numeric_columns].fillna(self.fill_value)

            # Ensure numeric columns are of numeric data type
            X_transformed[numeric_columns] = X_transformed[numeric_columns].astype(float)

            return X_transformed

        except Exception as e:
            logging.error(f"Error in FeatureEngineering.transform: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

# main.py

import logging
import traceback
import sys
import sklearn
import pandas as pd
import os
from config import (
    RAW_DATA_PATH, PLOT_DIR, METADATA_PATH, MODEL_PATH,
    OPTUNA_STUDY_PATH, PROCESSED_DATA_DIR, MODEL_DIR
)
from data_cleaning import data_cleaning
from outlier_removal import sophisticated_outlier_removal
from data_preparation import prepare_data
from eda import perform_eda
from model_training import train_model, make_prediction_with_range
from reproducibility import reproducibility_guidelines
from plotting import plot_average_price_per_month, plot_cumulative_change

def run_full_analysis():
    """
    Execute the full analysis workflow including data loading, cleaning,
    outlier removal, preparation, exploratory data analysis, model training,
    and reproducibility guidelines.
    """
    try:
        logging.info("Starting full analysis workflow...")

        # Load your data
        logging.info("Loading data...")
        data = pd.read_csv(RAW_DATA_PATH)
        if data is None or data.empty:
            raise ValueError("Data loading failed or data is empty.")
        logging.info(f"Data shape after loading: {data.shape}")

        # Data Cleaning
        logging.info("Cleaning data...")
        data = data_cleaning(data)
        logging.info(f"Data shape after cleaning: {data.shape}")
        logging.info(f"Columns after data cleaning: {data.columns.tolist()}")

        # Remove outliers
        logging.info("Removing outliers...")
        numeric_features_for_outlier_removal = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Capital Value', 'Last Rental Price'
        ]
        data, num_outliers = sophisticated_outlier_removal(
            data,
            property_type_column='Bed',
            numerical_features=numeric_features_for_outlier_removal,
            output_dir=PLOT_DIR
        )
        logging.info(f"Removed {num_outliers} outliers. Remaining data shape: {data.shape}")

        # Data Preparation
        logging.info("Preparing data...")
        data = prepare_data(data, PLOT_DIR)
        logging.info(f"Data shape after preparation: {data.shape}")

        # Ensure required columns are present
        if 'Last Rental Date' in data.columns and 'Last Rental Price' in data.columns:
            date_column = 'Last Rental Date'
            price_column = 'Last Rental Price'
            plot_dir = PLOT_DIR  # Ensure PLOT_DIR is correctly defined

            # Before plotting, check data integrity
            if data[date_column].isnull().any() or data[price_column].isnull().any():
                logging.error("Data contains null values in date or price columns. Cannot generate plots.")
            else:
                # Proceed with plotting
                plot_average_price_per_month(data, date_column, price_column, plot_dir)
                plot_cumulative_change(data, date_column, price_column, plot_dir)
        else:
            logging.error("Required columns for plotting are missing from data.")

        # Validate numerical features
        numeric_features = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Capital Value',
            'Percentage_Diff', 'VWAP_3M'
        ]
        categorical_features = ['Active Listing', 'Suburb', 'Bed']

        # Ensure that all required columns are present
        required_columns = numeric_features + categorical_features + ['Last Rental Price', 'Last Rental Date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} are missing from the dataset after preparation.")

        logging.info("All required columns are present in the dataset.")
        logging.info(f"Dataset shape: {data.shape}")

        # Exploratory Data Analysis
        logging.info("Performing exploratory data analysis...")
        perform_eda(data, PLOT_DIR)

        # Train the model
        logging.info("Training the model...")
        result = train_model(data)
        if 'error' in result:
            logging.error(f"Model training failed with error: {result['error']}")
            raise Exception(f"Model training failed: {result['error']}")
        full_pipeline = result['full_pipeline']
        model = result['model']
        metadata = result['metadata']
        date_column = result['date_column']

        if full_pipeline is None or model is None:
            raise ValueError("Model training failed.")
        logging.info("Model trained successfully.")

        # Prepare the test instance
        test_features = numeric_features + categorical_features + ['Last Rental Date']
        logging.info(f"Test features: {test_features}")

        # Validate test features
        missing_features = [feat for feat in test_features if feat not in data.columns]
        if missing_features:
            logging.error(f"Missing features in data: {missing_features}")
            raise ValueError(f"Missing features in data: {missing_features}")

        # Extract the latest instance for prediction
        X_test = data[test_features].iloc[-1:]
        y_test_actual = data['Last Rental Price'].iloc[-1]

        # Extract 'mae' and 'mape' from metadata
        mae = metadata['MAE']
        mape = metadata['MAPE']

        # Log the values of mae and mape
        logging.info(f"MAE: {mae}")
        logging.info(f"MAPE: {mape}")

        # Make prediction using the full pipeline
        y_pred, lower, upper = make_prediction_with_range(
            full_pipeline, X_test, mae
        )
        logging.info("Prediction made successfully.")

        # Output the results
        print(f"Predicted price: ${y_pred:.2f}")
        print(f"Expected range: ${lower:.2f} to ${upper:.2f}")
        print(f"Actual price: ${y_test_actual:.2f}")

        # Reproducibility guidelines
        logging.info("Generating reproducibility guidelines...")
        reproducibility_guidelines(data, PLOT_DIR)
        logging.info("Reproducibility guidelines generated successfully.")

        logging.info("Full analysis complete.")
    except Exception as e:
        logging.error(f"An error occurred during the analysis: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    finally:
        logging.info("Analysis process finished.")

if __name__ == "__main__":
    # Configure logging once at the entry point
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("main.log"),
            logging.StreamHandler()
        ]
    )

    # Log Python executable and scikit-learn version for debugging
    logging.info(f"Using Python executable: {sys.executable}")
    logging.info(f"scikit-learn version: {sklearn.__version__}")

    # Run your analysis
    run_full_analysis()

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

# transformers.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging
import traceback

class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts cyclical date features from a date column.
    """
    def __init__(self, date_column='Last Rental Date', drop_original=True, date_format=None):
        """
        Initializes the DateTransformer.

        Parameters:
            date_column (str): The name of the date column to transform.
            drop_original (bool): Whether to drop the original date column after transformation.
            date_format (str): The format of the date strings. Example: '%Y-%m-%d'
        """
        self.date_column = date_column
        self.drop_original = drop_original
        self.earliest_date = None  # For calculating Time_Index
        self.date_format = date_format
        self.new_feature_names_ = None  # To store the names of the new features

    def fit(self, X, y=None):
        try:
            # Ensure date column exists
            if self.date_column not in X.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during fit.")

            # Convert to datetime and handle errors
            X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')

            # Store the earliest date for Time_Index calculation
            self.earliest_date = X[self.date_column].min()

            if pd.isnull(self.earliest_date):
                raise ValueError(f"All dates in column '{self.date_column}' are NaT or invalid.")

            return self
        except Exception as e:
            logging.error(f"Error in DateTransformer.fit: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def transform(self, X):
        try:
            X = X.copy()

            # Ensure the date column exists
            if self.date_column not in X.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in input data during transform.")

            # Convert to datetime and handle errors
            X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format, errors='coerce')

            # Extract cyclical month features
            X['Month'] = X[self.date_column].dt.month.fillna(0).astype(int)
            X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
            X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)

            # Create Time_Index (e.g., months since earliest date)
            X['Time_Index'] = ((X[self.date_column] - self.earliest_date).dt.days // 30).fillna(0).astype(int)

            # Store the new feature names for get_feature_names_out
            self.new_feature_names_ = ['Month_sin', 'Month_cos', 'Time_Index']

            # Drop the original date column and 'Month' if configured to do so
            columns_to_drop = [self.date_column, 'Month'] if self.drop_original else ['Month']
            X = X.drop(columns=columns_to_drop, errors='ignore')

            return X
        except Exception as e:
            logging.error(f"Error in DateTransformer.transform: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def get_feature_names_out(self, input_features=None):
        """
        Returns the names of the features produced by the transformer.

        Parameters:
            input_features (array-like of str): Input feature names. If None, defaults to the date column.

        Returns:
            List[str]: Output feature names.
        """
        return self.new_feature_names_

# data_cleaning.py

import pandas as pd
import numpy as np
import logging
from config import CLEANED_DATA_PATH  # Ensure this path is correctly set in your config
import traceback

def data_cleaning(data):
    """Clean the dataset by handling missing values and ensuring correct data types."""
    logging.info("Starting data cleaning...")

    try:
        # Normalize the 'Suburb' column
        data['Suburb'] = data['Suburb'].str.strip().str.upper()
        logging.info("Normalized 'Suburb' column.")

        # Rename columns to ensure consistent 'sqm' usage
        data.rename(columns={
            'Land Size (m²)': 'Land Size (sqm)',
            'Floor Size (m²)': 'Floor Size (sqm)'
        }, inplace=True)
        logging.info(f"Columns after renaming: {data.columns.tolist()}")

        # Convert numeric columns to appropriate data types
        numeric_columns = [
            'Bed', 'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Last Rental Price', 'Days on Market', 'Capital Value'
        ]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                missing_values = data[col].isnull().sum()
                logging.info(f"Converted '{col}' to numeric, missing values: {missing_values}")
            else:
                logging.warning(f"Column '{col}' not found in the data.")

        # Fill missing values for numeric columns with median (without using inplace=True)
        fill_value_columns = {
            'Car': 'Car',
            'Land Size (sqm)': 'Land Size (sqm)',
            'Floor Size (sqm)': 'Floor Size (sqm)',
            'Year Built': 'Year Built',
            'Last Rental Price': 'Last Rental Price',
            'Days on Market': 'Days on Market'
        }

        for col in fill_value_columns:
            if col in data.columns:
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                logging.info(f"Filled missing values in '{col}' with median value: {median_value}")
            else:
                logging.warning(f"Column '{col}' not found in the data.")

        # Handle parsing of 'Last Rental Date'
        if 'Last Rental Date' in data.columns:
            # Log sample dates before parsing
            sample_dates = data['Last Rental Date'].dropna().head(10).tolist()
            logging.info(f"Sample 'Last Rental Date' values before parsing: {sample_dates}")

            # Specify the correct date format based on your data
            date_format = '%d-%b-%y'  # Example format: '1-Mar-24'
            data['Last Rental Date'] = pd.to_datetime(
                data['Last Rental Date'],
                format=date_format,
                errors='coerce'
            )
            logging.info(f"Date parsing with format='{date_format}' completed.")

            # Check how many dates were successfully parsed
            valid_dates = data['Last Rental Date'].notnull().sum()
            logging.info(f"Number of valid 'Last Rental Date' after parsing: {valid_dates}")

            missing_dates = data['Last Rental Date'].isnull().sum()
            logging.info(f"Missing 'Last Rental Date' after parsing: {missing_dates}")

            if missing_dates > 0:
                median_date = data['Last Rental Date'].median()
                if pd.isnull(median_date):
                    logging.warning("Median date is NaT. All dates might be invalid.")
                else:
                    data['Last Rental Date'] = data['Last Rental Date'].fillna(median_date)
                    logging.info(f"Filled missing 'Last Rental Date' values with median date: {median_date}")
            else:
                logging.info("No missing 'Last Rental Date' values after parsing.")
        else:
            logging.warning("'Last Rental Date' column not found in the data.")

        # Remove properties with more than 5 bedrooms
        data = data[data['Bed'] <= 5]
        logging.info(f"Data shape after removing 6+ bedroom properties: {data.shape}")

        # Log the final state of the data
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {data.columns.tolist()}")

        # Save cleaned data to a CSV for reproducibility purposes
        data.to_csv(CLEANED_DATA_PATH, index=False)
        logging.info(f"Cleaned data saved at '{CLEANED_DATA_PATH}'")

        return data
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        logging.error(traceback.format_exc())
        raise

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from scipy import stats
import plotly.graph_objects as go
from plotting import plot_vwap_interactive, plot_global_vwap_interactive, plot_global_median_interactive
import traceback

def remove_outliers(group):
    z_scores = np.abs(stats.zscore(group['Last Rental Price']))
    return group[z_scores < 4]

def plot_moving_medians(data, bed_type, plot_dir):
    try:
        bed_data = data[data['Bed'] == bed_type].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(bed_data['Last Rental Date'], bed_data['Rolling_Median_3M'], label='3-Month Moving Median')
        plt.plot(bed_data['Last Rental Date'], bed_data['Rolling_Median_12M'], label='12-Month Moving Median')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(bed_data['Last Rental Date'], bed_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title(f'Moving Medians for {bed_type}-Bedroom Properties')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = f'moving_medians_{bed_type}bed.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Moving medians plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_moving_medians for bed type {bed_type}: {str(e)}")
        raise

def plot_vwap(data, bed_type, plot_dir):
    try:
        bed_data = data[data['Bed'] == bed_type].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(bed_data['Last Rental Date'], bed_data['VWAP_3M'], label='3-Month VWAP')
        plt.plot(bed_data['Last Rental Date'], bed_data['VWAP_12M'], label='12-Month VWAP')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(bed_data['Last Rental Date'], bed_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title(f'VWAP for {bed_type}-Bedroom Properties')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = f'vwap_{bed_type}bed.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"VWAP plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_vwap for bed type {bed_type}: {str(e)}")
        raise

def plot_global_vwap(data, plot_dir):
    """Plot the global (all-bedroom) VWAP"""
    try:
        global_data = data.copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(global_data['Last Rental Date'], global_data['VWAP_3M_all'], label='3-Month VWAP (All Beds)')
        plt.plot(global_data['Last Rental Date'], global_data['VWAP_12M_all'], label='12-Month VWAP (All Beds)')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(global_data['Last Rental Date'], global_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title('VWAP for All Bedroom Properties Combined')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = 'vwap_all_beds.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Global VWAP plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_global_vwap: {str(e)}")
        raise

def prepare_data(data, plot_dir):
    try:
        logging.info("Starting data preparation...")
        data = data.copy()

        # Sort data chronologically
        data = data.sort_values('Last Rental Date').reset_index(drop=True)

        # Calculate rolling VWAP for each bedroom type
        for bed_type in data['Bed'].unique():
            bed_data = data[data['Bed'] == bed_type].copy()
            bed_data = bed_data.set_index('Last Rental Date').sort_index()

            # Assume each rental has a volume of 1
            bed_data['Volume'] = 1

            # Calculate rolling sum of price * volume and volume for 3M and 12M
            bed_data['Rolling_PV_3M'] = (bed_data['Last Rental Price'] * bed_data['Volume']).rolling(
                window='90D', min_periods=5).sum()
            bed_data['Rolling_V_3M'] = bed_data['Volume'].rolling(window='90D', min_periods=5).sum()

            bed_data['Rolling_PV_12M'] = (bed_data['Last Rental Price'] * bed_data['Volume']).rolling(
                window='365D', min_periods=20).sum()
            bed_data['Rolling_V_12M'] = bed_data['Volume'].rolling(window='365D', min_periods=20).sum()

            # Calculate VWAP
            bed_data['VWAP_3M'] = bed_data['Rolling_PV_3M'] / bed_data['Rolling_V_3M']
            bed_data['VWAP_12M'] = bed_data['Rolling_PV_12M'] / bed_data['Rolling_V_12M']

            # Shift the VWAP to avoid look-ahead bias
            bed_data['VWAP_3M'] = bed_data['VWAP_3M'].shift(1)
            bed_data['VWAP_12M'] = bed_data['VWAP_12M'].shift(1)

            # Assign back to main dataframe
            data.loc[data['Bed'] == bed_type, 'Volume'] = bed_data['Volume'].values
            data.loc[data['Bed'] == bed_type, 'VWAP_3M'] = bed_data['VWAP_3M'].values
            data.loc[data['Bed'] == bed_type, 'VWAP_12M'] = bed_data['VWAP_12M'].values

        # After assigning VWAP values back to the main dataframe for each bed type
        # Now calculate the VWAP for all combined bedroom types
        data = data.set_index('Last Rental Date').sort_index()
        data['Volume_all'] = 1

        data['Rolling_PV_3M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='90D', min_periods=5).sum()
        data['Rolling_V_3M_all'] = data['Volume_all'].rolling(window='90D', min_periods=5).sum()

        data['Rolling_PV_12M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='365D', min_periods=20).sum()
        data['Rolling_V_12M_all'] = data['Volume_all'].rolling(window='365D', min_periods=20).sum()

        data['VWAP_3M_all'] = data['Rolling_PV_3M_all'] / data['Rolling_V_3M_all']
        data['VWAP_12M_all'] = data['Rolling_PV_12M_all'] / data['Rolling_V_12M_all']

        # Shift the global VWAP to avoid look-ahead bias
        data['VWAP_3M_all'] = data['VWAP_3M_all'].shift(1)
        data['VWAP_12M_all'] = data['VWAP_12M_all'].shift(1)

        # Revert index to column form
        data.reset_index(inplace=True)

        # Remove rows with NaN values in VWAP columns
        data.dropna(subset=['VWAP_3M', 'VWAP_12M'], inplace=True)
        data.dropna(subset=['VWAP_3M_all', 'VWAP_12M_all'], inplace=True)

        # Calculate the percentage differences
        data['Percentage_Diff'] = ((data['VWAP_3M'] - data['VWAP_12M']) / data['VWAP_12M']) * 100
        data['Percentage_Diff'] = data['Percentage_Diff'].replace([np.inf, -np.inf], np.nan).fillna(0)

        data['Percentage_Diff_all'] = ((data['VWAP_3M_all'] - data['VWAP_12M_all']) / data['VWAP_12M_all']) * 100
        data['Percentage_Diff_all'] = data['Percentage_Diff_all'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Plot VWAP for different bedroom types
        for bed_type in data['Bed'].unique():
            logging.info(f"Processing {bed_type}-bedroom properties...")

            bed_data = data[data['Bed'] == bed_type].copy()
            if bed_data.empty:
                logging.warning(f"No data available for {bed_type}-bedroom properties after preprocessing. Skipping plots.")
                continue  # Skip to the next bedroom type

            logging.info(f"Plotting VWAP for {bed_type}-bedroom properties...")
            plot_vwap(bed_data, bed_type, plot_dir)

            logging.info(f"Plotting interactive VWAP for {bed_type}-bedroom properties...")
            plot_vwap_interactive(bed_data, bed_type, plot_dir)

        # Plot global VWAP
        plot_global_vwap(data, plot_dir)
        plot_global_vwap_interactive(data, plot_dir)

        # Create Quarterly Period column
        data['Quarterly Period'] = pd.to_datetime(data['Last Rental Date']).dt.to_period('Q').astype(str)

        # Ensure 'Last Rental Date' is datetime and set as index
        data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'])
        data.set_index('Last Rental Date', inplace=True)
        data.sort_index(inplace=True)

        # Calculate global rolling medians for 1M and 12M
        data['Rolling_Median_1M_all'] = data['Last Rental Price'].rolling(window='30D', min_periods=1).median()
        data['Rolling_Median_12M_all'] = data['Last Rental Price'].rolling(window='365D', min_periods=5).median()

        # Optionally remove the shift if not needed
        # data['Rolling_Median_1M_all'] = data['Rolling_Median_1M_all'].shift(1)
        # data['Rolling_Median_12M_all'] = data['Rolling_Median_12M_all'].shift(1)

        # Reset index to bring 'Last Rental Date' back as a column
        data.reset_index(inplace=True)

        # Remove rows with NaN values in rolling median columns
        data.dropna(subset=['Rolling_Median_1M_all', 'Rolling_Median_12M_all'], inplace=True)

        # Plot global rolling median
        plot_global_median_interactive(data, plot_dir)

        logging.info("Data preparation completed successfully.")
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        logging.error(f"Error in prepare_data: {str(e)}")
        logging.error(traceback.format_exc())
        raise

# outlier_removal.py

import pandas as pd
import numpy as np
import logging
import traceback
import os

def sophisticated_outlier_removal(data, target_variable='Last Rental Price',
                                  numerical_features=None, output_dir='output',
                                  property_type_column='Bed'):
    """
    Enhanced function to remove outliers, including mislabeled properties based on 'Bed' and 'Year Built',
    using frequency analysis instead of z-scores for discrete variables.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        target_variable (str): The target variable column name.
        numerical_features (list): List of supporting numerical feature column names.
        output_dir (str): Directory to save the dump file of removed outliers.
        property_type_column (str): Column name defining different property types.

    Returns:
        Tuple[pd.DataFrame, int]: The DataFrame with outliers removed and the number of outliers removed.
    """
    try:
        if numerical_features is None:
            numerical_features = [
                'Bed', 'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
                'Capital Value', 'Time_Index', 'Year Built'  # Ensure 'Year Built' is included
            ]

        logging.info("Starting sophisticated outlier removal...")

        # Check if required columns exist in the DataFrame
        required_columns = set([target_variable, property_type_column] + numerical_features)
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the DataFrame: {missing_columns}")

        # Adjusted thresholds
        TARGET_ZSCORE_THRESHOLD = 1.8  # Threshold for target variable
        SUPPORTING_FEATURES_ZSCORE_THRESHOLD = 1  # Threshold for supporting features
        MIN_SUPPORTING_FEATURES = 2  # Minimum number of supporting features required for removal

        # Function to calculate modified z-score for a group
        def modified_zscore_group(group):
            median = group.median()
            mad = (group - median).abs().median()
            mad = mad if mad != 0 else 1e-9  # Prevent division by zero
            return 0.6745 * (group - median) / mad

        # Calculate modified z-scores for target variable and numerical features by property type
        target_z_scores = data.groupby(property_type_column)[target_variable].transform(modified_zscore_group)
        feature_z_scores = data.groupby(property_type_column)[numerical_features].transform(modified_zscore_group)

        # Identify potential outliers in the target variable
        potential_outliers = data[np.abs(target_z_scores) > TARGET_ZSCORE_THRESHOLD].index

        logging.info(f"Initial potential outliers in target variable: {len(potential_outliers)}")

        # Frequency analysis for 'Bed' based on 'Year Built'
        logging.info("Performing frequency analysis on 'Bed' based on 'Year Built'...")
        bed_year_counts = data.groupby('Year Built')['Bed'].value_counts(normalize=True)
        bed_year_counts = bed_year_counts.reset_index(name='frequency')

        # Define a frequency threshold for rare 'Bed' counts
        FREQUENCY_THRESHOLD = 0.01  # Define as needed (e.g., 5%)

        # Create a set of tuples (Year Built, Bed) for rare combinations
        rare_bed_year_set = set(
            bed_year_counts[bed_year_counts['frequency'] < FREQUENCY_THRESHOLD][['Year Built', 'Bed']].apply(tuple, axis=1)
        )

        # For each potential outlier, check conditions for removal
        outliers_to_remove = []
        outlier_details = []
        for idx in potential_outliers:
            capital_value_zscore = feature_z_scores.at[idx, 'Capital Value']
            # Check if Capital Value z-score is positive and greater than +3.9
            if capital_value_zscore > 3.5:
                # Exclude this outlier from removal
                logging.debug(f"Index {idx} is excluded from removal due to high positive 'Capital Value' z-score: {capital_value_zscore}")
                continue  # Skip to the next potential outlier

            supporting_outliers = []
            property_type = data.at[idx, property_type_column]

            # Check supporting features (excluding 'Capital Value')
            for feature in numerical_features:
                if feature == 'Capital Value':
                    continue
                # Handle 'Bed' separately
                if feature == 'Bed':
                    year_built = data.at[idx, 'Year Built']
                    bed_value = data.at[idx, 'Bed']
                    if (year_built, bed_value) in rare_bed_year_set:
                        supporting_outliers.append('Bed (rare for Year Built)')
                else:
                    # Calculate the threshold for this feature based on its distribution within the same property type
                    feature_threshold = np.percentile(
                        np.abs(feature_z_scores[data[property_type_column] == property_type][feature]), 80)
                    threshold = max(SUPPORTING_FEATURES_ZSCORE_THRESHOLD, feature_threshold)
                    if np.abs(feature_z_scores.at[idx, feature]) > threshold:
                        supporting_outliers.append(feature)

            # If there are at least the minimum number of supporting outliers, mark for removal
            if len(supporting_outliers) >= MIN_SUPPORTING_FEATURES:
                outliers_to_remove.append(idx)
                outlier_details.append({
                    'index': idx,
                    'property_type': property_type,
                    'target_value': data.at[idx, target_variable],
                    'target_zscore': target_z_scores.at[idx],
                    'Capital Value_zscore': capital_value_zscore,
                    'supporting_outliers': ', '.join(supporting_outliers),
                    **{f'{feature}_value': data.at[idx, feature] for feature in numerical_features},
                    **{f'{feature}_zscore': feature_z_scores.at[idx, feature] for feature in numerical_features}
                })
            else:
                logging.debug(f"Index {idx} has insufficient supporting outlier features: {supporting_outliers}")

        # Remove identified outliers
        data_cleaned = data.drop(index=outliers_to_remove)
        num_outliers_removed = len(outliers_to_remove)

        logging.info(f"Removed {num_outliers_removed} outliers.")
        logging.info(f"Data shape after outlier removal: {data_cleaned.shape}")

        # Create dump file of removed outliers
        if num_outliers_removed > 0:
            os.makedirs(output_dir, exist_ok=True)
            outlier_df = pd.DataFrame(outlier_details)
            dump_file_path = os.path.join(output_dir, 'removed_outliers.csv')
            outlier_df.to_csv(dump_file_path, index=False)
            logging.info(f"Dump file of removed outliers created at: {dump_file_path}")

        return data_cleaned, num_outliers_removed
    except Exception as e:
        logging.error(f"Error in sophisticated_outlier_removal: {str(e)}")
        logging.error(traceback.format_exc())
        raise


# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
import os
from plotting import (
    plot_average_price_per_month,
    plot_cumulative_change,
    plot_vwap_interactive
)

def perform_eda(data, plot_dir):
    """
    Perform exploratory data analysis, including plotting median rental prices,
    advanced statistical tests, and interactive visualizations.
    """
    try:
        logging.info("Starting exploratory data analysis...")

        # Ensure the plot directory exists
        os.makedirs(plot_dir, exist_ok=True)

        # Plot median rental prices per suburb
        plot_median_rental_price_per_suburb(data, plot_dir)

        # Add the new plotting functions here
        date_column = 'Last Rental Date'
        price_column = 'Last Rental Price'
        
        # Plot average price per month
        logging.info("Generating average price per month plot...")
        plot_average_price_per_month(data, date_column, price_column, plot_dir)
        
        # Plot cumulative change
        logging.info("Generating cumulative change plot...")
        plot_cumulative_change(data, date_column, price_column, plot_dir)

        # Perform advanced statistical tests
        advanced_statistical_tests(data, plot_dir)

        # Create interactive visualizations
        interactive_visualizations(data, plot_dir)

        logging.info("Exploratory data analysis completed successfully.")
    except Exception as e:
        logging.error(f"Error in perform_eda: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_median_rental_price_per_suburb(data, plot_dir):
    """Plot median rental prices for each suburb."""
    logging.info("Plotting median rental prices per suburb...")
    try:
        if 'Suburb' not in data.columns or 'Last Rental Price' not in data.columns:
            raise KeyError("'Suburb' or 'Last Rental Price' column is missing.")

        suburb_median = data.groupby('Suburb')['Last Rental Price'].median().reset_index()
        suburb_median = suburb_median.rename(columns={'Last Rental Price': 'Median Rental Price'})

        # Sort suburbs by median rental price
        suburb_median = suburb_median.sort_values(by='Median Rental Price', ascending=False)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=suburb_median, x='Suburb', y='Median Rental Price', palette='viridis')
        plt.xlabel('Suburb')
        plt.ylabel('Median Rental Price (NZD per week)')
        plt.title('Median Rental Prices by Suburb')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, 'median_rental_prices_by_suburb.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error plotting median rental prices per suburb: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream

def advanced_statistical_tests(data, plot_dir):
    """Perform advanced statistical analysis and generate plots."""
    logging.info("Performing advanced statistical tests...")
    try:
        import numpy as np  # Ensure numpy is imported

        # Ensure the plot directory exists
        os.makedirs(plot_dir, exist_ok=True)

        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # Calculate correlation for numeric columns
        numeric_correlation = data[numeric_cols].corr()

        # Plot correlation heatmap for numeric columns
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap (Numeric Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'numeric_correlation_heatmap.png'))
        plt.close()
        logging.info("Numeric correlation heatmap saved.")

        # Analyze categorical columns
        for cat_col in categorical_cols:
            if cat_col not in ['Street Address', 'Last Rental Date']:  # Exclude columns not needed
                plt.figure(figsize=(10, 6))
                avg_price = data.groupby(cat_col)['Last Rental Price'].mean().sort_values(ascending=False)

                if avg_price.empty:
                    logging.info(f"No data for categorical column '{cat_col}'. Skipping.")
                    continue

                avg_price.plot(kind='bar')
                plt.title(f'Average Rental Price by {cat_col}')
                plt.xlabel(cat_col)
                plt.ylabel('Average Rental Price')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                filename = f"avg_price_by_{cat_col.replace(' ', '_').lower()}.png"
                plt.savefig(os.path.join(plot_dir, filename))
                plt.close()
                logging.info(f"Plot saved: {filename}")

        logging.info("Advanced statistical tests completed successfully.")

    except Exception as e:
        logging.error(f"Error performing advanced statistical tests: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream

def interactive_visualizations(data, plot_dir):
    """Create interactive visualizations."""
    logging.info("Creating interactive visualizations...")
    try:
        import plotly.express as px
        os.makedirs(plot_dir, exist_ok=True)

        # Interactive scatter plot
        fig = px.scatter(
            data_frame=data,
            x='Floor Size (sqm)',
            y='Last Rental Price',
            color='Suburb',
            title='Floor Size vs. Rental Price by Suburb'
        )
        plot_path = os.path.join(plot_dir, 'interactive_scatter.html')
        fig.write_html(plot_path)
        logging.info(f"Interactive scatter plot saved: {plot_path}")

        # Rental Price Distribution
        fig_dist = px.histogram(
            data_frame=data,
            x='Last Rental Price',
            nbins=30,
            title='Distribution of Rental Prices'
        )
        dist_plot_path = os.path.join(plot_dir, 'rental_price_distribution.html')
        fig_dist.write_html(dist_plot_path)
        logging.info(f"Rental price distribution plot saved: {dist_plot_path}")

        # Rental Price vs Number of Bedrooms
        fig_bed = px.box(
            data_frame=data,
            x='Bed',
            y='Last Rental Price',
            title='Rental Price vs Number of Bedrooms'
        )
        bed_plot_path = os.path.join(plot_dir, 'rental_price_vs_bedrooms.html')
        fig_bed.write_html(bed_plot_path)
        logging.info(f"Rental price vs bedrooms plot saved: {bed_plot_path}")

        # Rental Price vs Suburb (Box Plot)
        fig_suburb = px.box(
            data_frame=data,
            x='Suburb',
            y='Last Rental Price',
            title='Rental Price Distribution by Suburb'
        )
        fig_suburb.update_xaxes(tickangle=45)
        suburb_plot_path = os.path.join(plot_dir, 'rental_price_by_suburb.html')
        fig_suburb.write_html(suburb_plot_path)
        logging.info(f"Rental price by suburb plot saved: {suburb_plot_path}")

    except Exception as e:
        logging.error(f"Error creating interactive visualizations: {e}")
        logging.error(traceback.format_exc())
        raise  # Keep raise to allow exception handling upstream

# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import numpy as np
import traceback
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_feature_importance(
    model,
    feature_names,
    plot_dir,
    importance_type='PredictionValuesChange',
    top_n=20
):
    """
    Plot and save feature importance based on the specified importance_type.

    Parameters:
        model: Trained CatBoost model.
        feature_names: List of feature names.
        plot_dir: Directory to save the plot.
        importance_type: Type of feature importance ('PredictionValuesChange' by default).
        top_n: Number of top features to plot (20 by default).
    """
    try:
        # Get feature importance from the model
        importance = model.get_feature_importance(type=importance_type)

        # Convert importance to list if necessary
        importance = importance.tolist() if isinstance(importance, np.ndarray) else importance

        # Retrieve feature names from the model to ensure alignment
        model_feature_names = model.feature_names_

        # Log lengths
        logging.info(f"Length of model_feature_names: {len(model_feature_names)}")
        logging.info(f"Length of importance: {len(importance)}")

        # Check if lengths match
        if len(model_feature_names) != len(importance):
            logging.error("Length of model's feature_names and importance do not match.")
            # Synchronize lengths
            min_length = min(len(model_feature_names), len(importance))
            model_feature_names = model_feature_names[:min_length]
            importance = importance[:min_length]
            logging.info(f"Synchronized lengths to {min_length}")

        # Create a DataFrame for plotting
        feature_importance = pd.DataFrame({
            'Feature': model_feature_names,
            'Importance': importance
        })

        # Check if DataFrame is empty
        if feature_importance.empty:
            logging.error("Feature importance DataFrame is empty.")
            return

        # Sort and select top features
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(top_n)

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title(f'Top {top_n} Feature Importance ({importance_type})', fontsize=16)
        plt.tight_layout()

        # Ensure the plot directory exists
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'feature_importance_{importance_type}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Feature importance plot ({importance_type}) saved at '{plot_path}'.")
    except Exception as e:
        logging.error(f"Error in plot_feature_importance: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_learning_curve(
    estimator,
    X,
    y,
    plot_dir,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),  # Reduced to 5 training sizes
    scoring='neg_mean_absolute_error',
    fit_params=None  # Add fit_params parameter
):
    """
    Plot and save the learning curve of the estimator.

    Parameters:
        estimator: The object implementing 'fit' and 'predict' methods.
        X: Training vector.
        y: Target vector.
        plot_dir: Directory to save the plot.
        cv: Cross-validation strategy.
        n_jobs: Number of jobs to run in parallel.
        train_sizes: Relative or absolute numbers of training examples.
        scoring: Scoring parameter.
        fit_params: Parameters to pass to the fit method of the estimator.
    """
    try:
        # Compute learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring=scoring,
            fit_params=fit_params  # Pass fit_params to learning_curve
        )

        # Calculate mean and standard deviation
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.title("Learning Curve", fontsize=16)
        plt.xlabel("Number of Training Examples", fontsize=14)
        plt.ylabel("Mean Absolute Error", fontsize=14)

        plt.grid(True)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()

        # Save plot
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'learning_curve.png'))
        plt.close()

        logging.info("Learning curve plot saved.")

    except Exception as e:
        logging.error(f"Error in plot_learning_curve: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_residuals(y_true, y_pred, plot_dir):
    """Plot and save residual plots."""
    try:
        residuals = y_true - y_pred
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Residual Plot
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax1)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=14)
        ax1.grid(True)

        # Residual Distribution
        sns.histplot(residuals, kde=True, bins=50, ax=ax2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_title('Distribution of Residuals', fontsize=14)
        ax2.grid(True)

        plt.suptitle(f'Residual Analysis\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}', fontsize=16)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        residual_plot_path = os.path.join(plot_dir, 'residual_analysis.png')
        fig.savefig(residual_plot_path)
        plt.close(fig)

        logging.info("Residual analysis plot saved.")
    except Exception as e:
        logging.error(f"Error in plot_residuals: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_actual_vs_predicted(y_true, y_pred, plot_dir):
    """Plot and save Actual vs Predicted values."""
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Actual vs Predicted', fontsize=14)
        ax.grid(True)
        
        # Add text box with metrics
        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'actual_vs_predicted.png')
        fig.savefig(plot_path)
        plt.close(fig)
        logging.info(f"Actual vs Predicted plot saved at '{plot_path}'.")
    except Exception as e:
        logging.error(f"Error in plot_actual_vs_predicted: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_average_price_per_month(data, date_column, price_column, plot_dir):
    """Plot and save the average price per month."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")

        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data = data.dropna(subset=[date_column, price_column])
        data['YearMonth'] = data[date_column].dt.to_period('M')
        monthly_avg = data.groupby('YearMonth')[price_column].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_avg, x='YearMonth', y=price_column, marker='o')
        plt.xlabel('Month')
        plt.ylabel(f'Average {price_column}')
        plt.title('Average Price Per Month')
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'average_price_per_month.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Average price per month plot saved at '{plot_path}'.")
    except Exception as e:
        logging.error(f"Error in plot_average_price_per_month: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_cumulative_change(data, date_column, price_column, plot_dir):
    """Plot and save the cumulative change of the average price over time."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")

        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data = data.dropna(subset=[date_column, price_column])
        data['YearMonth'] = data[date_column].dt.to_period('M')
        monthly_avg = data.groupby('YearMonth')[price_column].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].dt.to_timestamp()

        monthly_avg['Cumulative Change (%)'] = (
            (monthly_avg[price_column] / monthly_avg[price_column].iloc[0] - 1) * 100
        )

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_avg, x='YearMonth', y='Cumulative Change (%)', marker='o')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Change (%)')
        plt.title('Cumulative Change of Average Price Over Time')
        plt.axhline(0, color='gray', linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'cumulative_change.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Cumulative change plot saved at '{plot_path}'.")
    except Exception as e:
        logging.error(f"Error in plot_cumulative_change: {e}")
        logging.error(traceback.format_exc())
        raise

def plot_vwap_interactive(data, bed_type, plot_dir):
    """
    Plot an interactive VWAP chart with subtle volume bars, drawing tools, and annotations
    for a specific bedroom type.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import os
        import logging
        import pandas as pd
        import traceback

        # Filter data for the specific bedroom type
        bed_data = data[data['Bed'] == bed_type].copy()
        bed_data.sort_values('Last Rental Date', inplace=True)

        # Ensure required columns are present
        required_columns = [
            'Last Rental Date',
            'VWAP_3M',
            'VWAP_12M',
            'Last Rental Price',
            'Volume'
        ]
        missing_columns = [col for col in required_columns if col not in bed_data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in bed_data for bed type {bed_type}: {missing_columns}")
            return  # Exit the function if columns are missing

        # Create title with additional info
        num_observations = len(bed_data)
        start_date = bed_data['Last Rental Date'].min().date()
        end_date = bed_data['Last Rental Date'].max().date()
        title_text = (
            f"Interactive VWAP with Drawing Tools for {bed_type}-Bedroom Properties<br>"
            f"Number of Observations: {num_observations}, "
            f"Period: {start_date} to {end_date}"
        )

        # Aggregate volume data at desired frequency (e.g., weekly)
        volume_data = (
            bed_data
            .set_index('Last Rental Date')
            .resample('W')
            .agg({'Volume': 'sum'})
            .reset_index()
        )

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Volume Bars on secondary y-axis (plotted first)
        fig.add_trace(go.Bar(
            x=volume_data['Last Rental Date'],
            y=volume_data['Volume'],
            name='Rental Volume',
            marker=dict(color='lightgray'),
            opacity=0.2,
            showlegend=False
        ), secondary_y=True)

        # Add 3-Month VWAP line
        fig.add_trace(go.Scatter(
            x=bed_data['Last Rental Date'],
            y=bed_data['VWAP_3M'],
            mode='lines',
            name='3-Month VWAP',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        # Add 12-Month VWAP line with prominence
        fig.add_trace(go.Scatter(
            x=bed_data['Last Rental Date'],
            y=bed_data['VWAP_12M'],
            mode='lines',
            name='12-Month VWAP',
            line=dict(color='red', width=4)
        ), secondary_y=False)

        # Add Actual Prices as scatter points
        fig.add_trace(go.Scatter(
            x=bed_data['Last Rental Date'],
            y=bed_data['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=5, color='gray', opacity=0.5)
        ), secondary_y=False)

        # Update layout with drawing tools and line styling
        fig.update_layout(
            title=title_text,
            xaxis_title='Date',
            yaxis_title='Rental Price',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],  # Add drawing tools
            newshape=dict(
                line_color='rgba(0, 0, 0, 0.5)',  # Near transparent black
                line_dash='dot'  # Dotted line
            ),
        )

        # Update y-axes properties
        fig.update_yaxes(title_text="Rental Price", secondary_y=False)
        fig.update_yaxes(title_text="Rental Volume", secondary_y=True, showgrid=False)

        # Adjust layering to ensure volume bars are behind other plots
        fig.data = fig.data[::-1]

        # Generate a unique ID for the plot div
        plot_div_id = f'plot_div_{bed_type}_{str(id(fig))}'

        # Save the interactive plot as an HTML file with custom JavaScript
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f'vwap_{bed_type}bed_with_drawing_tools_and_annotations.html'
        plot_path = os.path.join(plot_dir, plot_filename)

        # Generate HTML string
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=plot_div_id)

        # Custom JavaScript to add annotations on drawn lines
        custom_js = f"""
        <script>
        var plot = document.getElementById('{plot_div_id}');

        plot.on('plotly_relayout', function(eventdata){{
            // Check if a new shape has been added
            if(eventdata['shapes']){{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];

                if(shape.type === 'line'){{
                    // Extract coordinates
                    var x0 = shape.x0;
                    var y0 = shape.y0;
                    var x1 = shape.x1;
                    var y1 = shape.y1;

                    // Calculate percentage change
                    var pct_change = y0 !== 0 ? ((y1 - y0) / y0) * 100 : 0;

                    // Format dates
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    // Prepare annotations
                    var newAnnotations = plot.layout.annotations ? plot.layout.annotations.slice() : [];

                    newAnnotations.push(
                        {{
                            x: x0,
                            y: y0,
                            xref: 'x',
                            yref: 'y',
                            text: 'Date: ' + x0_date + '<br>Price: ' + y0.toFixed(2),
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }},
                        {{
                            x: x1,
                            y: y1,
                            xref: 'x',
                            yref: 'y',
                            text: 'Date: ' + x1_date + '<br>Price: ' + y1.toFixed(2) + '<br>Change: ' + pct_change.toFixed(2) + '%',
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }}
                    );

                    // Update the plot
                    Plotly.relayout(plot, {{
                        'annotations': newAnnotations
                    }});
                }}
            }}

            // Remove annotations if shapes are removed
            if(eventdata['shapes[0]'] === null){{
                Plotly.relayout(plot, {{'annotations': []}});
            }}

        }});
        </script>
        """

        # Insert the custom JavaScript before the closing </body> tag
        html_str = html_str.replace('</body>', custom_js + '</body>')

        # Save the modified HTML
        with open(plot_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Interactive VWAP plot with drawing tools and annotations saved for {bed_type}-bedroom: {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_vwap_interactive for bed type {bed_type}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def plot_global_vwap_interactive(data, plot_dir):
    """Plot an interactive global VWAP chart with subtle volume bars, drawing tools, and annotations on drawn lines."""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        import os
        import logging
        import pandas as pd
        import traceback

        # Copy and sort the data
        global_data = data.copy()
        global_data.sort_values('Last Rental Date', inplace=True)

        # Ensure required columns are present
        required_columns = [
            'Last Rental Date',
            'VWAP_3M_all',
            'VWAP_12M_all',
            'Last Rental Price',
            'Volume_all'
        ]
        missing_columns = [col for col in required_columns if col not in global_data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in global_data: {missing_columns}")
            return  # Exit the function if columns are missing

        # Aggregate volume data at desired frequency (e.g., weekly)
        volume_data = (
            global_data
            .set_index('Last Rental Date')
            .resample('W')
            .agg({'Volume_all': 'sum'})
            .reset_index()
        )

        # Create title with additional info
        num_observations = len(global_data)
        start_date = global_data['Last Rental Date'].min().date()
        end_date = global_data['Last Rental Date'].max().date()
        title_text = (
            f"Interactive VWAP with Drawing Tools for All Bedroom Properties Combined<br>"
            f"Number of Observations: {num_observations}, "
            f"Period: {start_date} to {end_date}"
        )

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add Volume Bars on secondary y-axis (plotted first)
        fig.add_trace(go.Bar(
            x=volume_data['Last Rental Date'],
            y=volume_data['Volume_all'],
            name='Rental Volume',
            marker=dict(color='lightgray'),
            opacity=0.2,
            showlegend=False
        ), secondary_y=True)

        # Add VWAP lines and actual prices
        fig.add_trace(go.Scatter(
            x=global_data['Last Rental Date'],
            y=global_data['VWAP_3M_all'],
            mode='lines',
            name='3-Month VWAP (All Beds)',
            line=dict(color='blue', width=2)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=global_data['Last Rental Date'],
            y=global_data['VWAP_12M_all'],
            mode='lines',
            name='12-Month VWAP (All Beds)',
            line=dict(color='red', width=4)
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=global_data['Last Rental Date'],
            y=global_data['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=5, color='gray', opacity=0.5)
        ), secondary_y=False)

        # Update layout
        fig.update_layout(
            title=title_text,
            xaxis_title='Date',
            yaxis_title='Rental Price',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            barmode='overlay',
            modebar_add=["drawline", "eraseshape"],  # Add drawing tools
            newshape=dict(
                line_color='rgba(0, 0, 0, 0.5)',  # Near transparent black
                line_dash='dot'  # Dotted line
            ),
        )

        # Update y-axes properties
        fig.update_yaxes(title_text="Rental Price", secondary_y=False)
        fig.update_yaxes(title_text="Rental Volume", secondary_y=True, showgrid=False)

        # Adjust layering to ensure volume bars are behind other plots
        fig.data = fig.data[::-1]

        # Generate a unique ID for the plot div
        plot_div_id = 'plot_div_' + str(id(fig))

        # Save the interactive plot as an HTML file with custom JavaScript
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = 'vwap_all_beds_with_drawing_tools_and_annotations.html'
        plot_path = os.path.join(plot_dir, plot_filename)

        # Generate HTML string
        html_str = fig.to_html(full_html=True, include_plotlyjs='cdn', div_id=plot_div_id)

        # Custom JavaScript to add annotations on drawn lines
        custom_js = f"""
        <script>
        var plot = document.getElementById('{plot_div_id}');

        plot.on('plotly_relayout', function(eventdata){{
            var update = {{}};

            // Check if a new shape has been added
            if(eventdata['shapes']){{
                var shapeKeys = Object.keys(eventdata['shapes']);
                var lastShapeKey = shapeKeys[shapeKeys.length - 1];
                var shape = eventdata['shapes'][lastShapeKey];

                if(shape.type === 'line'){{
                    // Style the line (already set via newshape in layout)
                    // Extract coordinates
                    var x0 = shape.x0;
                    var y0 = shape.y0;
                    var x1 = shape.x1;
                    var y1 = shape.y1;

                    // Calculate percentage change
                    var pct_change = y0 !== 0 ? ((y1 - y0) / y0) * 100 : 0;

                    // Format dates
                    var x0_date = new Date(x0).toLocaleDateString();
                    var x1_date = new Date(x1).toLocaleDateString();

                    // Prepare annotations
                    var newAnnotations = plot.layout.annotations ? plot.layout.annotations.slice() : [];

                    newAnnotations.push(
                        {{
                            x: x0,
                            y: y0,
                            xref: 'x',
                            yref: 'y',
                            text: 'Date: ' + x0_date + '<br>Price: ' + y0.toFixed(2),
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }},
                        {{
                            x: x1,
                            y: y1,
                            xref: 'x',
                            yref: 'y',
                            text: 'Date: ' + x1_date + '<br>Price: ' + y1.toFixed(2) + '<br>Change: ' + pct_change.toFixed(2) + '%',
                            showarrow: true,
                            arrowhead: 7,
                            ax: -40,
                            ay: -40
                        }}
                    );

                    // Update the plot
                    Plotly.relayout(plot, {{
                        'annotations': newAnnotations
                    }});
                }}
            }}

            // Remove annotations if shapes are removed
            if(eventdata['shapes[0]'] === null){{
                Plotly.relayout(plot, {{'annotations': []}});
            }}

        }});
        </script>
        """

        # Insert the custom JavaScript before the closing </body> tag
        html_str = html_str.replace('</body>', custom_js + '</body>')

        # Save the modified HTML
        with open(plot_path, 'w') as f:
            f.write(html_str)

        logging.info(f"Interactive global VWAP plot with drawing tools and annotations saved: {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_vwap_interactive: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def plot_global_median_interactive(data, plot_dir):
    """Plot an interactive global rolling median chart for all bedroom properties combined."""
    try:
        median_data = data.copy()
        median_data.sort_values('Last Rental Date', inplace=True)

        # Ensure required columns are present
        required_columns = [
            'Last Rental Date',
            'Rolling_Median_1M_all',
            'Rolling_Median_12M_all',
            'Last Rental Price'
        ]
        missing_columns = [col for col in required_columns if col not in median_data.columns]
        if missing_columns:
            logging.error(f"Missing required columns in median_data: {missing_columns}")
            return  # Exit the function if columns are missing

        # Calculate number of observations and time period
        num_observations = len(median_data)
        start_date = median_data['Last Rental Date'].min().date()
        end_date = median_data['Last Rental Date'].max().date()

        # Create title with additional info
        title_text = (
            f"Interactive Rolling Medians for All Bedroom Properties Combined<br>"
            f"Number of Observations: {num_observations}, "
            f"Period: {start_date} to {end_date}"
        )

        fig = go.Figure()

        # Add 1-Month Rolling Median line
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Rolling_Median_1M_all'],
            mode='lines',
            name='1-Month Rolling Median',
            line=dict(color='blue', width=2)
        ))

        # Add 12-Month Rolling Median line with prominence
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Rolling_Median_12M_all'],
            mode='lines',
            name='12-Month Rolling Median',
            line=dict(color='red', width=4)
        ))

        # Add Actual Prices as scatter points
        fig.add_trace(go.Scatter(
            x=median_data['Last Rental Date'],
            y=median_data['Last Rental Price'],
            mode='markers',
            name='Actual Price',
            marker=dict(size=5, color='gray', opacity=0.5)
        ))

        # Update layout
        fig.update_layout(
            title=title_text,
            xaxis_title='Date',
            yaxis_title='Rental Price',
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            ),
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        # Save the interactive plot as an HTML file
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = 'rolling_median_all_beds_interactive.html'
        plot_path = os.path.join(plot_dir, plot_filename)
        fig.write_html(plot_path)

        logging.info(f"Interactive global rolling median plot saved: {plot_path}")

    except Exception as e:
        logging.error(f"Error in plot_global_median_interactive: {str(e)}")
        logging.error(traceback.format_exc())
        raise

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

# prediction.py

import logging
import os
import sys
import traceback
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from config import MODEL_DIR, CLEANED_DATA_PATH
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)

def load_full_pipeline_and_metadata():
    try:
        # Paths to the saved files
        full_pipeline_path = os.path.join(MODEL_DIR, 'full_pipeline.joblib')
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')

        # Check if files exist
        if not os.path.exists(full_pipeline_path):
            logging.error(f"Full pipeline file not found at: {full_pipeline_path}")
            sys.exit(1)
        if not os.path.exists(metadata_path):
            logging.error(f"Metadata file not found at: {metadata_path}")
            sys.exit(1)

        # Load the full pipeline
        full_pipeline = joblib.load(full_pipeline_path)
        logging.info(f"Full pipeline loaded successfully from '{full_pipeline_path}'.")

        # Load the metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info(f"Metadata loaded successfully from '{metadata_path}'.")

        return full_pipeline, metadata

    except Exception as e:
        logging.error(f"Error loading full pipeline and metadata: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def calculate_vwap(data, bed_type, last_rental_date):
    try:
        # Ensure 'Bed' is of string type
        data['Bed'] = data['Bed'].astype(str)

        # Filter data for the given bed_type and before last_rental_date
        bed_data = data[
            (data['Bed'] == bed_type) &
            (data['Last Rental Date'] < last_rental_date)
        ].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        if bed_data.empty:
            logging.warning(f"No historical data available for VWAP calculation for Bed='{bed_type}' before {last_rental_date.date()}.")
            return None, None

        # Log the number of records
        logging.info(f"Number of historical records for Bed='{bed_type}': {len(bed_data)}")

        # Calculate VWAP_3M
        start_date_3M = last_rental_date - pd.DateOffset(months=3)
        vw_data_3M = bed_data[bed_data['Last Rental Date'] >= start_date_3M]
        if len(vw_data_3M) >= 5:
            VWAP_3M = vw_data_3M['Last Rental Price'].mean()
        else:
            VWAP_3M = bed_data['Last Rental Price'].mean()  # Fallback to all data

        # Calculate VWAP_12M
        start_date_12M = last_rental_date - pd.DateOffset(months=12)
        vw_data_12M = bed_data[bed_data['Last Rental Date'] >= start_date_12M]
        if len(vw_data_12M) >= 20:
            VWAP_12M = vw_data_12M['Last Rental Price'].mean()
        else:
            VWAP_12M = bed_data['Last Rental Price'].mean()  # Fallback to all data

        return VWAP_3M, VWAP_12M

    except Exception as e:
        logging.error(f"Error calculating VWAP: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def predict_rental_price(features: pd.DataFrame, full_pipeline, metadata):
    try:
        # Log initial features
        logging.info(f"Initial features:\n{features}\n")

        # Ensure 'Last Rental Date' is in datetime format
        if 'Last Rental Date' in features.columns and not pd.api.types.is_datetime64_any_dtype(features['Last Rental Date']):
            features['Last Rental Date'] = pd.to_datetime(features['Last Rental Date'])

        # Ensure that categorical features are strings
        categorical_features = metadata['cat_features']
        for col in categorical_features:
            if col in features.columns:
                features[col] = features[col].astype(str)
            else:
                features[col] = 'Unknown'  # Handle missing categorical features

        # Make prediction using the full pipeline
        predicted_price = full_pipeline.predict(features)
        return predicted_price[0]

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def display_feature_importances(full_pipeline):
    try:
        # Extract the model from the pipeline
        model = full_pipeline.named_steps['model']

        # Get feature importances from the model (prettified)
        feature_importances = model.get_feature_importance(prettified=True)

        # Log the feature importances
        logging.info("\nModel Feature Importances:")
        logging.info(feature_importances.to_string(index=False))

    except Exception as e:
        logging.error(f"Error displaying feature importances: {str(e)}")
        logging.error(traceback.format_exc())

def main():
    try:
        # Load the full pipeline and metadata
        full_pipeline, metadata = load_full_pipeline_and_metadata()

        # Optionally display feature importances
        display_feature_importances(full_pipeline)

        # Load historical data for VWAP calculation
        historical_data = pd.read_csv(CLEANED_DATA_PATH)
        historical_data['Last Rental Date'] = pd.to_datetime(historical_data['Last Rental Date'])
        historical_data['Last Rental Price'] = pd.to_numeric(historical_data['Last Rental Price'], errors='coerce')
        # Ensure 'Bed' is of string type
        historical_data['Bed'] = historical_data['Bed'].astype(str)

        # Log the date range and unique 'Bed' values
        logging.info(f"Historical data date range: {historical_data['Last Rental Date'].min()} to {historical_data['Last Rental Date'].max()}")
        logging.info(f"Unique 'Bed' values in historical data: {historical_data['Bed'].unique()}")

        # Define base features
        base_features = {
            'Bed': '4',  # Ensure it's a string
            'Bath': 1,
            'Car': 1,
            'Land Size (sqm)': 760,
            'Floor Size (sqm)': 120,
            'Year Built': 1950,
            'Days on Market': 6,
            'Suburb': 'Papanui, CHRISTCHURCH',
            'Active Listing': 'No',
            'Capital Value': None,  # Will be set in loop
            'Last Rental Date': pd.to_datetime('2024-05-01'),
            'VWAP_3M': None,       # Will be calculated
            'VWAP_12M': None,      # Will be calculated
            'Percentage_Diff': None  # Will be calculated
        }

        # Get dynamic VWAP values
        VWAP_3M, VWAP_12M = calculate_vwap(
            historical_data,
            bed_type=base_features['Bed'],
            last_rental_date=base_features['Last Rental Date']
        )

        if VWAP_3M is None or VWAP_12M is None:
            logging.error("Insufficient historical data to calculate VWAP features.")
            sys.exit(1)

        base_features['VWAP_3M'] = VWAP_3M
        base_features['VWAP_12M'] = VWAP_12M

        # Calculate Percentage_Diff
        base_features['Percentage_Diff'] = ((VWAP_3M - VWAP_12M) / VWAP_12M) * 100

        # Define different Capital Values to test
        capital_values = [
            550000,   # -20%
            645000,   # Original
            1000000   # +20%
        ]

        logging.info("\nTesting different Capital Values:")
        for cv in capital_values:
            logging.info(f"\n{'='*50}")
            logging.info(f"Testing Capital Value: ${cv:,.2f}")
            logging.info(f"{'='*50}")

            # Create a copy of the base features and update 'Capital Value'
            features = base_features.copy()
            features['Capital Value'] = cv

            # Create DataFrame for prediction
            example_features = pd.DataFrame([features])

            # Predict rental price
            predicted_price = predict_rental_price(example_features, full_pipeline, metadata)
            logging.info(f"Final predicted rental price: ${predicted_price:.2f}")
            logging.info(f"{'='*50}\n")

    except Exception as e:
        logging.error(f"An error occurred during the prediction process: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()

# config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Data file paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'rental_data.csv')  # Ensure this matches your actual filename
CLEANED_DATA_PATH = os.path.join(INTERIM_DATA_DIR, 'cleaned_rental_data.csv')

# Plot directory
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Cache directory
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# Model saving directory and path
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rental_price_model.joblib')

# Metadata saving path
METADATA_DIR = os.path.join(BASE_DIR, 'metadata')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.joblib')

# Optuna study saving path
STUDIES_DIR = os.path.join(BASE_DIR, 'studies')
OPTUNA_STUDY_PATH = os.path.join(STUDIES_DIR, 'optuna_study.pkl')

# Extra directories for models, metadata, and studies
EXTRA_DIRS = [
    MODEL_DIR,
    PLOT_DIR,
    CACHE_DIR,
    METADATA_DIR,
    STUDIES_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR
]

print(f"MODEL_PATH: {MODEL_PATH}")

# Ensure MODEL_DIR exists
os.makedirs(MODEL_DIR, exist_ok=True)

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from scipy import stats
import plotly.graph_objects as go
from plotting import plot_vwap_interactive, plot_global_vwap_interactive, plot_global_median_interactive
import traceback

def remove_outliers(group):
    z_scores = np.abs(stats.zscore(group['Last Rental Price']))
    return group[z_scores < 4]

def plot_moving_medians(data, bed_type, plot_dir):
    try:
        bed_data = data[data['Bed'] == bed_type].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(bed_data['Last Rental Date'], bed_data['Rolling_Median_3M'], label='3-Month Moving Median')
        plt.plot(bed_data['Last Rental Date'], bed_data['Rolling_Median_12M'], label='12-Month Moving Median')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(bed_data['Last Rental Date'], bed_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title(f'Moving Medians for {bed_type}-Bedroom Properties')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = f'moving_medians_{bed_type}bed.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Moving medians plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_moving_medians for bed type {bed_type}: {str(e)}")
        raise

def plot_vwap(data, bed_type, plot_dir):
    try:
        bed_data = data[data['Bed'] == bed_type].copy()
        bed_data = bed_data.sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(bed_data['Last Rental Date'], bed_data['VWAP_3M'], label='3-Month VWAP')
        plt.plot(bed_data['Last Rental Date'], bed_data['VWAP_12M'], label='12-Month VWAP')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(bed_data['Last Rental Date'], bed_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title(f'VWAP for {bed_type}-Bedroom Properties')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = f'vwap_{bed_type}bed.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"VWAP plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_vwap for bed type {bed_type}: {str(e)}")
        raise

def plot_global_vwap(data, plot_dir):
    """Plot the global (all-bedroom) VWAP"""
    try:
        global_data = data.copy().sort_values('Last Rental Date')

        plt.figure(figsize=(15, 8))
        plt.plot(global_data['Last Rental Date'], global_data['VWAP_3M_all'], label='3-Month VWAP (All Beds)')
        plt.plot(global_data['Last Rental Date'], global_data['VWAP_12M_all'], label='12-Month VWAP (All Beds)')

        # Plot actual prices with reduced alpha and smaller point size
        plt.scatter(global_data['Last Rental Date'], global_data['Last Rental Price'],
                    label='Actual Price', alpha=0.1, s=5, color='gray')

        plt.title('VWAP for All Bedroom Properties Combined')
        plt.xlabel('Date')
        plt.ylabel('Rental Price')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_filename = 'vwap_all_beds.png'
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Global VWAP plot saved: {plot_path}")
    except Exception as e:
        logging.error(f"Error in plot_global_vwap: {str(e)}")
        raise

def prepare_data(data, plot_dir):
    try:
        logging.info("Starting data preparation...")
        data = data.copy()

        # Sort data chronologically
        data = data.sort_values('Last Rental Date').reset_index(drop=True)

        # Calculate rolling VWAP for each bedroom type
        for bed_type in data['Bed'].unique():
            bed_data = data[data['Bed'] == bed_type].copy()
            bed_data = bed_data.set_index('Last Rental Date').sort_index()

            # Assume each rental has a volume of 1
            bed_data['Volume'] = 1

            # Calculate rolling sum of price * volume and volume for 3M and 12M
            bed_data['Rolling_PV_3M'] = (bed_data['Last Rental Price'] * bed_data['Volume']).rolling(
                window='90D', min_periods=5).sum()
            bed_data['Rolling_V_3M'] = bed_data['Volume'].rolling(window='90D', min_periods=5).sum()

            bed_data['Rolling_PV_12M'] = (bed_data['Last Rental Price'] * bed_data['Volume']).rolling(
                window='365D', min_periods=20).sum()
            bed_data['Rolling_V_12M'] = bed_data['Volume'].rolling(window='365D', min_periods=20).sum()

            # Calculate VWAP
            bed_data['VWAP_3M'] = bed_data['Rolling_PV_3M'] / bed_data['Rolling_V_3M']
            bed_data['VWAP_12M'] = bed_data['Rolling_PV_12M'] / bed_data['Rolling_V_12M']

            # Shift the VWAP to avoid look-ahead bias
            bed_data['VWAP_3M'] = bed_data['VWAP_3M'].shift(1)
            bed_data['VWAP_12M'] = bed_data['VWAP_12M'].shift(1)

            # Assign back to main dataframe
            data.loc[data['Bed'] == bed_type, 'Volume'] = bed_data['Volume'].values
            data.loc[data['Bed'] == bed_type, 'VWAP_3M'] = bed_data['VWAP_3M'].values
            data.loc[data['Bed'] == bed_type, 'VWAP_12M'] = bed_data['VWAP_12M'].values

        # After assigning VWAP values back to the main dataframe for each bed type
        # Now calculate the VWAP for all combined bedroom types
        data = data.set_index('Last Rental Date').sort_index()
        data['Volume_all'] = 1

        data['Rolling_PV_3M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='90D', min_periods=5).sum()
        data['Rolling_V_3M_all'] = data['Volume_all'].rolling(window='90D', min_periods=5).sum()

        data['Rolling_PV_12M_all'] = (data['Last Rental Price'] * data['Volume_all']).rolling(
            window='365D', min_periods=20).sum()
        data['Rolling_V_12M_all'] = data['Volume_all'].rolling(window='365D', min_periods=20).sum()

        data['VWAP_3M_all'] = data['Rolling_PV_3M_all'] / data['Rolling_V_3M_all']
        data['VWAP_12M_all'] = data['Rolling_PV_12M_all'] / data['Rolling_V_12M_all']

        # Shift the global VWAP to avoid look-ahead bias
        data['VWAP_3M_all'] = data['VWAP_3M_all'].shift(1)
        data['VWAP_12M_all'] = data['VWAP_12M_all'].shift(1)

        # Revert index to column form
        data.reset_index(inplace=True)

        # Remove rows with NaN values in VWAP columns
        data.dropna(subset=['VWAP_3M', 'VWAP_12M'], inplace=True)
        data.dropna(subset=['VWAP_3M_all', 'VWAP_12M_all'], inplace=True)

        # Calculate the percentage differences
        data['Percentage_Diff'] = ((data['VWAP_3M'] - data['VWAP_12M']) / data['VWAP_12M']) * 100
        data['Percentage_Diff'] = data['Percentage_Diff'].replace([np.inf, -np.inf], np.nan).fillna(0)

        data['Percentage_Diff_all'] = ((data['VWAP_3M_all'] - data['VWAP_12M_all']) / data['VWAP_12M_all']) * 100
        data['Percentage_Diff_all'] = data['Percentage_Diff_all'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Plot VWAP for different bedroom types
        for bed_type in data['Bed'].unique():
            logging.info(f"Processing {bed_type}-bedroom properties...")

            bed_data = data[data['Bed'] == bed_type].copy()
            if bed_data.empty:
                logging.warning(f"No data available for {bed_type}-bedroom properties after preprocessing. Skipping plots.")
                continue  # Skip to the next bedroom type

            logging.info(f"Plotting VWAP for {bed_type}-bedroom properties...")
            plot_vwap(bed_data, bed_type, plot_dir)

            logging.info(f"Plotting interactive VWAP for {bed_type}-bedroom properties...")
            plot_vwap_interactive(bed_data, bed_type, plot_dir)

        # Plot global VWAP
        plot_global_vwap(data, plot_dir)
        plot_global_vwap_interactive(data, plot_dir)

        # Create Quarterly Period column
        data['Quarterly Period'] = pd.to_datetime(data['Last Rental Date']).dt.to_period('Q').astype(str)

        # Ensure 'Last Rental Date' is datetime and set as index
        data['Last Rental Date'] = pd.to_datetime(data['Last Rental Date'])
        data.set_index('Last Rental Date', inplace=True)
        data.sort_index(inplace=True)

        # Calculate global rolling medians for 1M and 12M
        data['Rolling_Median_1M_all'] = data['Last Rental Price'].rolling(window='30D', min_periods=1).median()
        data['Rolling_Median_12M_all'] = data['Last Rental Price'].rolling(window='365D', min_periods=5).median()

        # Optionally remove the shift if not needed
        # data['Rolling_Median_1M_all'] = data['Rolling_Median_1M_all'].shift(1)
        # data['Rolling_Median_12M_all'] = data['Rolling_Median_12M_all'].shift(1)

        # Reset index to bring 'Last Rental Date' back as a column
        data.reset_index(inplace=True)

        # Remove rows with NaN values in rolling median columns
        data.dropna(subset=['Rolling_Median_1M_all', 'Rolling_Median_12M_all'], inplace=True)

        # Plot global rolling median
        plot_global_median_interactive(data, plot_dir)

        logging.info("Data preparation completed successfully.")
        logging.info(f"Final data shape: {data.shape}")
        logging.info(f"Final columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        logging.error(f"Error in prepare_data: {str(e)}")
        logging.error(traceback.format_exc())
        raise
