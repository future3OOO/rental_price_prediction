import logging
import traceback
import joblib
import pandas as pd
import numpy as np
import json
import os
import shap
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, learning_curve, train_test_split
from transformers import DateTransformer
from feature_engineering import FeatureEngineering
from data_cleaning import data_cleaning
from config import MODEL_DIR, PLOT_DIR
from plotting import (
    plot_feature_importance,
    plot_residuals,
    plot_actual_vs_predicted
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import Memory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

memory = Memory(location='cache_dir', verbose=0)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0
    if not np.any(nonzero_indices):
        return np.inf
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

def make_prediction_with_range(full_pipeline, X_new, mae):
    prediction = full_pipeline.predict(X_new)[0]
    margin = mae
    lower_bound = max(prediction - margin, 0)
    upper_bound = prediction + margin
    logging.info(f"Prediction: {prediction}, Lower: {lower_bound}, Upper: {upper_bound}")
    return prediction, lower_bound, upper_bound

@memory.cache
def preprocess_data(data, numeric_features, categorical_features, date_column, target_column):
    # Convert numeric features
    for col in numeric_features + [target_column]:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float32)
        data[col] = data[col].fillna(data[col].median())

    # Ensure categorical features
    for col in categorical_features:
        if col not in data.columns:
            data[col] = 'Unknown'
        data[col] = data[col].fillna('Unknown').astype(str)

    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data = data.sort_values(date_column).reset_index(drop=True)

    return data

def train_model(data):
    try:
        logging.info("Starting model training with Optuna hyperparameter tuning...")

        target_column = 'Last Rental Price'
        date_column = 'Last Rental Date'

        # Features (no interactions for simplicity)
        final_numeric_features = [
            'Bath', 'Car', 'Land Size (sqm)', 'Floor Size (sqm)',
            'Year Built', 'Days on Market', 'Land Value', 'Capital Value', 'VWAP_3M'
        ]
        final_categorical_features = ['Active Listing', 'Suburb', 'Bed']

        # Add date features
        date_features = ['Month_sin', 'Month_cos', 'Time_Index']

        data = preprocess_data(
            data=data,
            numeric_features=final_numeric_features,
            categorical_features=final_categorical_features,
            date_column=date_column,
            target_column=target_column
        )

        # Split data (train/val/test via time-based split)
        dates = data[date_column]
        cutoff_date = dates.sort_values().iloc[int(len(dates)*0.8)]
        train_mask = dates <= cutoff_date

        X = data[final_numeric_features + final_categorical_features + [date_column]].copy()
        y = data[target_column]

        X_train = X[train_mask].copy()
        X_test = X[~train_mask].copy()
        y_train = y[train_mask]
        y_test = y[~train_mask]

        # Build pipeline
        preprocessing_pipeline = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ]), final_numeric_features + date_features),
            ('cat', 'passthrough', final_categorical_features)
        ], remainder='drop')

        full_pipeline = Pipeline(steps=[
            ('date_transformer', DateTransformer(date_column=date_column, drop_original=True)),
            ('feature_engineering', FeatureEngineering(
                numeric_features=final_numeric_features,
                categorical_features=final_categorical_features,
                interaction_terms=None  # no interactions for now
            )),
            ('preprocessor', preprocessing_pipeline)
        ])

        # Transform once to know cat_features indices
        X_train_transformed = full_pipeline.fit_transform(X_train, y_train)
        X_test_transformed = full_pipeline.transform(X_test)

        total_features = X_train_transformed.shape[1]
        numeric_count = len(final_numeric_features) + len(date_features)
        cat_count = len(final_categorical_features)
        if numeric_count + cat_count != total_features:
            raise ValueError("Feature count mismatch. Check columns.")

        cat_features_indices = list(range(numeric_count, numeric_count + cat_count))

        # Optuna hyperparameter tuning
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                 'task_type': 'GPU',  # use GPU for speed
            'devices': '0',
            'gpu_ram_part': 0.5,  # limit to 70% GPU RAM
            'thread_count': 4,    # limit CPU threads to ~50% usage
            'loss_function': 'MAE' # limit CPU usage
            }

            model = CatBoostRegressor(
                **params,
                cat_features=cat_features_indices,
                random_seed=42,
                verbose=0
            )

            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
                X_train_transformed, y_train, test_size=0.2, shuffle=False
            )
            model.fit(
                X_train_sub, y_train_sub,
                eval_set=(X_val_sub, y_val_sub),
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=0
            )

            y_val_pred = model.predict(X_val_sub)
            mae_val = mean_absolute_error(y_val_sub, y_val_pred)
            return mae_val

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)
        best_params = study.best_params

        # Train final model on full training set with best_params
        model = CatBoostRegressor(
            **best_params,
            cat_features=cat_features_indices,
            random_seed=42,
            verbose=100
        )
        model.fit(
            X_train_transformed, y_train,
            eval_set=(X_test_transformed, y_test),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=100
        )

        y_pred = model.predict(X_test_transformed)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = calculate_mape(y_test, y_pred)

        final_feature_names = final_numeric_features + date_features + final_categorical_features

        # Optional learning curve check with controlled parallelism
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train_transformed,
            y=y_train,
            cv=TimeSeriesSplit(n_splits=2),
            n_jobs=2,  # controlled parallelism
            train_sizes=np.linspace(0.1, 1.0, 3),
            scoring='neg_mean_absolute_error'
        )

        plot_feature_importance(
            model=model,
            feature_names=final_feature_names,
            plot_dir=PLOT_DIR
        )

        plot_residuals(y_test, y_pred, PLOT_DIR)
        plot_actual_vs_predicted(y_test, y_pred, PLOT_DIR)

        # SHAP on small sample to help model refinement without huge resource use
        explainer = shap.TreeExplainer(model)
        sample_size = min(50, X_train_transformed.shape[0])
        shap_values = explainer.shap_values(X_train_transformed[:sample_size])
        try:
            shap.summary_plot(shap_values, pd.DataFrame(X_train_transformed[:sample_size], columns=final_feature_names), show=False)
            plt.savefig(os.path.join(PLOT_DIR, 'shap_summary.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting SHAP summary: {e}")

        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'catboost_model.cbm')
        model.save_model(model_path)

        preprocessing_pipeline_path = os.path.join(MODEL_DIR, 'preprocessing_pipeline.joblib')
        joblib.dump(full_pipeline, preprocessing_pipeline_path)

        full_pipeline_with_model = Pipeline([
            ('preprocessor', full_pipeline),
            ('model', model)
        ])

        full_pipeline_path = os.path.join(MODEL_DIR, 'full_pipeline.joblib')
        joblib.dump(full_pipeline_with_model, full_pipeline_path)

        metadata = {
            'MAE': float(mae),
            'R2_Score': float(r2),
            'MAPE': float(mape),
            'feature_names': final_feature_names,
            'best_params': best_params,
            'cat_features_indices': cat_features_indices,
            'date_column': date_column,
            'numeric_features': final_numeric_features,
            'categorical_features': final_categorical_features
        }

        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logging.info("Model training and evaluation completed successfully.")
        return {
            'model': model,
            'preprocessing_pipeline': full_pipeline,
            'full_pipeline': full_pipeline_with_model,
            'metadata': metadata,
            'numeric_features': final_numeric_features,
            'categorical_features': final_categorical_features,
            'feature_names': final_feature_names,
            'date_column': date_column
        }

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        logging.error(traceback.format_exc())
        raise
