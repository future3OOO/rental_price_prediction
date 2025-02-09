# Importing necessary libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import optuna

# Load and prepare data
file_path = r'C:\Users\Property Partner\Desktop\propert partner\Dev work\Rental project\data\interim\cleaned_rental_data.csv'
df = pd.read_csv(file_path)

features = ['Bed', 'Bath', 'Car', 'Land Size (sqm)', 'Year Built', 'Capital Value']
target = 'Last Rental Price'

df_subset = df[features + [target]].copy()
df_subset.fillna(0, inplace=True)

X = df_subset[features]
y = df_subset[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial XGBoost model
initial_params = {
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10,
    'n_estimators': 100
}

xg_reg = xgb.XGBRegressor(**initial_params)
xg_reg.fit(X_train, y_train)

# Function to get leaf values for a single data point
def get_leaf_values(model, X_single):
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_single)
    
    leaf_indices = booster.predict(dmat, pred_leaf=True)
    
    leaf_values = []
    trees = booster.get_dump(dump_format='json')
    
    for tree_id, leaf_index in enumerate(leaf_indices[0]):
        tree = json.loads(trees[tree_id])
        
        def find_leaf_value(node, leaf_index):
            if 'leaf' in node:
                return node['leaf']
            if leaf_index >= node['split_condition']:
                return find_leaf_value(node['children'][1], leaf_index)
            return find_leaf_value(node['children'][0], leaf_index)
        
        leaf_value = find_leaf_value(tree, leaf_index)
        leaf_values.append(leaf_value)
    
    return leaf_values

# Select a sample data point
sample_index = 0
sample_X = X_test.iloc[[sample_index]]
sample_y = y_test.iloc[sample_index]

# Get leaf values for the sample
leaf_values = get_leaf_values(xg_reg, sample_X)

# Calculate initial prediction (mean of target variable)
initial_prediction = y_train.mean()

print(f"Initial prediction (mean of target): {initial_prediction:.4f}")

# Print leaf values and their cumulative effect
print("\nLeaf values and cumulative predictions:")
cumulative_sum = initial_prediction
for i, value in enumerate(leaf_values):
    cumulative_sum += initial_params['learning_rate'] * value
    print(f"Tree {i+1}: Raw leaf value: {value:.4f} {'(Negative)' if value < 0 else ''}, "
          f"Scaled contribution: {initial_params['learning_rate'] * value:.4f}, "
          f"Cumulative prediction: {cumulative_sum:.4f}")

# Optuna objective function
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_trial.params

print('\nBest parameters found by Optuna:')
for key, value in best_params.items():
    print(f'    {key}: {value}')

# Train final model with best parameters
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error on test set: {mse}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(final_model)
plt.title('Feature Importance in Optimized XGBoost Model')
plt.show()

# Visualize a single tree (e.g., the first tree)
plt.figure(figsize=(20, 10))
xgb.plot_tree(final_model, num_trees=0)
plt.title('First Tree in Optimized XGBoost Model')
plt.show()