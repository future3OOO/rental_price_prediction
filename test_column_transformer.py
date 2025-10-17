# test_column_transformer.py

import sklearn
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

print(f"scikit-learn version: {sklearn.__version__}")

try:
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Use sparse_output instead of sparse
    ])

    # Initialize ColumnTransformer without 'transform_output'
    ct = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['A']),
            ('cat', categorical_transformer, ['B'])
        ],
        remainder='drop',
        verbose_feature_names_out=True
    )
    print("ColumnTransformer initialized successfully.")
    
    # Example data
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4, 5, 6]
    })
    
    # Transform the data
    transformed_data = ct.fit_transform(data)
    
    # Convert to pandas DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=ct.get_feature_names_out())
    print(transformed_df)
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
