# test_sklearn.py

import sklearn
from sklearn.compose import ColumnTransformer

print(f"scikit-learn version: {sklearn.__version__}")

try:
    ct = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['A']),
            ('cat', 'passthrough', ['B'])
        ],
        remainder='drop',
        transform_output='pandas'  # This should work in scikit-learn 1.2.2
    )
    print("ColumnTransformer with 'transform_output' initialized successfully.")
except TypeError as e:
    print(f"Error: {e}")
