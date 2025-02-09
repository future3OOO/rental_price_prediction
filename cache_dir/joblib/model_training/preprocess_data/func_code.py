# first line: 49
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
