# first line: 62
@memory.cache
def preprocess_data(data, numeric_features, categorical_features, date_column, target_column):
    """
    Basic numeric/categorical handling & sort by date.
    """
    for col in numeric_features + [target_column]:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float32)
        data[col].fillna(data[col].median(), inplace=True)

    for col in categorical_features:
        if col not in data.columns:
            data[col] = 'Unknown'
        data[col] = data[col].fillna('Unknown').astype(str)

    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data.sort_values(date_column, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data
