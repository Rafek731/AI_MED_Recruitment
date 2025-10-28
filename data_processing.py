import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_float_t = np.float64

def preprocess_data(data: pd.DataFrame, label_col: str, feature_cols: list[str]|None = None, test_size=0.2) -> tuple[np.ndarray]:
    """Preprocess a pandas DataFrame and return train / test numpy arrays.

    The function prepares data by selecting feature columns, converting 
    string numeric values (with comma decimals) to floats, min-max normalizing 
    each column to [0, 1], and splitting the dataset into train and test sets.

    Note:
    - ``feature_cols`` should be a list of column names to use as features. 
    If ``feature_cols`` is not a list (or is None) the function will default to
    ``list(data.columns)``.
    - The function may modify the input DataFrame (due to column
    assignments). If you need to preserve the original pass
    ``data.copy()``.

    Args:
        data (pd.DataFrame): Input table containing feature columns and the label column.
        label_col (str): Name of the column to use as the target labels.
        feature_cols (list[str] | None): List of feature column names to
            include. If None or not a list, defaults to ``data.columns``.
        test_size (float | int, optional): Fraction from range [0, 1] to reserve 
        for the test set. Defaults to 0.2.

    Returns:
        tuple[np.ndarray]: ``(X_train, y_train, X_test, y_test)``.

    Raises:
        ValueError: If ``label_col`` is not a string or is not present in ``data``.

    Example:
        >>> df = pd.DataFrame({
        ...     'age': ['20,0', '30,0', '40,0'],
        ...     'score': ['0,5', '0,7', '0,2'],
        ...     'label': [0, 1, 0]
        ... })
        >>> X_train, y_train, X_test, y_test = preprocess_data(df, 'label', feature_cols=['age','score'], test_size=0.33)
    """

    if not isinstance(feature_cols, list):
        feature_cols = data.columns.tolist()

    if label_col in feature_cols:
        feature_cols.remove(label_col)

    if not isinstance(label_col, str):
        raise ValueError("'label_col' must be a string representing the column name.")
    
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    

    # choose only specified feature columns and label column
    data = data[feature_cols + [label_col]]
    # Convert strings to floats
    for col_name in data:
        new_col = []
        for val in data[col_name]:
            if isinstance(val, str):
                val = val.replace(',', '.')
            new_col.append(_float_t(val))
        # Apply regularization of data (scale them down to range [0,1])
        new_col = (np.array(new_col) - np.min(new_col)) / (np.max(new_col) - np.min(new_col))
        data[col_name] = new_col

    X_train, X_test = train_test_split(data, test_size=test_size)
    y_train, y_test = X_train[label_col], X_test[label_col]

    X_train.drop(label_col, axis=1, inplace=True)
    X_test.drop(label_col, axis=1, inplace=True)

    return np.array(X_train, dtype=_float_t), np.array(y_train, dtype=int), np.array(X_test, dtype=_float_t), np.array(y_test, dtype=int)
    
