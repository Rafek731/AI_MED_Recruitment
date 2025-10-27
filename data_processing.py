import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_float_t = np.float64

def preprocess_data(data: pd.DataFrame, label_col: str, drop_labels: list[str]|None = None, test_size=0.2) -> tuple[np.ndarray]:
    """Preprocess a DataFrame and return a train/test split as numpy arrays.

    This function performs the following steps in-place on the provided
    DataFrame `data`:
    - Drops any columns listed in `drop_labels` (if provided).
    - Converts values that are strings (for example, "1,23") into floats by
      replacing comma decimal separators with a dot, then casting to float.
    - Scales each column to the range [0, 1] using min-max normalization.
    - Splits the DataFrame into train and test subsets using
      `sklearn.model_selection.train_test_split`.
    - Separates the label column (specified by `label_col`) from features and
      returns (X_train, y_train, X_test, y_test) as numpy arrays.

    Note: The input `data` is mutated in-place (columns are dropped and values
    replaced). If you need to preserve the original DataFrame, pass a copy
    (e.g. `data.copy()`).

    Args:
        data (pd.DataFrame): Input table containing features and the label column.
        label_col (str): Name of the column in `data` to use as the target
            variable. This column will be removed from the returned feature
            arrays.
        drop_labels (list[str] | None, optional): Column names to drop before
            processing (for example, IDs or metadata). Defaults to None.
        test_size (float | int, optional): If float between 0 and 1, fraction of
            data to reserve for the test set. If int, number of samples for the
            test set. Defaults to 0.2.

    Returns:
        tuple[np.ndarray]: A 4-tuple with the following numpy arrays and dtypes
            already applied:
            - X_train (np.ndarray, dtype=float64): Training features, shape (n_train, n_features).
            - y_train (np.ndarray, dtype=int): Training labels, shape (n_train,).
            - X_test (np.ndarray, dtype=float64): Test features, shape (n_test, n_features).
            - y_test (np.ndarray, dtype=int): Test labels, shape (n_test,).

    Raises:
        ValueError: If `label_col` is not present in `data`.

    Example:
        >>> df = pd.DataFrame({
        ...     'feat': ['1,0', '2,0', '3,0'],
        ...     'label': [0, 1, 0]
        ... })
        >>> X_train, y_train, X_test, y_test = preprocess_data(df, 'label', test_size=0.33)

    """

    if drop_labels is None or not isinstance(drop_labels, list):
        drop_labels = []

    if not isinstance(label_col, str):
        raise ValueError("'label_col' must be a string representing the column name.")
    
    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    
    # drop useless data
    data.drop(labels=drop_labels , axis=1, inplace=True)
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
    
