import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_float_t = np.float64

def preprocess_data(data: pd.DataFrame, columns: list[str]|None = None) -> pd.DataFrame:
    """Select, convert and min-max normalize columns from a DataFrame.

    The function performs a lightweight preprocessing of the provided
    DataFrame and returns a new DataFrame with the selected columns where
    every value has been converted to a float (strings with a comma decimal
    separator are supported) and each column has been min-max scaled to the
    range [0, 1].

    Behaviour summary:
      - If ``columns`` is not a list, the function defaults to all columns
        found in ``data``.
      - The function selects only the requested columns from ``data`` and
        works on that selection (it does not modify the original DataFrame
        object passed in).
      - For each selected column, string values like "1,23" are converted to
        floats by replacing the comma with a dot, then casting to
        ``np.float64``. After conversion each column is scaled using
        min-max normalization: (x - min) / (max - min).

    Args:
        data (pd.DataFrame): Input table containing columns to preprocess.
        columns (list[str] | None): List of column names to select and
            preprocess. If None or not a list, defaults to all columns in ``data``.

    Returns:
        pd.DataFrame: A new DataFrame containing the selected columns with
            values converted to ``data_processing._float_t`` and min-max normalized to [0, 1].

    Raises:
        ValueError: If a value cannot be converted to float during casting
            (for example, non-numeric strings).

    Example:
        >>> df = pd.DataFrame({
        ...     'a': ['1,0', '2,0'],
        ...     'b': ['0,5', '0,7']
        ... })
        >>> preprocess_data(df, ['a','b'])
             a    b
        0  0.0  0.0
        1  1.0  1.0
    """

    if not isinstance(columns, list):
        columns = data.columns.tolist()

    # choose only specified columns
    data = data[columns]
    # Convert strings to floats
    for col_name in data.columns:
        new_col = []
        for val in data[col_name]:
            if isinstance(val, str):
                val = val.replace(',', '.')
            new_col.append(_float_t(val))
        # Apply regularization of data (scale them down to range [0,1])
        new_col = (np.array(new_col) - np.min(new_col)) / (np.max(new_col) - np.min(new_col))
        data[col_name] = new_col

    return data

    
def split_data(data: pd.DataFrame, label_col: str, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets.

    This function splits the provided DataFrame into training and testing
    sets based on the specified test size. The label column is separated from
    the feature columns.

    Args:
        data (pd.DataFrame): Input DataFrame containing features and labels.
        label_col (str): Name of the column containing labels.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            training features, training labels, testing features, and testing labels.

    Raises:
        ValueError: If the label column does not exist in the DataFrame.
    """

    if label_col not in data.columns:
        raise ValueError(f"Label column '{label_col}' does not exist in the DataFrame.")

    X_train, X_test = train_test_split(data, test_size=test_size)
    y_train, y_test = X_train[label_col], X_test[label_col]

    X_train.drop(label_col, axis=1, inplace=True)
    X_test.drop(label_col, axis=1, inplace=True)

    return np.array(X_train, dtype=_float_t), np.array(y_train, dtype=int), np.array(X_test, dtype=_float_t), np.array(y_test, dtype=int)