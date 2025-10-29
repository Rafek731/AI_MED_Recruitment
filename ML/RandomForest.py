import numpy as np
from collections import Counter

from .Model import Model
from .DecisionTree import DecisionTree

class RandomForest(Model):
    """Random forest classifier composed of multiple DecisionTree instances.

    This is a lightweight implementation of a bagging ensemble where each
    tree is trained on a bootstrap sample of the training data. Predictions
    are made by majority vote among the trees.
    """
    def __init__(self, 
                 n_trees:int = 5,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int|None = None,
                 *,
                 name: str = 'RandomForest'):
        """Create a RandomForest instance.

        Args:
            n_trees (int): Default number of trees to create when fitting.
            min_samples_split (int): Minimum samples required to split a
                node in each decision tree.
            max_depth (int): Maximum depth for each decision tree.
            n_features (int | None): Number of features to consider when
                looking for the best split in each tree. If None, the value
                will be determined per-tree based on the input data during
                fitting.
            name (str): Optional name for the model used in evaluation reporting.

        Attributes created:
            _forest (list[DecisionTree]): List that will hold trained trees
                after calling ``fit``.
        """

        super().__init__(name)
        self._n_trees = n_trees
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._n_features = n_features
        self._forest = []

    def fit(self, X: np.ndarray, y: np.ndarray,*, n_trees: int = 5, sample_size: float=0.8) -> None:
        """Train the random forest on the provided dataset.

        Args:
            X (np.ndarray): 2-D array of shape (n_samples, n_features).
            y (np.ndarray): 1-D array of length n_samples with integer
                class labels.
            n_trees (int, optional): Number of trees to train. Defaults to 5.
            sample_size (float, optional): Fraction of training samples to use
                for each tree's bootstrap sample. Must be in (0, 1]. Defaults to 0.8.
        """

        n_samples = round(X.shape[0] * sample_size)
        for _ in range(n_trees):
            tree = DecisionTree(self._min_samples_split, self._max_depth, self._n_features)
            # bootstrap sampling (with replacement) from the original dataset
            idxs = np.random.choice(X.shape[0], n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self._forest.append(tree)
        
    def predict(self, features: np.ndarray) -> int:
        return Counter([tree.predict(features) for tree in self._forest]).most_common(1)[0][0]
    
    def clear(self) -> None:
        self._forest.clear()
