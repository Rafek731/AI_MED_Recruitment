import numpy as np
from collections import Counter

from .Model import Model
from .DecisionTree import DecisionTree

class RandomForest(Model):
    def __init__(self, 
                 n_trees:int = 5,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int|None = None,
                 *,
                 name: str = 'RandomForest'):
        super().__init__(name)
        self._n_trees = n_trees
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._n_features = n_features
        self._forest = []
    
    def fit(self, X: np.ndarray, y: np.ndarray,*, n_trees: int = 5, sample_size: float=0.8) -> None:
        n_samples = round(X.shape[0] * sample_size)
        for _ in range(n_trees):
            tree = DecisionTree(self._min_samples_split, self._max_depth, self._n_features)
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree.fit(X[idxs], y[idxs])
            self._forest.append(tree)
        
    def predict(self, features: np.ndarray) -> int:
        return Counter([tree.predict(features) for tree in self._forest]).most_common(1)[0][0]
