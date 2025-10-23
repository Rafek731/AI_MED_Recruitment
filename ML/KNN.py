import numpy as np
from collections import Counter

from .Model import Model

class KNN_classifier(Model):
    def __init__(self, k: int=5):
        self.k = k


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        

    def predict(self, features: np.ndarray) -> int:
        """Calculate distances of the points to the given vector of features (point)"""
        # Since sqrt(x) is increasing we can just calculate sqared distance and it won't matter in sorting
        distances = [(np.sum((X - features)**2), y) for X, y in zip(self.X, self.y)]
        # Sort them in ascending order
        distances.sort(key=lambda x: x[0])
        # return the most common label
        counter = Counter(np.array(distances, dtype=int)[:self.k, 1])
        return counter.most_common(1)[0][0]
