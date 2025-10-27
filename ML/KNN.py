import numpy as np
from collections import Counter

from .Model import Model

class KNN_classifier(Model):
    """k-Nearest Neighbors classifier.

    A minimal KNN implemented from scratch that stores the training data and
    predicts the label of a single sample by majority vote among the k
    nearest training samples (Euclidean distance).

    Attributes:
        X (np.ndarray | None): Training feature matrix of shape
            (n_samples, n_features) set by ``fit``.
        y (np.ndarray | None): Training labels of length n_samples set by ``fit``.
        k (int): Number of neighbors to consider for prediction.
    """

    def __init__(self, k: int = 5):
        """Create a KNN classifier.

        Args:
            k (int, optional): Number of neighbors to use for majority vote.
                Must be a positive integer. Defaults to 5.
        """
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store training data for later nearest-neighbour queries.

        This method does not perform any learning; it simply stores the
        training feature matrix ``X`` and label vector ``y``. Both arrays are
        assumed to be aligned (same first-dimension length).

        Args:
            X (np.ndarray): 2-D array with training features, shape
                (n_samples, n_features).
            y (np.ndarray): 1-D array with integer labels of length n_samples.

        Raises:
            ValueError: If X and y have incompatible lengths.
        """

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        self.X = X
        self.y = y
        

    def predict(self, features: np.ndarray) -> int:
        """Predict the class label for a single feature vector.

        The method computes (squared) Euclidean distances between ``features``
        and each training sample, selects the ``k`` nearest neighbours, and
        returns the most common label among them.

        Args:
            features (np.ndarray): 1-D array of length n_features for a single
                sample.

        Returns:
            int: Predicted class label (an integer).

        Raises:
            ValueError: If the classifier has not been fitted (``X`` or ``y`` is None) or if ``k`` is not set.
        """

        if self.X is None or self.y is None or self.k is None:
            raise ValueError("Classifier is not fitted or 'k' is not set")

        # Since sqrt is monotonic, squared distances are sufficient for ranking
        distances = [(np.sum((X - features) ** 2), y) for X, y in zip(self.X, self.y)]
        # Sort pairs by distance (ascending)
        distances.sort(key=lambda x: x[0])
        # Extract the labels of the k nearest neighbours
        k_nearest_labels = [label for _, label in distances[: self.k]]
        # Return the most common label
        counter = Counter(k_nearest_labels)
        return counter.most_common(1)[0][0]
