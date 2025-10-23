import numpy as np
from collections import Counter

from .Model import Model

class Node:
    def __init__(self,
                 feature: int = None, 
                 threshold: float = None,
                 left=None,
                 right=None,*,
                 value=None):
        
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTree(Model):
    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int|None = None):
        super().__init__()
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._n_features = n_features
        self._root=None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits DecisionTree to given data

        Args:
            X (np.ndarray): train data
            y (np.ndarray): train labels
        """
        if self._n_features is None or self._n_features < 1:
            self._n_features = X.shape[1]
        else:
            self._n_features = min(self._n_features, X.shape[1])
        
        self.root = self._grow_tree(X, y)
        

    def predict(self, features) -> int:
        """Predicts label to given data

        Args:
            features (np.ndarray): vector of features

        Returns:
            int: label of prediction
        """
        return self._traverse_tree(features, self.root)


    def _traverse_tree(self, x, node: Node) -> int:
        """Traverses the tree in search for right answear

        Args:
            x (np.ndarray): vector of features
            node (Node): node to start traversing (recusively)

        Returns:
            int: label
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> Node:
        """Recursively grows tree

        Args:
            X (np.ndarray): feature vector
            y (np.ndarray): labels vector
            depth (int, optional): Variable tracking the current depth of the tree. Defaults to 0.

        Returns:
            Node: Current node from which rest of the tree grows
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if depth >= self._max_depth or n_labels == 1 or n_samples < self._min_samples_split:
            # If this condition is satisfied then we're in leaf node and we have to calculate its value
            return Node(value=self._most_common_label(y))
        
        # Add a bit of randomness
        feature_idxs = np.random.choice(n_features, self._n_features, replace=False)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_idxs)
        
        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)


    def _most_common_label(self, y: np.ndarray) -> int:
        """Returns most common label in given array y

        Args:
            y (np.ndarray): array of labels

        Returns:
            int: most common label
        """
        return Counter(y).most_common(1)[0][0]


    def _best_split(self, X: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> tuple[np.ndarray, float]:
        """Calculates indexes for best split of the given feature vector

        Args:
            X (np.ndarray): feature vector
            y (np.ndarray): corresponding labels vector
            feature_idxs (n.ndarray): chosen feature indexes to split by

        Returns:
            tuple: 
            (`split_idx`, `split_threshold`) - 
            `split_idx` = index of feature to choose for splitting, 
            `split_threshold` = threshold to split by 
        """
        # Initialize variables
        best_gain: float = -1.
        split_idx, split_threshold = None, None

        # Check best gain for each feature among the chosen ones and update variables
        for feature_idx in feature_idxs:
            X_column: np.ndarray = X[:, feature_idx]
            thresholds: np.ndarray = np.unique(X_column)

            for threshold in thresholds:
                # Calculate the information gain
                gain = self._information_gain(X_column, y, threshold)

                # if current gain is better than the current gain update variables
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold




    def _information_gain(self, X_column: np.ndarray, y: np.ndarray, threshold: float) -> float:
        """Calculates IG (information gain)

        Args:
            X_column (np.ndarray): Column of feature values
            y (np.ndarray): labels corresponding to features from X_column
            threshold (float): threshold to calculate IG with

        Returns:
            float: Information gain
        """

        # parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        # Number of datapoints in each child
        n_l, n_r = len(left_idxs), len(right_idxs)
        # Entropy of each child
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = n_l / n * e_l + n_r / n * e_r

        return parent_entropy - child_entropy

    def _split(self, X_column: np.ndarray, split_threshold: float) -> tuple[np.ndarray]:
        """Splits given column of features into two (returns idxs of splitted features)

        Args:
            X_column (np.ndarray): Column to split
            split_threshold (float): threshold to split by

        Returns:
            tuple[np.ndarray]: left and right idxs
        """
        return np.argwhere(X_column < split_threshold).flatten(), np.argwhere(X_column >= split_threshold).flatten()


    def _entropy(self, y: np.ndarray) -> float:
        """Calculates entropy of given array

        Args:
            y (np.ndarray): array for which the entropy is to be calculated

        Returns:
            float: entropy value
        """
        pxs = np.bincount(y) / len(y)
        return -np.sum([px * np.log(px) for px in pxs if px > 0])
