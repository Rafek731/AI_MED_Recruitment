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
        """Container for a single decision-tree node.

        This node may represent either an internal split (in which case
        ``feature`` and ``threshold`` are set and ``left``/``right`` point to
        child nodes) or a leaf (in which case ``value`` is set to the
        predicted class label and children are ``None``).

        Args:
            feature (int | None): Index of the feature used for splitting at
                this node. None for leaf nodes.
            threshold (float | None): Threshold value to compare the feature
                against to decide left/right traversal. None for leaf nodes.
            left (Node | None): Left child node (values <= threshold go left).
            right (Node | None): Right child node (values > threshold go right).
            value (int | None): Class label for leaf nodes; None for internal
                nodes.
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    

    def is_leaf_node(self) -> bool:
        """Return True if this node is a leaf (has a predicted value).

        A node is considered a leaf when ``value`` is not ``None``. Internal
        nodes used for splitting have ``value`` == ``None``.

        Returns:
            bool: True when node is a leaf, False otherwise.
        """

        return self.value is not None


class DecisionTree(Model):
    """A simple decision tree classifier implemented from scratch.

    The implementation uses information gain (entropy) to select splits and
    supports limiting tree depth and minimum samples per split. It expects
    integer class labels (0...n_classes-1) in ``y`` and numeric feature
    values in ``X``.
    """
    def __init__(self,
                 min_samples_split: int = 2,
                 max_depth: int = 10,
                 n_features: int|None = None):
        """Create a DecisionTree classifier.

        Args:
            min_samples_split (int): Minimum number of samples required to
                attempt a split at a node. Nodes with fewer samples become
                leaves. Defaults to 2.
            max_depth (int): Maximum depth of the tree. Depth counting starts
                at 0 for the root. Defaults to 10.
            n_features (int | None): Number of features to consider when
                looking for the best split (useful for randomness). If None,
                all features are considered. When provided, the value will be
                clipped to the number of available features during fitting.
        """

        super().__init__()
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._n_features = n_features
        self._root=None


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the decision tree on the provided dataset.

        This builds the tree in-place and stores its root node on
        ``self.root``. The method sets ``self._n_features`` to the number of
        features used for split search (clipped to the available feature
        count).

        Args:
            X (np.ndarray): 2-D array of shape (n_samples, n_features) with
                numeric feature values.
            y (np.ndarray): 1-D integer array of shape (n_samples,) with class
                labels (expected as non-negative integers).

        Raises:
            ValueError: If X and y have inconsistent first-dimension lengths.

        Side effects:
            - Modifies ``self._n_features`` when it was None or larger than
              the number of features in ``X``.
            - Sets ``self.root`` to the constructed tree Node.

        Example:
            >>> clf = DecisionTree(max_depth=5)
            >>> clf.fit(X_train, y_train)
        """
        if self._n_features is None or self._n_features < 1:
            self._n_features = X.shape[1]
        else:
            self._n_features = min(self._n_features, X.shape[1])
        
        self.root = self._grow_tree(X, y)
        

    def predict(self, features: np.ndarray) -> int:
        """Predict the class label for a single feature vector.

        Args:
            features (np.ndarray): 1-D array of length n_features containing
                numeric feature values for a single sample.

        Returns:
            int: Predicted class label (as an integer).

        Notes:
            Use this method for single-sample prediction. For multiple inputs
            you may call this repeatedly or implement a vectorized wrapper.
        """

        return self._traverse_tree(features, self.root)


    def _traverse_tree(self, x, node: Node) -> int:
        """Recursively traverse the tree to predict the label for ``x``.

        Args:
            x (np.ndarray): 1-D feature array for a single sample.
            node (Node): Current node to inspect.

        Returns:
            int: The predicted class label from the reached leaf node.

        Raises:
            RuntimeError: If an internal node has no valid children (should not
                happen in a correctly built tree).
        """

        if node.is_leaf_node():
            return node.value

        # Decide to go left or right based on the split condition
        if x[node.feature] <= node.threshold:
            if node.left is None:
                raise RuntimeError("Left child is missing for internal node")
            return self._traverse_tree(x, node.left)

        if node.right is None:
            raise RuntimeError("Right child is missing for internal node")
        return self._traverse_tree(x, node.right)


    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int=0) -> Node:
        """Recursively build the decision tree starting from the given data.

        The function examines stopping criteria (max depth, pure node or
        insufficient samples) to decide whether to create a leaf node or to
        search for the best split and create internal nodes.

        Args:
            X (np.ndarray): 2-D array of shape (n_samples, n_features).
            y (np.ndarray): 1-D array of labels of length n_samples.
            depth (int, optional): Current depth in the tree (root=0).

        Returns:
            Node: A Node object representing the root of the subtree built for
                (X, y).

        Raises:
            ValueError: If X and y have mismatched sample counts.
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
        """Return the most common label in ``y``.

        Args:
            y (np.ndarray): 1-D array-like of integer labels.

        Returns:
            int: The label with highest frequency. In case of a tie the
            label returned is the one appearing first in Counter's ordering.
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
        """Compute the information gain from splitting ``X_column`` at ``threshold``.

        Information gain is defined as parent_entropy - weighted_child_entropy.

        Args:
            X_column (np.ndarray): 1-D array with values of a single feature for all samples.
            y (np.ndarray): 1-D array of class labels aligned with X_column``.
            threshold (float): Numeric threshold to split the column into left (< threshold) and right (>= threshold) subsets.

        Returns:
            float: The information gain (non-negative). Returns 0.0 when a
            split produces an empty child.
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
        """Split a 1-D feature array into left and right index arrays.

        Left indices correspond to samples where X_column < split_threshold.
        Right indices correspond to samples where X_column >= split_threshold.

        Args:
            X_column (np.ndarray): 1-D array to partition.
            split_threshold (float): Numeric threshold for the split.

        Returns:
            tuple[np.ndarray, np.ndarray]: (left_indices, right_indices), both
                as 1-D integer numpy arrays.
        """

        return np.argwhere(X_column < split_threshold).flatten(), np.argwhere(X_column >= split_threshold).flatten()


    def _entropy(self, y: np.ndarray) -> float:
        """Compute the Shannon entropy of the label distribution in ``y``.

        The entropy is calculated as -sum(p_i * log(p_i)) over all label
        classes with non-zero probability. Natural log is used.

        Args:
            y (np.ndarray): 1-D integer array of class labels.

        Returns:
            float: Entropy value (>= 0).
        """

        pxs = np.bincount(y) / len(y)
        return -np.sum([px * np.log(px) for px in pxs if px > 0])
