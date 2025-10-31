import numpy as np
import pandas as pd

from .Evaluator import Evaluator


class Model:
    """Parent class for all models"""
    def __init__(self, name:str = 'model'):
        self.name = name
        self.eval = Evaluator()
    
    # this method must be overwritten
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplemented
    
    # this method must be overwritten
    def predict(self, features: np.ndarray) -> int:
        raise NotImplemented
    
    def clear(self) -> None:
        # this method must be overwritten
        raise NotImplemented
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, intermediate_states: bool = False) -> tuple[str, str, pd.DataFrame]:
        """Evaluate the model over a dataset and return summary results.

        This method runs the model's ``predict`` method for each sample in
        ``X`` paired with its label in ``y``, records the prediction outcome
        using the internal ``Evaluator``, and optionally
        prints intermediate metrics after every prediction.

        Args:
            X (np.ndarray): 2-D array-like of shape (n_samples, n_features)
                containing feature vectors to evaluate.
            y (np.ndarray): 1-D array-like of length n_samples with ground
                truth integer labels corresponding to rows in ``X``.
            intermediate_states (bool, optional): If True, print evaluation
                metrics after each sample is judged. Defaults to False.

        Returns:
            tuple: ``(model_name, metrics, conf_matrix)`` where
                - model_name (str): the model's name (``self.name``),
                - metrics (dict): metric values as stored in
                  ``self.eval.metrics``, and
                - conf_matrix: confusion matrix stored in
                  ``self.eval.conf_matrix`` (type depends on Evaluator
                  implementation).

        Side effects:
            - Mutates ``self.eval`` by recording judgments.
            - May print to stdout when ``intermediate_states`` is True.
        """
        if intermediate_states:
            self.eval.print_head()

        for features, label in zip(X, y):
            prediction = self.predict(features)
            self.eval.judge(prediction, label)
            if intermediate_states:
                self.eval.print()

        return self.name, self.eval.metrics, self.eval.conf_matrix
    
    def print_results(self) -> None:
        """Print evaluation metrics stored in the Evaluator."""

        print(f"\nEvaluation results for model: {self.name}\n")
        print(self.eval.metrics, end='\n\n')
        print(self.eval.conf_matrix, end='\n\n')
 

