import numpy as np
from .Evaluator import Evaluator

class Model:
    """Parent class for all models"""
    def __init__(self):
        pass
    
    # this method must be overwritten
    def fit(X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplemented
    
    # this method must be overwritten
    def predict(features: np.ndarray) -> int:
        raise NotImplemented
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, intermediate_states: bool = False) -> None:
        eval = Evaluator()
        if intermediate_states:
            eval.print_head()

        for features, label in zip(X, y):
            prediction = self.predict(features)
            eval.judge(prediction, label)
            if intermediate_states:
                eval.print()

        print('\nFinal:')
        print(eval.metrics)
        print()
        print(eval.conf_matrix)
 

