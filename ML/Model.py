import numpy as np

class Model:
    """Parent class for all models"""
    def __init__(self):
        pass

    def fit(X: np.ndarray, y: np.ndarray):
        raise NotImplemented
    
    def predict(X: np.ndarray):
        raise NotImplemented