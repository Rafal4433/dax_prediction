# src/models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the training data.
        
        :param X: Training features.
        :param y: Training target.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict values using the trained model.
        
        :param X: Input features.
        :return: Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model using test data.
        
        :param X: Test features.
        :param y: True target values.
        :return: Dictionary with evaluation metrics.
        """
        pass