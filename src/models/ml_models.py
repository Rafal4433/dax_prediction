# src/models/ml_models.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the RandomForest model.
        
        :param n_estimators: Number of trees in the forest.
        :param random_state: Seed used by the random number generator.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    
    def fit(self, X, y):
        """
        Train the RandomForest model.
        
        :param X: Training features.
        :param y: Training target.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict using the trained RandomForest model.
        
        :param X: Input features.
        :return: Predicted values.
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data using RMSE and MAE.
        
        :param X: Test features.
        :param y: True target values.
        :return: Dictionary with RMSE and MAE.
        """
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        return {"rmse": rmse, "mae": mae}