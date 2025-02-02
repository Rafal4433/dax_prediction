# src/models/dl_models.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_shape, units=50, dropout=0.2, learning_rate=1e-3, epochs=50, batch_size=32):
        """
        Initialize LSTMModel with model parameters.
        
        :param input_shape: Tuple representing (timesteps, features).
        :param units: Number of LSTM units.
        :param dropout: Dropout rate.
        :param learning_rate: Learning rate for optimizer.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size during training.
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        """Build and compile the LSTM model."""
        self.model = Sequential([
            LSTM(self.units, input_shape=self.input_shape, return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return self.model

    def fit(self, X, y, validation_data=None, verbose=1):
        """Train the LSTM model."""
        if self.model is None:
            self.build_model()
        self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, verbose=verbose)
        return self.history

    def predict(self, X):
        """Predict using the trained LSTM model."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model using MSE."""
        loss = self.model.evaluate(X, y, verbose=0)
        return {"mse": loss}

    def plot_training_history(self):
        """Plot training and validation loss using matplotlib."""
        if not hasattr(self, 'history'):
            print("No training history available.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()