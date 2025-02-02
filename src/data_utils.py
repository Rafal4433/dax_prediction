# src/data_utils.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_data(X):
    """
    Scale data using StandardScaler.
    
    :param X: NumPy array or pandas DataFrame with numeric features.
    :return: Scaled data and the scaler instance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def create_sequences(X, y, timesteps=10):
    """
    Create sequences of features and corresponding target values.
    
    :param X: NumPy array with features (samples, features).
    :param y: NumPy array with target values.
    :param timesteps: Number of time steps in each sequence.
    :return: X_seq (3D array) and y_seq (1D array).
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)