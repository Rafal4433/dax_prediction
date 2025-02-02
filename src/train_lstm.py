# src/train_lstm.py
import numpy as np
import pandas as pd
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from data_utils import scale_data, create_sequences
from models.dl_models import LSTMModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load and preprocess data
loader = DataLoader(symbol="^GDAXI", interval="5m", period="60d")
raw_data = loader.fetch_data()

preprocessor = DataPreprocessor(raw_data)
processed_data = preprocessor.clean_data()
processed_data = preprocessor.format_datetime(datetime_col="Datetime", new_index=True)
processed_data = preprocessor.sort_data()

# 2. Feature engineering
fe = FeatureEngineer(processed_data)
df_features = fe.engineer_features()

# 3. Create target variable: predict next 'Close' value
df_features["target"] = df_features["Close"].shift(-1)
df_features.dropna(inplace=True)

# 4. Select features (możesz dostosować, tutaj przykładowo wybieramy tylko kolumny liczbowe)
features = df_features.select_dtypes(include=["float64", "int64"]).drop(columns=["target"])
target = df_features["target"].values

# 5. Scale features
X_scaled, scaler = scale_data(features.values)

# 6. Create sequences for LSTM
timesteps = 10  # Liczba kroków czasowych w sekwencji
X_seq, y_seq = create_sequences(X_scaled, target, timesteps=timesteps)

# 7. Podział na zbiór treningowy i walidacyjny
split = int(0.8 * len(X_seq))
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

# 8. Inicjalizacja i trenowanie modelu LSTM
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
lstm_model = LSTMModel(input_shape=input_shape, units=64, dropout=0.3, learning_rate=1e-3, epochs=30, batch_size=32)
lstm_model.build_model()
history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val))

# 9. Ewaluacja modelu
evaluation = lstm_model.evaluate(X_val, y_val)
print("Evaluation Metrics:", evaluation)

# 10. Wizualizacja historii treningu
lstm_model.plot_training_history()