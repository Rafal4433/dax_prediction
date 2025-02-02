# src/tuner.py
import tensorflow as tf
import keras_tuner as kt

def build_model(hp, input_shape):
    """
    Build a Keras model with hyperparameters to be tuned.
    
    :param hp: HyperParameters instance from Keras Tuner.
    :param input_shape: Tuple representing (timesteps, features).
    :return: Compiled Keras model.
    """
    model = tf.keras.Sequential()
    # LSTM layer with tunable number of units
    model.add(tf.keras.layers.LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        input_shape=input_shape,
        return_sequences=False
    ))
    # Tunable dropout
    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(tf.keras.layers.Dense(1))
    
    # Tunable learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

def tune_model(X_train, y_train, input_shape, max_trials=10, executions_per_trial=1, epochs=20):
    """
    Tune hyperparameters for the LSTM model using Keras Tuner.
    
    :param X_train: Training features.
    :param y_train: Training targets.
    :param input_shape: Input shape for the model.
    :param max_trials: Maximum number of different hyperparameter configurations to try.
    :param executions_per_trial: Number of executions per trial.
    :param epochs: Number of epochs to train during tuning.
    :return: Tuner instance and best hyperparameters.
    """
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='tuner_dir',
        project_name='lstm_tuning'
    )
    tuner.search(X_train, y_train, epochs=epochs, validation_split=0.2)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:", best_hp.values)
    return tuner, best_hp