# src/trainer.py
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class Trainer:
    def __init__(self, df: pd.DataFrame, target_col: str = "Close"):
        """
        Initialize Trainer with a DataFrame containing engineered features.
        
        :param df: DataFrame with features.
        :param target_col: Name of the column to predict.
        """
        self.df = df.copy()
        self.target_col = target_col
        
    def create_target(self, shift: int = -1):
        """
        Create a target column shifted by the specified number of steps.
        In this case, we predict the next time step's 'Close' price.
        
        :param shift: Shift value to create the target (e.g., -1 for next time step).
        :return: DataFrame with the new 'target' column.
        """
        self.df["target"] = self.df[self.target_col].shift(shift)
        self.df.dropna(inplace=True)
        return self.df
    
    def split_data(self, train_size: float = 0.8):
        """
        Split the data into training and testing sets without shuffling.
        
        :param train_size: Proportion of data to use for training.
        :return: X_train, X_test, y_train, y_test
        """
        # Ensure that the target column exists
        if "target" not in self.df.columns:
            self.create_target(shift=-1)
        
        # Use only numeric features for training (drop non-numeric columns if needed)
        features = self.df.drop(columns=["target"])
        X = features.select_dtypes(include=["float64", "int64"])
        y = self.df["target"]
        
        # Split the dataset based on time ordering
        split_index = int(len(self.df) * train_size)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model):
        """
        Train the given model using the training set and evaluate it on the test set.
        
        :param model: An instance of a model that implements fit, predict, and evaluate.
        :return: The trained model and evaluation metrics.
        """
        X_train, X_test, y_train, y_test = self.split_data()
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        return model, metrics

if __name__ == "__main__":
    # Sample test: Full pipeline from data loading to model training.
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    from models.ml_models import RandomForestModel
    
    # Fetch data using yfinance
    loader = DataLoader(symbol="^GDAXI", interval="5m", period="60d")
    raw_data = loader.fetch_data()
    
    # Preprocess the data: cleaning, formatting datetime, and sorting
    preprocessor = DataPreprocessor(raw_data)
    processed_data = preprocessor.clean_data()
    processed_data = preprocessor.format_datetime(datetime_col="Datetime", new_index=True)
    processed_data = preprocessor.sort_data()
    
    # Feature engineering: add technical indicators and lag features
    fe = FeatureEngineer(processed_data)
    df_features = fe.engineer_features()
    
    # Initialize Trainer with feature engineered data
    trainer = Trainer(df_features, target_col="Close")
    # Create target variable (next time step's close price)
    trainer.create_target(shift=-1)
    
    # Initialize a RandomForest model
    rf_model = RandomForestModel(n_estimators=100, random_state=42)
    
    # Train the model and obtain evaluation metrics
    trained_model, metrics = trainer.train_model(rf_model)
    print("Evaluation Metrics:", metrics)