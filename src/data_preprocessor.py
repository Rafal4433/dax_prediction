# src/data_preprocessor.py
import pandas as pd

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataPreprocessor with a DataFrame.
        
        :param df: DataFrame containing the raw data.
        """
        self.df = df.copy()
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values and duplicate records.
        
        :return: Cleaned DataFrame.
        """
        # Drop duplicate rows
        self.df.drop_duplicates(inplace=True)
        
        # Fill missing values using forward fill method and then backward fill
        self.df.ffill(inplace=True)
        self.df.bfill(inplace=True)
        
        return self.df
    
    def format_datetime(self, datetime_col: str = "Datetime", new_index: bool = True) -> pd.DataFrame:
        """
        Convert the datetime column to datetime type and optionally set it as the index.
        
        :param datetime_col: The name of the datetime column.
        :param new_index: If True, set the datetime column as index.
        :return: DataFrame with formatted datetime column.
        """
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        if new_index:
            self.df.set_index(datetime_col, inplace=True)
        return self.df
    
    def sort_data(self) -> pd.DataFrame:
        """
        Sort the data by datetime index or datetime column.
        
        :return: Sorted DataFrame.
        """
        # Check if the index is a DatetimeIndex
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df.sort_index(inplace=True)
        else:
            # Otherwise, sort by the datetime column if it exists
            if 'Datetime' in self.df.columns:
                self.df.sort_values('Datetime', inplace=True)
        return self.df

if __name__ == "__main__":
    # Sample test using DataLoader and DataPreprocessor
    # Make sure that the DataLoader is accessible. Adjust the import path if needed.
    from data_loader import DataLoader  
    
    # Create a DataLoader instance to fetch data using yfinance
    loader = DataLoader(symbol="^GDAXI", interval="5m", period="60d")
    df_raw = loader.fetch_data()
    
    print("Raw Data:")
    print(df_raw.head())
    
    # Initialize the DataPreprocessor with the raw data
    preprocessor = DataPreprocessor(df_raw)
    
    # Clean the data (remove duplicates and fill missing values)
    df_cleaned = preprocessor.clean_data()
    # Format the datetime column and set it as the index
    df_formatted = preprocessor.format_datetime(datetime_col="Datetime", new_index=True)
    # Sort the data by datetime
    df_sorted = preprocessor.sort_data()
    
    print("\nProcessed Data:")
    print(df_sorted.head())