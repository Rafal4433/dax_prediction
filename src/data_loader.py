# src/data_loader.py
import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, symbol: str = "^GDAXI", interval: str = "5m", period: str = "60d"):
        self.symbol = symbol
        self.interval = interval
        self.period = period

    def fetch_data(self) -> pd.DataFrame:

        data = yf.download(
            tickers=self.symbol,
            period=self.period,
            interval=self.interval,
            progress=False
        )
        
        data.reset_index(inplace=True)
        
        data.sort_values('Datetime', inplace=True)
        
        return data

if __name__ == "__main__":
    loader = DataLoader(symbol="^GDAXI", interval="5m", period="60d")
    df = loader.fetch_data()
    print(df.head())