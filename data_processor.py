
import ccxt
import pandas as pd
import numpy as np
from typing import Optional
from config import Config


class DataProcessor:

    def __init__(self, testnet: bool = True):
        self.exchange = ccxt.binance({
            'apiKey': Config.API_KEY,
            'secret': Config.API_SECRET,
            'enableRateLimit': True,
        })

        if testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_current_price(self, symbol: str = 'BTC/USDT') -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching price: {e}")
            return 0.0

    def fetch_ohlcv(self, symbol: str = 'BTC/USDT',
                   timeframe: str = '1m',
                   limit: int = 100) -> pd.DataFrame:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching OHLCV: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)

        return df
