#!/usr/bin/env python3

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from typing import Optional, Tuple
import time


class HistoricalDataFetcher:

    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '1h', cache_dir: str = './data_cache'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_data(self, days: int = 365, force_refresh: bool = False) -> pd.DataFrame:
        cache_file = os.path.join(
            self.cache_dir,
            f"{self.symbol.replace('/', '_')}_{self.timeframe}_{days}d.pkl"
        )

        if not force_refresh and os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 3600:
                print(f"📂 Загрузка из кеша ({cache_age/60:.1f} мин назад)...")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        print(f"🌐 Загрузка {days} дней данных {self.symbol} ({self.timeframe})...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        all_candles = []
        current_date = start_date

        while current_date < end_date:
            try:
                since = int(current_date.timestamp() * 1000)
                candles = self.exchange.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    since=since,
                    limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                last_timestamp = candles[-1][0]
                current_date = datetime.fromtimestamp(last_timestamp / 1000)

                print(f"  Загружено до {current_date.strftime('%Y-%m-%d %H:%M')} ({len(all_candles)} свечей)")

                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"⚠️ Ошибка загрузки: {e}")
                time.sleep(5)
                continue

        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df = df[~df.index.duplicated(keep='first')]

        df.sort_index(inplace=True)

        print(f"✅ Загружено {len(df)} свечей от {df.index[0]} до {df.index[-1]}")

        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_25'] = df['close'].rolling(window=25).mean()
        df['sma_99'] = df['close'].rolling(window=99).mean()

        df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema_25'] = df['close'].ewm(span=25, adjust=False).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()

        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        df['volatility'] = df['returns'].rolling(window=20).std()

        df.dropna(inplace=True)

        return df

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"""
📊 Разделение данных:
  Train: {len(train_df)} свечей ({train_df.index[0]} → {train_df.index[-1]})
  Val:   {len(val_df)} свечей ({val_df.index[0]} → {val_df.index[-1]})
  Test:  {len(test_df)} свечей ({test_df.index[0]} → {test_df.index[-1]})
        """)

        return train_df, val_df, test_df


def main():
    fetcher = HistoricalDataFetcher(symbol='BTC/USDT', timeframe='1h')

    df = fetcher.fetch_data(days=365)

    df = fetcher.add_technical_indicators(df)

    train_df, val_df, test_df = fetcher.split_data(df)

    print(f"\n✅ Данные готовы!")
    print(f"Колонки: {list(df.columns)}")
    print(f"\nПример данных:")
    print(df.tail())


if __name__ == "__main__":
    main()
