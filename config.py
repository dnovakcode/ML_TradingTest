# config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance Testnet credentials
    API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', '')
    
    # Trading parameters
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1m'  # 1 минута для скальпинга
    INITIAL_BALANCE = 10000  # USDT
    
    # ML parameters
    LOOKBACK_WINDOW = 100  # Количество свечей для анализа
    EPISODES = 1000  # Количество эпизодов обучения
    LEARNING_RATE = 0.0003
    
    # Risk management (начальные значения, бот будет их адаптировать)
    MAX_POSITION_SIZE = 0.1  # 10% от баланса максимум
    
    # Database
    DATABASE_URL = 'sqlite:///trading_bot.db'
    
    # Monitoring
    LOG_LEVEL = 'INFO'
    TENSORBOARD_DIR = './tensorboard_logs'
