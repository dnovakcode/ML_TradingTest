
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', '')
    API_SECRET = os.getenv('BINANCE_TESTNET_API_SECRET', '')

    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1m'
    INITIAL_BALANCE = 10000

    LOOKBACK_WINDOW = 100
    EPISODES = 1000
    LEARNING_RATE = 0.0003

    MAX_POSITION_SIZE = 0.1

    DATABASE_URL = 'sqlite:///trading_bot.db'

    LOG_LEVEL = 'INFO'
    TENSORBOARD_DIR = './tensorboard_logs'
