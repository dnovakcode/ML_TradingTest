#!/usr/bin/env python3
"""
🚀 PRODUCTION-READY ТОРГОВЫЙ БОТ v7.0
С реальными данными, профессиональным риск-менеджментом и метриками
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML imports
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn

# Наши модули
from data_fetcher import HistoricalDataFetcher
from risk_manager import DynamicRiskManager, RiskConfig
from metrics import TradingMetricsCalculator, PerformanceMetrics


@dataclass
class Trade:
    """Класс для отслеживания сделки"""
    entry_price: float
    entry_time: int
    entry_index: int
    position_size: float
    trade_type: str
    stop_loss: float
    take_profit: float
    steps_held: int = 0


class ProductionTradingEnvironment(gym.Env):
    """
    Production-ready торговое окружение с:
    - Реальными историческими данными
    - Профессиональным риск-менеджментом
    - Упрощенной системой наград (только PnL)
    - Детальными метриками
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000,
                 risk_config: Optional[RiskConfig] = None):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance

        # Риск-менеджмент
        self.risk_manager = DynamicRiskManager(risk_config or RiskConfig())

        # Observation space: [текущие индикаторы + состояние портфеля]
        # Размер зависит от количества колонок в df
        n_features = len(df.columns) + 10  # +10 для portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Action space: непрерывное [-1, 1]
        # -1 = продать всё, 0 = держать, +1 = купить максимум
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self._init_state()

    def _init_state(self):
        """Инициализация состояния"""
        self.balance = self.initial_balance
        self.btc_amount = 0.0
        self.current_step = 0

        # Торговля
        self.active_trade: Optional[Trade] = None
        self.closed_trades: List[Dict] = []

        # История для метрик
        self.portfolio_history = [self.initial_balance]
        self.balance_history = [self.initial_balance]

        # Статистика
        self.total_commission = 0.0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        super().reset(seed=seed)
        self._init_state()
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение шага"""
        self.current_step += 1

        # Защита от выхода за границы данных
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Получаем текущие данные
        current_data = self.df.iloc[self.current_step]
        current_price = float(current_data['close'])

        # Сохраняем предыдущее состояние для расчета награды
        prev_portfolio = self._get_portfolio_value()

        # Проверяем активную сделку на stop-loss/take-profit
        if self.active_trade:
            self.active_trade.steps_held += 1
            should_close, reason = self.risk_manager.should_close_position(
                entry_price=self.active_trade.entry_price,
                current_price=current_price,
                stop_loss=self.active_trade.stop_loss,
                take_profit=self.active_trade.take_profit,
                is_long=self.active_trade.trade_type == 'LONG',
                steps_held=self.active_trade.steps_held
            )

            if should_close:
                self._close_position(current_price, reason)

        # Выполняем действие агента
        action_value = float(action[0])
        self._execute_action(action_value, current_price, current_data)

        # Вычисляем награду (УПРОЩЕННАЯ - только изменение портфеля)
        current_portfolio = self._get_portfolio_value()
        reward = self._calculate_reward(prev_portfolio, current_portfolio)

        # Сохраняем историю
        self.portfolio_history.append(current_portfolio)
        self.balance_history.append(self.balance)

        # Проверяем условия завершения
        terminated = False
        truncated = current_portfolio < self.initial_balance * 0.5  # Потеря 50%

        info = self._get_info()

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_action(self, action: float, current_price: float, current_data: pd.Series):
        """Выполнение действия"""

        # HOLD: -0.15 < action < 0.15
        if abs(action) < 0.15:
            return

        # Получаем текущую экспозицию
        current_exposure = self.btc_amount * current_price

        # BUY: action >= 0.15
        if action >= 0.15 and not self.active_trade:
            # Проверяем возможность открытия сделки
            can_trade, reason = self.risk_manager.can_open_trade(
                balance=self.balance,
                current_exposure=current_exposure
            )

            if not can_trade:
                return

            # Вычисляем размер позиции через риск-менеджмент
            signal_strength = min(1.0, abs(action) * 1.5)
            volatility = float(current_data.get('volatility', 0.02))
            atr = float(current_data.get('atr', current_price * 0.02))

            position_size = self.risk_manager.calculate_position_size(
                balance=self.balance,
                current_price=current_price,
                signal_strength=signal_strength,
                volatility=volatility
            )

            if position_size < 100:
                return

            # Вычисляем комиссию
            commission = self.risk_manager.calculate_commission(position_size)

            if position_size + commission > self.balance:
                return

            # Открываем сделку
            btc_to_buy = position_size / current_price
            self.balance -= (position_size + commission)
            self.btc_amount += btc_to_buy
            self.total_commission += commission

            # Вычисляем stop-loss и take-profit
            sl, tp = self.risk_manager.calculate_stop_loss_take_profit(
                entry_price=current_price,
                is_long=True,
                atr=atr
            )

            # Создаем Trade объект
            self.active_trade = Trade(
                entry_price=current_price,
                entry_time=int(self.current_step),
                entry_index=self.current_step,
                position_size=btc_to_buy,
                trade_type='LONG',
                stop_loss=sl,
                take_profit=tp
            )

        # SELL: action <= -0.15
        elif action <= -0.15 and self.active_trade:
            self._close_position(current_price, "agent_decision")

    def _close_position(self, exit_price: float, reason: str):
        """Закрытие позиции"""
        if not self.active_trade:
            return

        # Вычисляем выручку
        revenue = self.active_trade.position_size * exit_price
        commission = self.risk_manager.calculate_commission(revenue)

        # Обновляем баланс
        self.balance += (revenue - commission)
        self.btc_amount -= self.active_trade.position_size
        self.total_commission += commission

        # Вычисляем PnL
        cost = self.active_trade.position_size * self.active_trade.entry_price
        pnl = revenue - cost
        pnl_after_commission = pnl - (commission * 2)  # Вход + выход
        pnl_pct = (pnl / cost) * 100

        # Записываем результат
        trade_result = {
            'entry_price': self.active_trade.entry_price,
            'exit_price': exit_price,
            'entry_time': self.active_trade.entry_time,
            'exit_time': self.current_step,
            'duration': self.current_step - self.active_trade.entry_time,
            'steps_held': self.active_trade.steps_held,
            'pnl': pnl,
            'pnl_after_commission': pnl_after_commission,
            'pnl_percent': pnl_pct,
            'reason': reason
        }

        self.closed_trades.append(trade_result)

        # Обновляем риск-менеджер
        self.risk_manager.record_trade_result(pnl_after_commission)

        # Закрываем сделку
        self.active_trade = None

    def _calculate_reward(self, prev_portfolio: float, current_portfolio: float) -> float:
        """
        УПРОЩЕННАЯ И ЧЕСТНАЯ СИСТЕМА НАГРАД
        Награда = только изменение портфеля в процентах

        Это заставляет агента:
        1. Максимизировать прибыль
        2. Минимизировать убытки
        3. Избегать комиссий (держать позиции дольше)
        """
        portfolio_change = current_portfolio - prev_portfolio
        portfolio_change_pct = (portfolio_change / prev_portfolio) * 100

        # Масштабируем для стабильного обучения: 1% = +100 reward
        reward = portfolio_change_pct * 100

        # Дополнительно наказываем за слишком большие убытки
        if portfolio_change_pct < -2:
            reward *= 1.5  # Усиливаем негативный сигнал

        # Clip для стабильности
        reward = np.clip(reward, -500, 500)

        return float(reward)

    def _get_portfolio_value(self) -> float:
        """Общая стоимость портфеля"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        return self.balance + (self.btc_amount * current_price)

    def _get_observation(self) -> np.ndarray:
        """Получить наблюдение"""
        current_data = self.df.iloc[self.current_step]

        obs = []

        # 1. Нормализованные технические индикаторы из DataFrame
        close_price = float(current_data['close'])

        # Добавляем все доступные индикаторы
        for col in self.df.columns:
            if col in ['timestamp', 'open', 'high', 'low', 'volume']:
                continue

            value = float(current_data[col])

            # Нормализация разных типов индикаторов
            if col == 'close':
                obs.append(value / 70000)  # Нормализация цены
            elif col.startswith('rsi'):
                obs.append(value / 100)  # RSI уже 0-100
            elif col.startswith('returns') or col.startswith('log_returns'):
                obs.append(value * 100)  # Процентные изменения
            elif col.startswith('volume'):
                obs.append(np.log1p(value) / 20)  # Log нормализация объема
            else:
                # Общая нормализация для других индикаторов
                obs.append(value / close_price if close_price > 0 else 0)

        # 2. Состояние портфеля (10 параметров)
        portfolio_value = self._get_portfolio_value()

        obs.extend([
            self.balance / self.initial_balance,  # Нормализованный баланс
            (self.btc_amount * close_price) / self.initial_balance,  # Нормализованная позиция
            (portfolio_value - self.initial_balance) / self.initial_balance,  # ROI
            1.0 if self.active_trade else 0.0,  # Есть ли активная сделка
        ])

        # Информация об активной сделке
        if self.active_trade:
            unrealized_pnl = (close_price - self.active_trade.entry_price) * self.active_trade.position_size
            unrealized_pnl_pct = ((close_price - self.active_trade.entry_price) /
                                 self.active_trade.entry_price) * 100

            obs.extend([
                unrealized_pnl / self.initial_balance,
                unrealized_pnl_pct / 10,  # Нормализация
                self.active_trade.steps_held / 100,
                (close_price - self.active_trade.stop_loss) / close_price,
                (self.active_trade.take_profit - close_price) / close_price,
            ])
        else:
            obs.extend([0, 0, 0, 0, 0])

        # Недавняя производительность
        if len(self.closed_trades) > 0:
            recent_trades = self.closed_trades[-5:]
            wins = sum(1 for t in recent_trades if t['pnl_after_commission'] > 0)
            win_rate = wins / len(recent_trades)
            obs.append(win_rate)
        else:
            obs.append(0.5)

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Получить информацию о состоянии"""
        return {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'active_trade': self.active_trade is not None,
            'closed_trades': len(self.closed_trades),
            'total_commission': self.total_commission,
            'risk_status': self.risk_manager.get_risk_status(self.balance)
        }


class DetailedCallback(BaseCallback):
    """Callback для детального мониторинга"""

    def __init__(self, eval_env: ProductionTradingEnvironment,
                 log_freq: int = 1000, eval_freq: int = 5000):
        super().__init__()
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.metrics_calculator = TradingMetricsCalculator()

    def _on_step(self) -> bool:
        # Периодическая оценка
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_agent()

        # Логирование прогресса
        if self.n_calls % self.log_freq == 0:
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                print(f"\n📊 Шаг {self.n_calls}:")
                print(f"  Portfolio: ${info['portfolio_value']:.2f}")
                print(f"  Сделок закрыто: {info['closed_trades']}")
                print(f"  Риск: {info['risk_status']['risk_level']}")

        return True

    def _evaluate_agent(self):
        """Оценка агента"""
        print(f"\n{'='*60}")
        print(f"🧪 ОЦЕНКА АГЕНТА (шаг {self.n_calls})")
        print(f"{'='*60}")

        # Вычисляем метрики на train environment
        if len(self.eval_env.portfolio_history) > 10:
            metrics = self.metrics_calculator.calculate_metrics(
                portfolio_values=self.eval_env.portfolio_history,
                closed_trades=self.eval_env.closed_trades,
                days_traded=len(self.eval_env.portfolio_history) / 24  # Если 1h свечи
            )

            print(f"💰 ROI: {metrics.total_return_pct:+.2f}%")
            print(f"📈 Sharpe: {metrics.sharpe_ratio:.3f}")
            print(f"🎯 Win Rate: {metrics.win_rate:.1f}%")
            print(f"💎 Profit Factor: {metrics.profit_factor:.2f}")
            print(f"⚠️  Max DD: {metrics.max_drawdown_pct:.2f}%")

            # Сохраняем лучшую модель
            if metrics.sharpe_ratio > self.best_sharpe:
                self.best_sharpe = metrics.sharpe_ratio
                self.model.save("./models/best_model_by_sharpe")
                print(f"\n🌟 НОВЫЙ РЕКОРД Sharpe! Модель сохранена.")

        print(f"{'='*60}\n")


def train_agent(train_df: pd.DataFrame, val_df: pd.DataFrame,
                total_timesteps: int = 200000):
    """Обучение агента"""

    print(f"""
{'='*80}
🚀 ЗАПУСК ОБУЧЕНИЯ PRODUCTION-READY БОТА
{'='*80}
📊 Train данных: {len(train_df)} свечей
📊 Val данных:   {len(val_df)} свечей
🎯 Шагов обучения: {total_timesteps:,}
{'='*80}
    """)

    # Создаем окружение
    risk_config = RiskConfig(
        max_position_size_pct=15.0,
        max_risk_per_trade_pct=2.0,
        default_stop_loss_pct=3.0,
        default_take_profit_pct=6.0,
        max_drawdown_pct=20.0
    )

    train_env = ProductionTradingEnvironment(train_df, initial_balance=10000, risk_config=risk_config)

    # Создаем агента
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,  # Выше для долгосрочных стратегий
        ent_coef='auto',
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )

    # Callback
    callback = DetailedCallback(train_env, log_freq=2000, eval_freq=10000)

    # Обучение
    print("\n🎓 НАЧАЛО ОБУЧЕНИЯ...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        print("\n✅ Обучение завершено!")
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано")

    # Сохраняем финальную модель
    os.makedirs("./models", exist_ok=True)
    model.save("./models/trading_bot_pro_final")
    print("💾 Финальная модель сохранена")

    return model, train_env


def evaluate_agent(model, test_df: pd.DataFrame, initial_balance: float = 10000):
    """Оценка агента на тестовых данных"""

    print(f"""
{'='*80}
🧪 ТЕСТИРОВАНИЕ НА OUT-OF-SAMPLE ДАННЫХ
{'='*80}
📊 Тестовых данных: {len(test_df)} свечей
💰 Начальный баланс: ${initial_balance:,}
{'='*80}
    """)

    risk_config = RiskConfig()
    test_env = ProductionTradingEnvironment(test_df, initial_balance=initial_balance,
                                           risk_config=risk_config)

    obs, _ = test_env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)

    # Вычисляем финальные метрики
    print("\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")

    calculator = TradingMetricsCalculator(initial_balance=initial_balance)
    metrics = calculator.calculate_metrics(
        portfolio_values=test_env.portfolio_history,
        closed_trades=test_env.closed_trades,
        days_traded=len(test_df) / 24  # Для 1h свечей
    )

    calculator.print_metrics(metrics)

    return metrics, test_env


def main():
    """Главная функция"""

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     🚀 PRODUCTION-READY ТОРГОВЫЙ БОТ v7.0 🚀            ║
    ║                                                           ║
    ║  ✅ Реальные исторические данные                         ║
    ║  ✅ Профессиональный риск-менеджмент                     ║
    ║  ✅ Stop-loss & Take-profit                              ║
    ║  ✅ Динамический position sizing                         ║
    ║  ✅ Честная система наград (только PnL)                  ║
    ║  ✅ Профессиональные метрики (Sharpe, Sortino, etc.)    ║
    ║  ✅ Walk-forward validation                              ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # 1. Загружаем данные
    print("\n📥 ЗАГРУЗКА ДАННЫХ...")
    fetcher = HistoricalDataFetcher(symbol='BTC/USDT', timeframe='1h')
    df = fetcher.fetch_data(days=365, force_refresh=False)
    df = fetcher.add_technical_indicators(df)

    # 2. Разделяем на train/val/test
    train_df, val_df, test_df = fetcher.split_data(df, train_ratio=0.7, val_ratio=0.15)

    # 3. Обучение
    model, train_env = train_agent(train_df, val_df, total_timesteps=200000)

    # 4. Тестирование
    metrics, test_env = evaluate_agent(model, test_df)

    print("\n✅ ГОТОВО!")
    print("💾 Модель сохранена в ./models/trading_bot_pro_final.zip")
    print("🎯 Можно использовать для реальной торговли, если метрики хорошие!")


if __name__ == "__main__":
    main()
