"""
🚀 УЛУЧШЕННЫЙ ТОРГОВЫЙ БОТ v5.0
Исправленная система наград и обучения
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import ccxt
import time
import os
import signal
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import json
from dataclasses import dataclass

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn

from config import Config


@dataclass
class Trade:
    """Класс для отслеживания отдельной сделки"""
    entry_price: float
    entry_time: int
    position_size: float
    trade_type: str
    stop_loss: float = 0.0
    take_profit: float = 0.0
    steps_held: int = 0


class TradeTracker:
    """Трекер для отслеживания торговых позиций"""
    
    def __init__(self):
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Dict] = []
        self.total_commission_paid = 0.0
        
    def open_trade(self, entry_price: float, position_size: float, 
                   trade_type: str, stop_loss: float = 0.0, take_profit: float = 0.0) -> Trade:
        trade = Trade(
            entry_price=entry_price,
            entry_time=int(time.time()),
            position_size=position_size,
            trade_type=trade_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            steps_held=0
        )
        self.active_trades.append(trade)
        return trade
    
    def close_trade(self, exit_price: float, commission: float = 0.0) -> Dict:
        if not self.active_trades:
            return {}
            
        trade = self.active_trades.pop(0)
        
        if trade.trade_type == 'LONG':
            pnl = trade.position_size * (exit_price - trade.entry_price)
        else:
            pnl = trade.position_size * (trade.entry_price - exit_price)
            
        pnl_percent = (pnl / (trade.position_size * trade.entry_price)) * 100
        pnl_after_commission = pnl - commission
        
        self.total_commission_paid += commission
        
        result = {
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'pnl_after_commission': pnl_after_commission,
            'duration': int(time.time()) - trade.entry_time,
            'steps_held': trade.steps_held
        }
        
        self.closed_trades.append(result)
        return result
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Получить текущий нереализованный PnL"""
        total_unrealized = 0.0
        for trade in self.active_trades:
            if trade.trade_type == 'LONG':
                unrealized = trade.position_size * (current_price - trade.entry_price)
            else:
                unrealized = trade.position_size * (trade.entry_price - current_price)
            total_unrealized += unrealized
        return total_unrealized
    
    def update_steps(self):
        """Увеличить счетчик шагов для активных сделок"""
        for trade in self.active_trades:
            trade.steps_held += 1


class ImprovedTradingEnvironment(gym.Env):
    """Улучшенное торговое окружение с правильной системой наград"""
    
    def __init__(self, initial_balance: float = 10000, detailed_logging: bool = True):
        super().__init__()
        
        self.initial_balance = initial_balance
        self.detailed_logging = detailed_logging
        
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )
        
        self._init_exchange()
        self._init_state()
        
    def _init_exchange(self):
        """Инициализация подключения к Binance"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': Config.API_KEY,
                'secret': Config.API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            self.exchange.set_sandbox_mode(True)
            print("✅ Подключен к Binance Testnet")
        except:
            print("📊 Работаем в режиме симуляции")
            self.exchange = None
            
    def _init_state(self):
        """Инициализация состояния"""
        self.balance = self.initial_balance
        self.btc_amount = 0.0
        self.current_price = 67000.0
        
        self.price_history = []
        self.action_history = []
        self.reward_history = []
        
        self.trade_tracker = TradeTracker()
        
        self.total_steps = 0
        self.consecutive_holds = 0
        self.last_action_type = 'HOLD'
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        super().reset(seed=seed)
        self._init_state()
        self._update_market_data()
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение действия"""
        self.total_steps += 1

        self._update_market_data()

        if self.current_price is None or self.current_price <= 0:
            print(f"⚠️ КРИТИЧЕСКАЯ ОШИБКА: current_price = {self.current_price}, сброс на 67000")
            self.current_price = 67000.0

        prev_portfolio_value = self._get_portfolio_value()
        prev_balance = self.balance
        prev_btc = self.btc_amount
        
        self.trade_tracker.update_steps()
        
        action_result = self._execute_action(action[0])  # Берем первый элемент массива
        
        reward = self._calculate_reward(
            action[0],
            action_result,
            prev_portfolio_value,
            prev_balance,
            prev_btc
        )
        
        self.action_history.append(action[0])
        self.reward_history.append(reward)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
            self.reward_history = self.reward_history[-100:]
        
        current_portfolio_value = self._get_portfolio_value()
        terminated = False
        truncated = current_portfolio_value < self.initial_balance * 0.1  # Потеря 90%
        
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'btc_amount': self.btc_amount,
            'current_price': self.current_price,
            'action_type': action_result['type'],
            'action_success': action_result['success'],
            'unrealized_pnl': self.trade_tracker.get_unrealized_pnl(self.current_price),
            'active_trades': len(self.trade_tracker.active_trades),
            'closed_trades': len(self.trade_tracker.closed_trades)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _execute_action(self, action: float) -> Dict:
        """Выполнение действия с непрерывным значением"""
        result = {'type': 'UNKNOWN', 'success': False, 'details': {}}

        if self.current_price is None or self.current_price <= 0:
            print(f"⚠️ ОШИБКА в _execute_action: current_price = {self.current_price}")
            self.current_price = 67000.0

        
        if -0.1 <= action <= 0.1:
            self.consecutive_holds += 1
            result['type'] = 'HOLD'
            result['success'] = True
            
            if self.detailed_logging and self.total_steps % 10 == 0:
                unrealized = self.trade_tracker.get_unrealized_pnl(self.current_price)
                print(f"⚪ HOLD @ ${self.current_price:.2f} | Unrealized: ${unrealized:+.2f}")
                
        elif action > 0.1:
            self.consecutive_holds = 0
            
            buy_strength = min(1.0, max(0.1, abs(action)))
            max_buy = self.balance * 0.4 * buy_strength  # До 40% баланса
            
            if max_buy < 100:  # Минимальная сделка $100
                result['type'] = 'BUY_FAILED'
                result['success'] = False
                result['details']['reason'] = 'insufficient_funds'
                
                if self.detailed_logging:
                    print(f"⛔ BUY FAILED: Недостаточно средств (${self.balance:.2f})")
            else:
                btc_to_buy = max_buy / self.current_price
                commission = max_buy * 0.001
                
                if max_buy + commission > self.balance:
                    max_buy = self.balance * 0.95  # Оставляем 5% на комиссию
                    btc_to_buy = max_buy / self.current_price
                    commission = max_buy * 0.001
                
                self.balance -= (max_buy + commission)
                self.btc_amount += btc_to_buy
                
                self.trade_tracker.open_trade(
                    entry_price=self.current_price,
                    position_size=btc_to_buy,
                    trade_type='LONG'
                )
                
                result['type'] = 'BUY'
                result['success'] = True
                result['details'] = {
                    'amount': btc_to_buy,
                    'price': self.current_price,
                    'cost': max_buy,
                    'strength': buy_strength
                }
                
                print(f"🟢 BUY: {btc_to_buy:.6f} BTC @ ${self.current_price:.2f} | Cost: ${max_buy:.2f} | Strength: {buy_strength:.2f}")
                
        elif action < -0.1:
            self.consecutive_holds = 0
            
            if self.btc_amount < 0.0001:  # Минимальное количество для продажи
                result['type'] = 'SELL_FAILED'
                result['success'] = False
                result['details']['reason'] = 'no_position'
                
                if self.detailed_logging:
                    print(f"⛔ SELL FAILED: Нет позиции для продажи")
            else:
                sell_strength = min(1.0, max(0.1, abs(action)))
                btc_to_sell = min(self.btc_amount, self.btc_amount * sell_strength)
                
                sell_amount = btc_to_sell * self.current_price
                commission = sell_amount * 0.001
                
                self.balance += (sell_amount - commission)
                self.btc_amount -= btc_to_sell
                
                closed_trades = []
                trades_to_close = max(1, int(len(self.trade_tracker.active_trades) * sell_strength))
                
                for _ in range(min(trades_to_close, len(self.trade_tracker.active_trades))):
                    trade_result = self.trade_tracker.close_trade(
                        exit_price=self.current_price,
                        commission=commission / max(1, trades_to_close)
                    )
                    if trade_result:
                        closed_trades.append(trade_result)
                
                result['type'] = 'SELL'
                result['success'] = True
                result['details'] = {
                    'amount': btc_to_sell,
                    'price': self.current_price,
                    'revenue': sell_amount,
                    'closed_trades': closed_trades,
                    'strength': sell_strength
                }
                
                if closed_trades:
                    total_pnl = sum(t['pnl_after_commission'] for t in closed_trades)
                    avg_pnl_pct = np.mean([t['pnl_percent'] for t in closed_trades])
                    print(f"🔴 SELL: {btc_to_sell:.6f} BTC @ ${self.current_price:.2f} | Strength: {sell_strength:.2f}")
                    print(f"   📊 Closed {len(closed_trades)} trades | Total PnL: ${total_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
                else:
                    print(f"🔴 SELL: {btc_to_sell:.6f} BTC @ ${self.current_price:.2f} | Strength: {sell_strength:.2f}")
        
        self.last_action_type = result['type']
        return result
    
    def _calculate_reward(self, action: float, action_result: Dict,
                         prev_portfolio: float, prev_balance: float,
                         prev_btc: float) -> float:
        """
        УЛУЧШЕННАЯ СБАЛАНСИРОВАННАЯ СИСТЕМА НАГРАД
        Все компоненты нормализованы к масштабу [-10, +10]
        """
        reward = 0.0

        current_portfolio = self._get_portfolio_value()
        portfolio_change_pct = ((current_portfolio - prev_portfolio) / prev_portfolio) * 100
        reward += np.clip(portfolio_change_pct * 10, -20, 20)

        if action_result['type'] == 'SELL' and action_result['success']:
            closed_trades = action_result['details'].get('closed_trades', [])

            for trade in closed_trades:
                pnl_pct = trade['pnl_percent']

                if pnl_pct > 0:
                    trade_reward = np.clip(pnl_pct * 3, 0, 15)
                else:
                    trade_reward = np.clip(pnl_pct * 3, -15, 0)

                reward += trade_reward

                if pnl_pct > 0.5 and trade['steps_held'] > 10:
                    patience_bonus = min(5, trade['steps_held'] * 0.2)
                    reward += patience_bonus

                if pnl_pct < -1 and trade['steps_held'] < 5:
                    reward -= 3

        if abs(action) <= 0.1 and self.trade_tracker.active_trades:
            unrealized_pnl = self.trade_tracker.get_unrealized_pnl(self.current_price)
            unrealized_pct = (unrealized_pnl / prev_portfolio) * 100

            if unrealized_pct > 1:  # Прибыль > 1%
                reward += min(5, unrealized_pct)  # Поощряем терпение
            elif unrealized_pct < -2:  # Убыток > 2%
                reward -= min(5, abs(unrealized_pct) * 0.5)  # Наказываем удержание убытков

        if not action_result['success']:
            if abs(action) > 0.3:  # Сильный сигнал, но не сработал
                reward -= 3
            elif abs(action) > 0.1:  # Слабый сигнал
                reward -= 1

        if self.total_steps < 1000 and action_result['success'] and abs(action) > 0.1:
            reward += 1

        if self.consecutive_holds > 30 and not self.trade_tracker.active_trades:
            reward -= 2

        reward = np.clip(reward, -50, 50)

        return reward
    
    def _update_market_data(self):
        """Обновление рыночных данных с защитой от ошибок"""
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                new_price = ticker.get('last', None)

                if new_price and isinstance(new_price, (int, float)) and new_price > 0:
                    self.current_price = float(new_price)
                else:
                    if self.current_price is None or self.current_price <= 0:
                        self.current_price = 67000.0
            else:
                if self.current_price is None or len(self.price_history) == 0:
                    self.current_price = 67000.0
                else:
                    trend = np.random.choice([-0.0001, 0, 0.0001], p=[0.3, 0.4, 0.3])
                    volatility = np.random.normal(0, 0.002)
                    new_price = self.current_price * (1 + trend + volatility)
                    self.current_price = max(50000, min(80000, new_price))

            if self.current_price and self.current_price > 0:
                self.price_history.append(self.current_price)
                if len(self.price_history) > 100:
                    self.price_history = self.price_history[-100:]

        except Exception as e:
            print(f"⚠️ Ошибка получения данных: {e}")
            if self.current_price is None or self.current_price <= 0:
                self.current_price = 67000.0
                if len(self.price_history) == 0:
                    self.price_history.append(self.current_price)
    
    def _get_portfolio_value(self) -> float:
        """Общая стоимость портфеля с защитой от ошибок"""
        if self.current_price is None or self.current_price <= 0:
            self.current_price = 67000.0
        return self.balance + (self.btc_amount * self.current_price)
    
    def _get_observation(self) -> np.ndarray:
        """Расширенное наблюдение с полной информацией"""
        obs = []
        
        obs.append(self.current_price / 70000)  # Нормализация около среднего
        
        for lookback in [1, 2, 5, 10, 20]:
            if len(self.price_history) > lookback:
                price_change = (self.current_price - self.price_history[-lookback]) / self.price_history[-lookback]
                obs.append(price_change * 100)
            else:
                obs.append(0)
        
        if len(self.price_history) >= 20:
            sma_5 = np.mean(self.price_history[-5:])
            sma_20 = np.mean(self.price_history[-20:])
            obs.append((self.current_price - sma_5) / sma_5)
            obs.append((self.current_price - sma_20) / sma_20)
            obs.append((sma_5 - sma_20) / sma_20)
        else:
            obs.extend([0, 0, 0])
        
        if len(self.price_history) >= 10:
            returns = np.diff(self.price_history[-10:]) / self.price_history[-10:-1]
            obs.append(np.std(returns) * 100)
        else:
            obs.append(0)
        
        portfolio_value = self._get_portfolio_value()
        
        obs.extend([
            self.balance / self.initial_balance,
            self.btc_amount * self.current_price / self.initial_balance if self.current_price > 0 else 0,
            (portfolio_value - self.initial_balance) / self.initial_balance,  # ROI
            len(self.trade_tracker.active_trades) / 10,  # Нормализованное количество сделок
        ])
        
        if self.trade_tracker.active_trades:
            unrealized = self.trade_tracker.get_unrealized_pnl(self.current_price)
            unrealized_pct = unrealized / self.initial_balance
            avg_entry = np.mean([t.entry_price for t in self.trade_tracker.active_trades])
            avg_hold_time = np.mean([t.steps_held for t in self.trade_tracker.active_trades])
            
            obs.extend([
                unrealized_pct * 10,  # Усиливаем сигнал
                (self.current_price - avg_entry) / avg_entry,
                avg_hold_time / 100,
            ])
        else:
            obs.extend([0, 0, 0])
        
        max_buy_amount = self.balance * 0.3
        can_buy = 1.0 if max_buy_amount >= 100 else 0.0
        can_sell = 1.0 if self.btc_amount >= 0.0001 else 0.0
        
        obs.extend([can_buy, can_sell, max_buy_amount / self.initial_balance])
        
        if self.trade_tracker.closed_trades:
            recent_trades = self.trade_tracker.closed_trades[-10:]
            
            wins = sum(1 for t in recent_trades if t['pnl_after_commission'] > 0)
            win_rate = wins / len(recent_trades)
            
            avg_win = np.mean([t['pnl_percent'] for t in recent_trades if t['pnl_after_commission'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl_percent'] for t in recent_trades if t['pnl_after_commission'] <= 0]) if wins < len(recent_trades) else 0
            
            total_pnl = sum(t['pnl_after_commission'] for t in recent_trades)
            
            obs.extend([
                win_rate,
                avg_win / 100,
                avg_loss / 100,
                total_pnl / self.initial_balance,
                len(self.trade_tracker.closed_trades) / 100,
            ])
        else:
            obs.extend([0, 0, 0, 0, 0])
        
        if len(self.action_history) >= 10:
            recent_actions = self.action_history[-10:]
            action_distribution = [
                recent_actions.count(0) / 10,  # % HOLD
                recent_actions.count(1) / 10,  # % BUY
                recent_actions.count(2) / 10,  # % SELL
            ]
            obs.extend(action_distribution)
        else:
            obs.extend([0, 0, 0])
        
        if len(self.reward_history) >= 10:
            avg_recent_reward = np.mean(self.reward_history[-10:])
            obs.append(avg_recent_reward / 100)  # Нормализация
        else:
            obs.append(0)
        
        obs.append(1.0 if self.last_action_type in ['BUY', 'SELL', 'HOLD'] else -1.0)
        
        obs.extend([
            self.consecutive_holds / 20,  # Нормализованное количество HOLD подряд
            self.total_steps / 1000,  # Прогресс обучения
            np.sin(self.total_steps * 0.01),  # Временной сигнал для циклических паттернов
            np.cos(self.total_steps * 0.01),
        ])
        
        while len(obs) < 40:
            obs.append(0.0)
        
        return np.array(obs[:40], dtype=np.float32)


class CustomCallback(BaseCallback):
    """Улучшенный callback для детального мониторинга обучения"""

    def __init__(self, verbose=0, log_freq=500):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_freq = log_freq
        self.best_mean_reward = -np.inf
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                portfolio = info.get('portfolio_value', 0)
                active_trades = info.get('active_trades', 0)
                closed_trades = info.get('closed_trades', 0)

                print(f"\n{'='*60}")
                print(f"📊 ПРОГРЕСС ОБУЧЕНИЯ - Шаг {self.n_calls}")
                print(f"{'='*60}")
                print(f"🏆 Средняя награда (20 эпизодов): {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"💰 Портфель: ${portfolio:.2f}")
                print(f"📈 Активных сделок: {active_trades} | Закрытых: {closed_trades}")
                print(f"🎯 Лучшая средняя награда: {self.best_mean_reward:.2f}")
                print(f"{'='*60}\n")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"🌟 НОВЫЙ РЕКОРД! Средняя награда: {mean_reward:.2f}\n")

        return True

    def _on_rollout_end(self) -> None:
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_rew = info['episode']['r']
                    ep_len = info['episode']['l']
                    self.episode_rewards.append(ep_rew)
                    self.episode_lengths.append(ep_len)
                    self.episode_count += 1


class ImprovedTradingAgent:
    """Улучшенный агент с оптимизированными параметрами"""

    def __init__(self, env: ImprovedTradingEnvironment, model_path: Optional[str] = None):
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Используется: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        policy_kwargs = dict(
            activation_fn=nn.Tanh,  # Tanh лучше для ограниченных действий
            net_arch=dict(
                pi=[256, 128],  # Меньше слоев для быстрого обучения
                qf=[256, 128]
            ),
            optimizer_kwargs=dict(
                eps=1e-5  # Для стабильности
            )
        )

        if model_path and os.path.exists(model_path):
            self.model = SAC.load(model_path, env=env, device=self.device)
            print(f"📂 Модель загружена из {model_path}")
        else:
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=1e-4,  # Меньше learning rate для стабильности
                buffer_size=100000,  # Больший buffer для лучшего обучения
                learning_starts=1000,  # Больше initial exploration
                batch_size=256,      # Больший batch для стабильности
                tau=0.005,          # Медленнее обновление target network
                gamma=0.99,         # Выше для долгосрочных наград
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',    # Автоматическая настройка энтропии
                target_update_interval=1,
                target_entropy='auto',
                use_sde=False,
                policy_kwargs=policy_kwargs,
                device=self.device,  # ✅ Используем GPU
                verbose=1
            )
            print("🆕 Создан новый улучшенный SAC агент")
        
        self.callback = CustomCallback()
    
    def train(self, total_timesteps: int = 10000, save_best: bool = True):
        """Обучение агента с автосохранением лучшей модели"""
        print(f"\n🎓 Начинаем обучение на {total_timesteps} шагов...")
        print(f"💾 Автосохранение лучшей модели: {'Вкл' if save_best else 'Выкл'}\n")

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callback,
                progress_bar=True,
                reset_num_timesteps=False  # Продолжаем с текущего шага
            )

            print("\n✅ Обучение завершено!")
            self._print_training_stats()

            if save_best:
                self.save("./models/improved_bot_latest")
                if self.callback.best_mean_reward > -np.inf:
                    print(f"💾 Модель автоматически сохранена")

        except KeyboardInterrupt:
            print("\n⚠️ Обучение прервано пользователем")
            if save_best:
                self.save("./models/improved_bot_interrupted")
                print("💾 Прогресс сохранен")

    def _print_training_stats(self):
        """Вывод детальной статистики обучения"""
        if self.callback.episode_rewards:
            rewards = self.callback.episode_rewards
            recent_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            recent_20 = rewards[-20:] if len(rewards) >= 20 else rewards

            print(f"""
{'='*60}
📈 ФИНАЛЬНАЯ СТАТИСТИКА ОБУЧЕНИЯ
{'='*60}
📊 Всего эпизодов: {len(rewards)}
🏆 Лучшая награда: {max(rewards):.2f}
💔 Худшая награда: {min(rewards):.2f}
📈 Средняя награда (последние 100): {np.mean(recent_100):.2f} ± {np.std(recent_100):.2f}
🎯 Средняя награда (последние 20): {np.mean(recent_20):.2f} ± {np.std(recent_20):.2f}
🌟 Рекорд средней награды: {self.callback.best_mean_reward:.2f}
{'='*60}
            """)
    
    def save(self, path: str = "./models/improved_bot"):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"💾 Модель сохранена: {path}")
    
    def test(self, episodes: int = 5):
        """Тестирование агента"""
        print(f"\n🧪 Тестирование на {episodes} эпизодах...")
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            print(f"\n📍 Эпизод {episode + 1}")
            
            while not done and not truncated and steps < 200:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if steps % 20 == 0:
                    print(f"  Шаг {steps}: Portfolio=${info['portfolio_value']:.2f}, Reward={reward:+.2f}")
            
            final_value = self.env._get_portfolio_value()
            roi = (final_value - self.env.initial_balance) / self.env.initial_balance * 100
            
            print(f"""
📊 Результат эпизода {episode + 1}:
  💰 Финальный баланс: ${final_value:.2f}
  📈 ROI: {roi:+.2f}%
  🏆 Общая награда: {total_reward:.2f}
  📍 Шагов: {steps}
  📊 Закрыто сделок: {len(self.env.trade_tracker.closed_trades)}
            """)


def main():
    """Главная функция для обучения и тестирования"""
    print("""
🚀 УЛУЧШЕННЫЙ ТОРГОВЫЙ БОТ v6.0
════════════════════════════════
✅ GPU Acceleration (CUDA)
✅ Исправлена система наград
✅ Улучшено пространство наблюдений
✅ Оптимизированы гиперпараметры
✅ Детальный мониторинг обучения
════════════════════════════════
    """)

    env = ImprovedTradingEnvironment(initial_balance=10000, detailed_logging=False)

    model_path = "./models/improved_bot_latest.zip"
    if os.path.exists(model_path):
        print(f"📂 Найдена сохраненная модель: {model_path}")
        response = input("Продолжить обучение с этой модели? (y/n): ").strip().lower()
        if response == 'y':
            agent = ImprovedTradingAgent(env, model_path=model_path)
            print("✅ Загружена сохраненная модель, продолжаем обучение\n")
        else:
            agent = ImprovedTradingAgent(env)
            print("✅ Создана новая модель\n")
    else:
        agent = ImprovedTradingAgent(env)

    print("\n1️⃣ ФАЗА ОБУЧЕНИЯ")
    print("💡 Совет: Прервать обучение можно с помощью Ctrl+C (прогресс сохранится)\n")

    agent.train(total_timesteps=50000)  # Увеличиваем до 50k шагов

    print("\n2️⃣ ФАЗА ТЕСТИРОВАНИЯ")
    agent.test(episodes=5)

    print("\n✅ Готово! Модель сохранена в ./models/improved_bot_latest.zip")
    print("💡 Запустите снова для продолжения обучения или тестирования")


if __name__ == "__main__":
    main()
