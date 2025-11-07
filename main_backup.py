#!/usr/bin/env python3
"""
🚀 АВТОНОМНЫЙ ТОРГОВЫЙ БОТ v3.1
Полная свобода действий + честная система наград
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
from typing import Dict, Tuple, Optional
import json

# Импорты для RL
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn

# Конфигурация из config.py
from config import Config
from dataclasses import dataclass
from typing import List


@dataclass
class Trade:
    """Класс для отслеживания отдельной сделки"""
    entry_price: float
    entry_time: int
    position_size: float  # Количество BTC
    trade_type: str  # 'LONG' или 'SHORT'
    stop_loss: float = 0.0
    take_profit: float = 0.0
    # Новые поля для отслеживания качества сделки
    had_stop_loss: bool = False  # Был ли стоп-лосс при открытии
    had_take_profit: bool = False  # Был ли тейк-профит при открытии
    max_unrealized_profit: float = 0.0  # Максимальная нереализованная прибыль
    min_unrealized_loss: float = 0.0  # Максимальный нереализованный убыток
    # Система контрактных обязательств
    min_hold_steps: int = 0  # Минимум шагов для удержания
    steps_held: int = 0  # Сколько шагов уже держим


class TradeTracker:
    """Улучшенный трекер для отслеживания торговых позиций"""

    def __init__(self):
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Dict] = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_commission_paid = 0.0
        
        # Новая статистика для анализа качества торговли
        self.trades_with_risk_management = 0  # Сделки со стоп-лоссом/тейк-профитом
        self.stopped_out_trades = 0  # Сделки закрытые по стоп-лоссу
        self.take_profit_trades = 0  # Сделки закрытые по тейк-профиту
        self.manual_close_trades = 0  # Сделки закрытые вручную

    def open_trade(self, entry_price: float, position_size: float,
                   trade_type: str, stop_loss: float = 0.0, take_profit: float = 0.0) -> Trade:
        """Открыть новую сделку с правильным учетом риск-менеджмента"""
        # Если есть RM, требуем минимум 5 шагов удержания
        min_hold = 5 if (stop_loss > 0 or take_profit > 0) else 0

        trade = Trade(
            entry_price=entry_price,
            entry_time=int(time.time()),
            position_size=position_size,
            trade_type=trade_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            had_stop_loss=(stop_loss > 0),
            had_take_profit=(take_profit > 0),
            min_hold_steps=min_hold,
            steps_held=0
        )
        
        # Учитываем статистику риск-менеджмента
        if stop_loss > 0 or take_profit > 0:
            self.trades_with_risk_management += 1
        
        self.active_trades.append(trade)
        return trade

    def update_unrealized_pnl(self, current_price: float):
        """Обновить максимальную нереализованную прибыль/убыток для активных сделок"""
        for trade in self.active_trades:
            if trade.trade_type == 'LONG':
                unrealized_pnl = (current_price - trade.entry_price) / trade.entry_price
            else:  # SHORT
                unrealized_pnl = (trade.entry_price - current_price) / trade.entry_price

            # Обновляем максимумы
            trade.max_unrealized_profit = max(trade.max_unrealized_profit, unrealized_pnl)
            trade.min_unrealized_loss = min(trade.min_unrealized_loss, unrealized_pnl)

            # Увеличиваем счетчик шагов удержания
            trade.steps_held += 1

    def close_trade(self, exit_price: float, commission: float = 0.0, 
                   reason: str = 'manual') -> Dict:
        """Закрыть сделку с указанием причины закрытия"""
        if not self.active_trades:
            return {}

        trade = self.active_trades.pop(0)  # FIFO
        
        # Рассчитываем PnL
        if trade.trade_type == 'LONG':
            pnl = trade.position_size * (exit_price - trade.entry_price)
        else:  # SHORT
            pnl = trade.position_size * (trade.entry_price - exit_price)
        
        pnl_percent = pnl / (trade.position_size * trade.entry_price) * 100
        pnl_after_commission = pnl - commission
        
        # Обновляем статистику по типу закрытия
        if reason == 'stop_loss':
            self.stopped_out_trades += 1
        elif reason == 'take_profit':
            self.take_profit_trades += 1
        else:
            self.manual_close_trades += 1
        
        # Обновляем серии
        if pnl_after_commission > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        self.total_commission_paid += commission
        
        # Проверяем нарушение контрактных обязательств
        contract_violated = (trade.steps_held < trade.min_hold_steps and
                           reason == 'manual' and
                           abs(pnl_percent) < 5.0)  # Исключение для больших движений

        trade_result = {
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'pnl_after_commission': pnl_after_commission,
            'duration': int(time.time()) - trade.entry_time,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'commission': commission,
            'position_size': trade.position_size,
            'trade_type': trade.trade_type,
            'close_reason': reason,
            'had_risk_management': trade.had_stop_loss or trade.had_take_profit,
            'max_unrealized_profit': trade.max_unrealized_profit,
            'min_unrealized_loss': trade.min_unrealized_loss,
            'gave_back_profit': trade.max_unrealized_profit - pnl_percent/100,  # Сколько прибыли упустили
            'steps_held': trade.steps_held,
            'min_hold_required': trade.min_hold_steps,
            'contract_violated': contract_violated
        }
        
        self.closed_trades.append(trade_result)
        return trade_result

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Получить общую нереализованную прибыль/убыток"""
        total_unrealized = 0.0
        for trade in self.active_trades:
            if trade.trade_type == 'LONG':
                unrealized = trade.position_size * (current_price - trade.entry_price)
            else:
                unrealized = trade.position_size * (trade.entry_price - current_price)
            total_unrealized += unrealized
        return total_unrealized
    
    def get_risk_management_quality(self) -> float:
        """Оценка качества риск-менеджмента (0-1)"""
        if not self.closed_trades:
            return 0.0
        
        total_trades = len(self.closed_trades)
        
        # Факторы качества
        rm_ratio = self.trades_with_risk_management / max(1, total_trades)
        
        # Процент сделок закрытых по тейк-профиту (хорошо)
        tp_ratio = self.take_profit_trades / max(1, total_trades)
        
        # Средняя упущенная прибыль (плохо если большая)
        avg_gave_back = np.mean([t.get('gave_back_profit', 0) for t in self.closed_trades[-10:]])
        gave_back_penalty = max(0, min(1, 1 - avg_gave_back * 10))  # Штраф за упущенную прибыль
        
        # Комбинированная оценка
        quality = (rm_ratio * 0.3 + tp_ratio * 0.3 + gave_back_penalty * 0.4)
        return min(1.0, max(0.0, quality))


class TradingEnvironment(gym.Env):
    """
    Торговое окружение с честной системой наград
    """

    def __init__(self, initial_balance: float = 10000, detailed_logging: bool = True):
        super().__init__()

        self.initial_balance = initial_balance
        self.detailed_logging = detailed_logging

        # Пространство действий остается прежним - полная свобода
        self.action_space = spaces.Box(
            low=np.array([
                -1.0,   # position_ratio: -100% (sell all) to +100% (buy all)
                0.0,    # stop_loss_pct: 0% to 20%
                0.0,    # take_profit_pct: 0% to 50%
            ]),
            high=np.array([
                1.0,    # position_ratio
                0.2,    # stop_loss_pct (max 20%)
                0.5,    # take_profit_pct (max 50%)
            ]),
            dtype=np.float32
        )

        # Пространство наблюдений (30 параметров для большей информативности)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        # Инициализация
        self._init_exchange()
        self._init_state()

    def _init_exchange(self):
        """Инициализация подключения к Binance"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': Config.API_KEY,
                'secret': Config.API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            })
            self.exchange.set_sandbox_mode(True)
            markets = self.exchange.load_markets()
            print(f"✅ Подключен к Binance Testnet, доступно {len(markets)} рынков")
        except Exception as e:
            print(f"⚠️ Ошибка подключения к Binance: {e}")
            print("📊 Работаем в режиме симуляции")
            self.exchange = None

    def _init_state(self):
        """Инициализация состояния"""
        self.balance = self.initial_balance
        self.btc_amount = 0.0
        self.current_price = 0.0
        
        # Статистика
        self.total_trades = 0
        self.total_profit_loss = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        
        # История
        self.price_history = []
        self.action_history = []
        
        # Трекер сделок
        self.trade_tracker = TradeTracker()
        
        # Детализация последней награды для отладки
        self.last_reward_breakdown = {
            'trade_result': 0.0,
            'holding_penalty': 0.0,
            'risk_reward': 0.0,
            'efficiency': 0.0
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        super().reset(seed=seed)
        self._init_state()
        self._update_market_data()
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение действия"""
        prev_portfolio_value = self._get_portfolio_value()
        
        # Обновляем рыночные данные
        self._update_market_data()
        
        # Обновляем нереализованный PnL для активных сделок
        self.trade_tracker.update_unrealized_pnl(self.current_price)
        
        # Проверяем стоп-ордера
        closed_trades = self._check_and_execute_stops()
        
        # Исполняем новое действие
        new_trade_opened = self._execute_action(action)
        
        # Рассчитываем награду
        current_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(
            prev_portfolio_value, 
            current_portfolio_value, 
            action,
            closed_trades,
            new_trade_opened
        )
        
        # Обновляем статистику
        self._update_statistics(current_portfolio_value)
        
        # Сохраняем историю действий
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # Условия завершения
        terminated = False  # Никогда не завершаем
        truncated = current_portfolio_value < self.initial_balance * 0.05  # При потере 95%
        
        info = {
            'portfolio_value': current_portfolio_value,
            'btc_amount': self.btc_amount,
            'current_price': self.current_price,
            'active_trades': len(self.trade_tracker.active_trades),
            'total_trades': self.total_trades,
            'win_rate': len([t for t in self.trade_tracker.closed_trades if t['pnl_after_commission'] > 0]) / max(1, len(self.trade_tracker.closed_trades)),
            'roi': (current_portfolio_value - self.initial_balance) / self.initial_balance,
            'risk_management_quality': self.trade_tracker.get_risk_management_quality(),
            'reward_breakdown': self.last_reward_breakdown.copy()
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _execute_action(self, action: np.ndarray) -> bool:
        """
        Исполнение действия с детальным логированием
        Возвращает True если открыта новая сделка
        """
        position_ratio, stop_loss_pct, take_profit_pct = action

        if self.current_price <= 0:
            if self.detailed_logging:
                print(f"⚫ SKIP: Invalid price ${self.current_price:.2f}")
            return False

        # Определяем размер позиции
        portfolio_value = self._get_portfolio_value()

        # ПОКУПКА (LONG)
        if position_ratio > 0.05:  # Минимум 5% для открытия позиции
            max_buy_amount = min(self.balance, portfolio_value * abs(position_ratio))

            if max_buy_amount <= 50:  # Слишком маленькая сумма
                if self.detailed_logging:
                    print(f"⚫ SKIP BUY: ${max_buy_amount:.2f} < $50 minimum | Pos:{position_ratio:+.2f}")
                return False

            btc_to_buy = max_buy_amount / self.current_price
            commission = max_buy_amount * 0.001

            if max_buy_amount + commission > self.balance:
                if self.detailed_logging:
                    print(f"⚫ SKIP BUY: Need ${max_buy_amount + commission:.2f}, have ${self.balance:.2f}")
                return False

            # Исполняем покупку
            self.balance -= (max_buy_amount + commission)
            self.btc_amount += btc_to_buy
            self.total_trades += 1

            # Рассчитываем уровни стоп-лосса и тейк-профита
            sl_price = self.current_price * (1 - stop_loss_pct) if stop_loss_pct > 0.005 else 0
            tp_price = self.current_price * (1 + take_profit_pct) if take_profit_pct > 0.005 else 0

            # Регистрируем сделку в трекере
            self.trade_tracker.open_trade(
                entry_price=self.current_price,
                position_size=btc_to_buy,
                trade_type='LONG',
                stop_loss=sl_price,
                take_profit=tp_price
            )

            self.trade_tracker.total_commission_paid += commission

            # Логирование с деталями риск-менеджмента
            risk_info = ""
            if sl_price > 0:
                risk_info += f" SL@${sl_price:.2f}(-{stop_loss_pct*100:.1f}%)"
            if tp_price > 0:
                risk_info += f" TP@${tp_price:.2f}(+{take_profit_pct*100:.1f}%)"

            print(f"🟢 LONG: {btc_to_buy:.6f} BTC @ ${self.current_price:.2f}{risk_info}")
            return True

        # ПРОДАЖА (закрытие позиции)
        elif position_ratio < -0.05:
            if self.btc_amount <= 0:
                if self.detailed_logging:
                    print(f"⚫ SKIP SELL: No BTC to sell | Pos:{position_ratio:+.2f}")
                return False

            sell_ratio = min(abs(position_ratio), 1.0)
            btc_to_sell = min(self.btc_amount, self.btc_amount * sell_ratio)

            if btc_to_sell <= 0.0001:  # Слишком мало для продажи
                if self.detailed_logging:
                    print(f"⚫ SKIP SELL: {btc_to_sell:.6f} BTC < 0.0001 minimum")
                return False

            sell_amount = btc_to_sell * self.current_price
            commission = sell_amount * 0.001

            # Исполняем продажу
            self.balance += (sell_amount - commission)
            self.btc_amount -= btc_to_sell
            self.total_trades += 1

            # Закрываем соответствующие сделки в трекере
            trades_to_close = int(len(self.trade_tracker.active_trades) * sell_ratio)
            if trades_to_close == 0 and len(self.trade_tracker.active_trades) > 0:
                trades_to_close = 1

            closed_info = []
            for _ in range(min(trades_to_close, len(self.trade_tracker.active_trades))):
                trade_result = self.trade_tracker.close_trade(
                    exit_price=self.current_price,
                    commission=commission / max(1, trades_to_close),
                    reason='manual'
                )
                if trade_result:
                    closed_info.append(trade_result)

            # Логирование с результатами
            if closed_info:
                avg_pnl = np.mean([t['pnl_percent'] for t in closed_info])
                print(f"🔴 CLOSE {len(closed_info)} trades @ ${self.current_price:.2f} | Avg PnL: {avg_pnl:+.2f}%")

            return False  # Закрытие позиции не считается открытием новой сделки

        # HOLD - никаких действий
        else:
            if self.detailed_logging:
                print(f"⚪ HOLD @ ${self.current_price:.2f} | Pos:{position_ratio:+.2f} SL:{stop_loss_pct:.1%} TP:{take_profit_pct:.1%}")
            return False

    def _check_and_execute_stops(self) -> List[Dict]:
        """Проверка и исполнение стоп-ордеров"""
        closed_trades = []
        
        if not self.trade_tracker.active_trades:
            return closed_trades
        
        trades_to_close = []
        
        for i, trade in enumerate(self.trade_tracker.active_trades):
            # Проверка стоп-лосса
            if trade.stop_loss > 0 and self.current_price <= trade.stop_loss:
                trades_to_close.append((i, 'stop_loss'))
                print(f"🛑 STOP LOSS triggered @ ${self.current_price:.2f}")
            
            # Проверка тейк-профита
            elif trade.take_profit > 0 and self.current_price >= trade.take_profit:
                trades_to_close.append((i, 'take_profit'))
                print(f"🎯 TAKE PROFIT triggered @ ${self.current_price:.2f}")
        
        # Закрываем сработавшие ордера (в обратном порядке чтобы не сбить индексы)
        for idx, reason in reversed(trades_to_close):
            trade = self.trade_tracker.active_trades[idx]
            
            # Продаем соответствующую часть BTC
            if self.btc_amount >= trade.position_size:
                sell_amount = trade.position_size * self.current_price
                commission = sell_amount * 0.001
                
                self.balance += (sell_amount - commission)
                self.btc_amount -= trade.position_size
                
                # Удаляем из активных и записываем результат
                self.trade_tracker.active_trades.pop(idx)
                trade_result = self.trade_tracker.close_trade(
                    exit_price=self.current_price,
                    commission=commission,
                    reason=reason
                )
                
                if trade_result:
                    closed_trades.append(trade_result)
                    self.total_trades += 1
        
        return closed_trades

    def _calculate_reward(self, prev_value: float, current_value: float,
                         action: np.ndarray, closed_trades: List[Dict],
                         new_trade_opened: bool) -> float:
        """
        РЕВОЛЮЦИОННАЯ СИСТЕМА НАГРАД v2.0
        Награды только за РЕЗУЛЬТАТЫ, штрафы за нарушение собственных правил
        """
        total_reward = 0.0
        self.last_reward_breakdown = {
            'trade_result': 0.0,
            'contract_penalty': 0.0,
            'discipline_bonus': 0.0,
            'efficiency': 0.0
        }

        # 1. НАГРАДА ЗА РЕЗУЛЬТАТЫ ЗАКРЫТЫХ СДЕЛОК
        if closed_trades:
            for trade in closed_trades:
                pnl_percent = trade['pnl_percent']

                # Базовая награда за результат сделки (без бонусов за RM!)
                if pnl_percent > 0:
                    if pnl_percent > 5:
                        trade_reward = 200
                    elif pnl_percent > 2:
                        trade_reward = 100
                    elif pnl_percent > 1:
                        trade_reward = 50
                    else:
                        trade_reward = 20
                else:
                    if pnl_percent < -5:
                        trade_reward = -200
                    elif pnl_percent < -2:
                        trade_reward = -100
                    elif pnl_percent < -1:
                        trade_reward = -50
                    else:
                        trade_reward = -20

                # ШТРАФ ЗА НАРУШЕНИЕ КОНТРАКТНЫХ ОБЯЗАТЕЛЬСТВ!
                if trade['contract_violated']:
                    violation_penalty = -50  # Серьезный штраф за досрочное закрытие
                    trade_reward += violation_penalty
                    self.last_reward_breakdown['contract_penalty'] += violation_penalty
                    print(f"⚠️ CONTRACT VIOLATED: Closed after {trade['steps_held']}/{trade['min_hold_required']} steps → {violation_penalty}")

                # БОНУС ЗА ДИСЦИПЛИНУ - следование собственному плану!
                if trade['close_reason'] in ['stop_loss', 'take_profit']:
                    discipline_bonus = 50  # Большая награда за следование плану
                    trade_reward += discipline_bonus
                    self.last_reward_breakdown['discipline_bonus'] += discipline_bonus
                    print(f"🎖️ DISCIPLINE BONUS: {trade['close_reason'].upper()} executed as planned → +{discipline_bonus}")

                self.last_reward_breakdown['trade_result'] += trade_reward
                total_reward += trade_reward

        # 2. БЕЗ ФИКТИВНЫХ НАГРАД ЗА ОТКРЫТИЕ! ✅

        # 3. ШТРАФ ЗА ХАОТИЧНУЮ ТОРГОВЛЮ (анти-карусель)
        if new_trade_opened and len(self.trade_tracker.closed_trades) > 0:
            last_trade = self.trade_tracker.closed_trades[-1]

            # Если последняя сделка была закрыта недавно без движения цены
            if (last_trade['steps_held'] < 3 and
                abs(last_trade['pnl_percent']) < 0.1):  # Почти ноль результат

                carousel_penalty = -0.10
                total_reward += carousel_penalty
                self.last_reward_breakdown['efficiency'] += carousel_penalty
                print(f"🎠 CAROUSEL PENALTY: Quick open→close→open pattern → {carousel_penalty}")

        # 4. ШТРАФ ЗА УДЕРЖАНИЕ УБЫТОЧНЫХ ПОЗИЦИЙ БЕЗ ЗАЩИТЫ
        if self.trade_tracker.active_trades:
            for trade in self.trade_tracker.active_trades:
                current_pnl = (self.current_price - trade.entry_price) / trade.entry_price

                # Если позиция убыточная и нет стоп-лосса
                if current_pnl < -0.03 and trade.stop_loss == 0:
                    unprotected_penalty = -20
                    total_reward += unprotected_penalty
                    self.last_reward_breakdown['efficiency'] += unprotected_penalty

        # 5. МЯГКИЙ СИГНАЛ ОБ ИЗМЕНЕНИИ ПОРТФЕЛЯ (сильно уменьшен)
        portfolio_change = (current_value - prev_value) / self.initial_balance
        total_reward += portfolio_change * 5  # Уменьшено с 10 до 5

        return total_reward

    def _update_market_data(self):
        """Получение актуальных данных"""
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker('BTC/USDT')
                self.current_price = ticker['last']
            else:
                # Симуляция цены
                if len(self.price_history) == 0:
                    self.current_price = 67000.0
                else:
                    # Реалистичная волатильность
                    returns = np.random.normal(0, 0.001)  # 0.1% стандартное отклонение
                    self.current_price = self.current_price * (1 + returns)
            
            self.price_history.append(self.current_price)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]
                
        except Exception as e:
            print(f"⚠️ Ошибка получения данных: {e}")
            if self.current_price <= 0:
                self.current_price = 67000.0

    def _get_portfolio_value(self) -> float:
        """Общая стоимость портфеля"""
        return self.balance + (self.btc_amount * self.current_price)

    def _get_observation(self) -> np.ndarray:
        """Расширенное наблюдение с акцентом на качество торговли"""
        obs = []
        
        # Базовая информация о цене
        obs.append(self.current_price / 100000)  # Нормализованная цена
        
        # Тренд цены (последние 5 изменений)
        if len(self.price_history) >= 6:
            for i in range(1, 6):
                change = (self.price_history[-i] - self.price_history[-i-1]) / self.price_history[-i-1]
                obs.append(change * 100)  # Усиливаем сигнал
        else:
            obs.extend([0] * 5)
        
        # Состояние портфеля
        portfolio_value = self._get_portfolio_value()
        obs.extend([
            self.balance / self.initial_balance - 1,
            (self.btc_amount * self.current_price) / self.initial_balance if self.current_price > 0 else 0,
            (portfolio_value - self.initial_balance) / self.initial_balance,
        ])
        
        # Информация об активных сделках
        obs.append(len(self.trade_tracker.active_trades) / 10)  # Нормализованное количество
        
        if self.trade_tracker.active_trades:
            # Средний нереализованный PnL
            unrealized_pnls = []
            for trade in self.trade_tracker.active_trades:
                pnl = (self.current_price - trade.entry_price) / trade.entry_price
                unrealized_pnls.append(pnl)
            obs.append(np.mean(unrealized_pnls))
            
            # Процент позиций с риск-менеджментом
            rm_positions = sum(1 for t in self.trade_tracker.active_trades if t.had_stop_loss or t.had_take_profit)
            obs.append(rm_positions / len(self.trade_tracker.active_trades))
        else:
            obs.extend([0, 0])
        
        # История торговли
        if self.trade_tracker.closed_trades:
            recent_trades = self.trade_tracker.closed_trades[-5:]
            
            # Средний результат последних сделок
            avg_pnl = np.mean([t['pnl_percent'] for t in recent_trades]) / 100
            obs.append(avg_pnl)
            
            # Win rate последних сделок
            wins = sum(1 for t in recent_trades if t['pnl_after_commission'] > 0)
            obs.append(wins / len(recent_trades))
            
            # Качество риск-менеджмента
            obs.append(self.trade_tracker.get_risk_management_quality())
        else:
            obs.extend([0, 0, 0])
        
        # Технические индикаторы
        if len(self.price_history) >= 20:
            # SMA
            sma_5 = np.mean(self.price_history[-5:])
            sma_20 = np.mean(self.price_history[-20:])
            obs.extend([
                self.current_price / sma_5 - 1,
                self.current_price / sma_20 - 1,
                sma_5 / sma_20 - 1,
            ])
            
            # Волатильность
            returns = []
            for i in range(1, 20):
                ret = (self.price_history[-i] - self.price_history[-i-1]) / self.price_history[-i-1]
                returns.append(ret)
            volatility = np.std(returns)
            obs.append(volatility * 100)  # Усиливаем сигнал
        else:
            obs.extend([0, 0, 0, 0])
        
        # История действий (помогает понять свой стиль)
        if len(self.action_history) >= 5:
            recent_actions = self.action_history[-5:]
            
            # Средние значения последних действий
            avg_position = np.mean([a[0] for a in recent_actions])
            avg_sl = np.mean([a[1] for a in recent_actions])
            avg_tp = np.mean([a[2] for a in recent_actions])
            
            obs.extend([avg_position, avg_sl, avg_tp])
        else:
            obs.extend([0, 0, 0])
        
        # Рыночная статистика
        obs.extend([
            self.trade_tracker.consecutive_wins / 10,
            self.trade_tracker.consecutive_losses / 10,
            self.max_drawdown,
        ])
        
        # Дополняем до 30 элементов
        while len(obs) < 30:
            obs.append(0.0)
        
        return np.array(obs[:30], dtype=np.float32)

    def _update_statistics(self, current_value: float):
        """Обновление статистики"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown


class TradingAgent:
    """SAC агент с улучшенным обучением"""
    
    def __init__(self, env: TradingEnvironment, model_path: Optional[str] = None):
        self.env = env
        
        # Настройки нейронной сети
        policy_kwargs = dict(
            activation_fn=nn.ReLU,  # ReLU часто лучше для финансовых данных
            net_arch=dict(
                pi=[512, 256, 128],  # Policy network
                qf=[512, 256, 128]   # Q-function network
            )
        )
        
        if model_path and os.path.exists(model_path):
            self.model = SAC.load(model_path, env=env)
            print(f"📂 Модель загружена из {model_path}")
            self._load_buffer(model_path)
        else:
            # Создаем новую модель с оптимизированными параметрами
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=0.0003,  # Немного выше для быстрого обучения
                buffer_size=100000,
                learning_starts=100,   # Начинаем учиться раньше
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                verbose=0
            )
            print("🆕 Создан новый SAC агент")
        
        self.step_count = 0
        self.learning_stats = {
            'total_updates': 0,
            'recent_rewards': [],
            'best_roi': -float('inf'),
            'recent_trade_results': []
        }

    def _load_buffer(self, model_path: str):
        """Загрузка replay buffer"""
        import pickle
        buffer_path = f"{model_path}_buffer.pkl"
        
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    data = pickle.load(f)
                
                if 'replay_buffer' in data:
                    self.model.replay_buffer = data['replay_buffer']
                    print(f"📦 Replay buffer загружен: {self.model.replay_buffer.size()} записей")
                
                if 'step_count' in data:
                    self.step_count = data['step_count']
                
                if 'learning_stats' in data:
                    self.learning_stats = data['learning_stats']
                    
            except Exception as e:
                print(f"⚠️ Ошибка загрузки buffer: {e}")

    def predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Предсказание действия"""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def learn_online(self, obs: np.ndarray, action: np.ndarray,
                    reward: float, next_obs: np.ndarray, done: bool):
        """Онлайн обучение"""
        self.step_count += 1
        
        # Добавляем в replay buffer
        self.model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
        
        # Обучаемся если достаточно опыта
        if self.model.replay_buffer.size() >= self.model.learning_starts:
            try:
                if not hasattr(self.model, '_logger'):
                    from stable_baselines3.common.logger import configure
                    self.model._logger = configure(folder=None, format_strings=[])
                
                self.model.train(gradient_steps=1)
                self.learning_stats['total_updates'] += 1
                
                # Логирование качества обучения
                if reward > 50:
                    print(f"🎯 Обучение #{self.learning_stats['total_updates']} | ОТЛИЧНОЕ действие: +{reward:.1f}")
                elif reward > 0:
                    print(f"✅ Обучение #{self.learning_stats['total_updates']} | Хорошее действие: +{reward:.1f}")
                elif reward < -50:
                    print(f"❌ Обучение #{self.learning_stats['total_updates']} | Плохое действие: {reward:.1f}")
                
                # Детализация каждые 20 обучений
                if self.learning_stats['total_updates'] % 20 == 0:
                    self._log_learning_progress()
                    
            except Exception as e:
                print(f"❌ Ошибка обучения: {e}")
        
        # Сохраняем статистику
        self.learning_stats['recent_rewards'].append(reward)
        if len(self.learning_stats['recent_rewards']) > 100:
            self.learning_stats['recent_rewards'] = self.learning_stats['recent_rewards'][-100:]

    def save(self, path: str = "./models/trading_bot"):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        self.model.save(path)
        
        # Сохраняем replay buffer
        import pickle
        buffer_path = f"{path}_buffer.pkl"
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump({
                    'replay_buffer': self.model.replay_buffer,
                    'step_count': self.step_count,
                    'learning_stats': self.learning_stats
                }, f)
            print(f"💾 Модель и buffer сохранены: {path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения buffer: {e}")

    def _log_learning_progress(self):
        """Детальный лог прогресса обучения"""
        if not self.learning_stats['recent_rewards']:
            return
        
        recent_rewards = self.learning_stats['recent_rewards'][-20:]
        avg_reward = np.mean(recent_rewards)
        positive_rewards = sum(1 for r in recent_rewards if r > 0)
        
        print(f"""
📊 ПРОГРЕСС ОБУЧЕНИЯ (#{self.learning_stats['total_updates']}):
├─ 📈 Средняя награда: {avg_reward:+.1f}
├─ ✅ Положительных: {positive_rewards}/20 ({positive_rewards*5}%)
├─ 📦 Buffer size: {self.model.replay_buffer.size()}
└─ 🎯 Лучший ROI: {self.learning_stats.get('best_roi', 0):.2%}
        """)


class AutonomousTradingBot:
    """Главный класс автономного бота"""
    
    def __init__(self):
        print("""
🚀 АВТОНОМНЫЙ ТОРГОВЫЙ БОТ v3.1
══════════════════════════════════
💰 Баланс: $10,000
🎯 Система наград: За результаты, не намерения
🧠 ИИ: SAC с полной свободой
══════════════════════════════════
        """)
        
        self.env = TradingEnvironment(Config.INITIAL_BALANCE, detailed_logging=True)
        
        model_path = "./models/trading_bot.zip"
        if os.path.exists(model_path):
            self.agent = TradingAgent(self.env, model_path)
        else:
            self.agent = TradingAgent(self.env)
        
        self.running = False
        self.start_time = None
        self.total_steps = 0
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start_trading(self):
        """Запуск автономной торговли"""
        print("\n🚀 ЗАПУСК АВТОНОМНОЙ ТОРГОВЛИ!")
        print("📌 Нажмите Ctrl+C для остановки\n")
        
        self.running = True
        self.start_time = time.time()
        
        obs, _ = self.env.reset()
        
        try:
            while self.running:
                # Получаем решение от ИИ
                action = self.agent.predict(obs, deterministic=False)
                
                # Исполняем действие
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # ОБУЧАЕМСЯ на каждом шаге
                self.agent.learn_online(obs, action, reward, next_obs, terminated or truncated)
                
                # Логирование
                self._log_step(info, action, reward)
                
                # Переход к следующему состоянию
                obs = next_obs
                self.total_steps += 1
                
                # Сброс при необходимости
                if terminated or truncated:
                    print("🔄 Сброс окружения...")
                    obs, _ = self.env.reset()
                
                # Сохранение каждые 100 шагов
                if self.total_steps % 100 == 0:
                    self.agent.save()
                
                # Пауза
              # 1 секунда между решениями
                
        except KeyboardInterrupt:
            print("\n⏸️ Остановка...")
        finally:
            self._shutdown()

    def _log_step(self, info: Dict, action: np.ndarray, reward: float):
        """Улучшенное логирование"""
        if self.total_steps % 10 == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            portfolio = info['portfolio_value']
            roi = info['roi'] * 100
            
            # Форматируем действие
            action_str = f"Pos:{action[0]:+.2f}"
            if action[1] > 0.005:
                action_str += f" SL:{action[1]:.1%}"
            if action[2] > 0.005:
                action_str += f" TP:{action[2]:.1%}"
            
            # Детализация награды
            reward_details = info.get('reward_breakdown', {})
            reward_str = f"{reward:+.1f}"
            if reward_details:
                components = []
                if reward_details.get('trade_result', 0) != 0:
                    components.append(f"Trade:{reward_details['trade_result']:+.0f}")
                if reward_details.get('contract_penalty', 0) != 0:
                    components.append(f"Contract:{reward_details['contract_penalty']:+.0f}")
                if reward_details.get('discipline_bonus', 0) != 0:
                    components.append(f"Discipline:{reward_details['discipline_bonus']:+.0f}")
                if reward_details.get('efficiency', 0) != 0:
                    components.append(f"Efficiency:{reward_details['efficiency']:+.0f}")

                if components:
                    reward_str += f" ({' | '.join(components)})"
            
            # Обновляем лучший ROI
            current_roi = info['roi']
            if current_roi > self.agent.learning_stats['best_roi']:
                self.agent.learning_stats['best_roi'] = current_roi

            print(f"""
[{current_time}] 📊 Шаг {self.total_steps}
├─ 💰 Портфель: ${portfolio:,.2f} (ROI: {roi:+.2f}%)
├─ 🎯 Действие: {action_str}
├─ 🏆 Награда: {reward_str}
├─ 📈 Сделок: {info['total_trades']} (WR: {info['win_rate']:.1%})
├─ 📍 Активных позиций: {info['active_trades']}
├─ 🎖️ Качество RM: {info['risk_management_quality']:.1%}
└─ 🧠 Обучений: {self.agent.learning_stats['total_updates']}
            """)

    def _signal_handler(self, signum, frame):
        """Обработка сигналов"""
        print(f"\n⚠️ Получен сигнал завершения")
        print("💾 Сохраняем прогресс...")
        
        try:
            self.agent.save()
            print("✅ Модель сохранена успешно")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
        
        self.running = False

    def _shutdown(self):
        """Завершение работы"""
        print("\n🛑 ЗАВЕРШЕНИЕ РАБОТЫ")
        
        self.agent.save()
        
        runtime = time.time() - self.start_time if self.start_time else 0
        final_value = self.env._get_portfolio_value()
        total_roi = (final_value - self.env.initial_balance) / self.env.initial_balance * 100
        
        # Статистика сделок
        closed_trades = self.env.trade_tracker.closed_trades
        if closed_trades:
            profitable = sum(1 for t in closed_trades if t['pnl_after_commission'] > 0)
            avg_profit = np.mean([t['pnl_percent'] for t in closed_trades if t['pnl_after_commission'] > 0]) if profitable > 0 else 0
            avg_loss = np.mean([t['pnl_percent'] for t in closed_trades if t['pnl_after_commission'] <= 0]) if profitable < len(closed_trades) else 0
            
            rm_trades = sum(1 for t in closed_trades if t['had_risk_management'])
            tp_exits = sum(1 for t in closed_trades if t['close_reason'] == 'take_profit')
            sl_exits = sum(1 for t in closed_trades if t['close_reason'] == 'stop_loss')
        else:
            profitable = avg_profit = avg_loss = rm_trades = tp_exits = sl_exits = 0
        
        print(f"""
📈 ФИНАЛЬНАЯ СТАТИСТИКА:
═════════════════════════════════════
⏱️  Время: {runtime/3600:.2f} часов
🔢 Шагов: {self.total_steps}
💰 Финальный баланс: ${final_value:,.2f}
📊 ROI: {total_roi:+.2f}%
📈 Всего сделок: {len(closed_trades)}
🏆 Win Rate: {profitable/max(1, len(closed_trades)):.1%}
💚 Средняя прибыль: {avg_profit:+.1f}%
💔 Средний убыток: {avg_loss:.1f}%
🛡️ Сделок с RM: {rm_trades} ({rm_trades/max(1, len(closed_trades)):.1%})
🎯 Закрыто по TP: {tp_exits}
🛑 Закрыто по SL: {sl_exits}
═════════════════════════════════════
        """)
        
        print("👋 До встречи!")


def main():
    """Главная функция"""
    bot = AutonomousTradingBot()
    bot.start_trading()


if __name__ == "__main__":
    main()
