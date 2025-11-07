#!/usr/bin/env python3
"""
Профессиональная система управления рисками
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Конфигурация риск-менеджмента"""
    # Position sizing
    max_position_size_pct: float = 10.0  # Максимальный размер позиции (% от баланса)
    max_total_exposure_pct: float = 30.0  # Максимальная общая экспозиция

    # Stop-loss и take-profit
    default_stop_loss_pct: float = 2.0   # Дефолтный stop-loss (%)
    default_take_profit_pct: float = 4.0  # Дефолтный take-profit (%)
    trailing_stop_activation_pct: float = 3.0  # Активация trailing stop
    trailing_stop_distance_pct: float = 1.0    # Расстояние trailing stop

    # Risk per trade
    max_risk_per_trade_pct: float = 1.0  # Максимальный риск на сделку (% от баланса)

    # Daily limits
    max_daily_loss_pct: float = 5.0      # Максимальная дневная просадка (%)
    max_daily_trades: int = 20           # Максимум сделок в день

    # Drawdown protection
    max_drawdown_pct: float = 15.0       # Критическая просадка для остановки

    # Commission
    commission_pct: float = 0.1          # Комиссия (%)


class RiskManager:
    """Менеджер рисков для торгового бота"""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

        # Tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0

    def calculate_position_size(self, balance: float, current_price: float,
                               signal_strength: float = 1.0,
                               volatility: Optional[float] = None) -> float:
        """
        Расчет размера позиции с учетом риска

        Args:
            balance: Текущий баланс
            current_price: Текущая цена актива
            signal_strength: Сила сигнала (0-1)
            volatility: Текущая волатильность (опционально)

        Returns:
            Размер позиции в USD
        """
        # Базовый размер позиции
        base_size = balance * (self.config.max_position_size_pct / 100)

        # Корректировка на силу сигнала
        adjusted_size = base_size * signal_strength

        # Корректировка на волатильность (Kelly Criterion подход)
        if volatility and volatility > 0:
            # Уменьшаем размер при высокой волатильности
            vol_adjustment = 1.0 / (1.0 + volatility)
            adjusted_size *= vol_adjustment

        # Учет максимального риска на сделку
        max_risk_amount = balance * (self.config.max_risk_per_trade_pct / 100)
        stop_loss_distance = self.config.default_stop_loss_pct / 100

        # Размер позиции не должен создавать риск больше max_risk_per_trade
        max_size_by_risk = max_risk_amount / stop_loss_distance

        # Выбираем меньшее значение
        position_size = min(adjusted_size, max_size_by_risk)

        # Минимальный размер сделки
        min_trade = 100.0
        if position_size < min_trade:
            return 0.0

        return position_size

    def calculate_stop_loss_take_profit(self, entry_price: float,
                                        is_long: bool,
                                        atr: Optional[float] = None
                                        ) -> Tuple[float, float]:
        """
        Расчет уровней stop-loss и take-profit

        Args:
            entry_price: Цена входа
            is_long: Long позиция?
            atr: Average True Range (опционально)

        Returns:
            (stop_loss_price, take_profit_price)
        """
        if atr:
            # Используем ATR для динамических уровней
            sl_distance = atr * 1.5  # 1.5 ATR для stop-loss
            tp_distance = atr * 3.0  # 3 ATR для take-profit
        else:
            # Используем процентные уровни
            sl_distance = entry_price * (self.config.default_stop_loss_pct / 100)
            tp_distance = entry_price * (self.config.default_take_profit_pct / 100)

        if is_long:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit

    def should_close_position(self, entry_price: float, current_price: float,
                             stop_loss: float, take_profit: float,
                             is_long: bool, steps_held: int) -> Tuple[bool, str]:
        """
        Проверка условий закрытия позиции

        Returns:
            (should_close, reason)
        """
        # Stop-loss
        if is_long and current_price <= stop_loss:
            return True, "stop_loss"
        if not is_long and current_price >= stop_loss:
            return True, "stop_loss"

        # Take-profit
        if is_long and current_price >= take_profit:
            return True, "take_profit"
        if not is_long and current_price <= take_profit:
            return True, "take_profit"

        # Trailing stop (если активирован)
        activation_price = entry_price * (1 + self.config.trailing_stop_activation_pct / 100)
        if is_long and current_price >= activation_price:
            trailing_stop = current_price * (1 - self.config.trailing_stop_distance_pct / 100)
            if current_price <= trailing_stop:
                return True, "trailing_stop"

        return False, ""

    def can_open_trade(self, balance: float, current_exposure: float) -> Tuple[bool, str]:
        """
        Проверка возможности открытия новой сделки

        Returns:
            (can_trade, reason_if_not)
        """
        # Обновляем пик баланса
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Вычисляем текущую просадку
        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - balance) / self.peak_balance) * 100
        else:
            self.current_drawdown = 0.0

        # Проверка критической просадки
        if self.current_drawdown >= self.config.max_drawdown_pct:
            return False, f"max_drawdown_exceeded ({self.current_drawdown:.1f}%)"

        # Проверка дневного лимита убытков
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl / balance) * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                return False, f"daily_loss_limit_exceeded ({daily_loss_pct:.1f}%)"

        # Проверка дневного лимита сделок
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"daily_trade_limit_exceeded ({self.daily_trades} trades)"

        # Проверка общей экспозиции
        exposure_pct = (current_exposure / balance) * 100 if balance > 0 else 0
        if exposure_pct >= self.config.max_total_exposure_pct:
            return False, f"max_exposure_exceeded ({exposure_pct:.1f}%)"

        return True, ""

    def update_daily_stats(self, pnl: float):
        """Обновить дневную статистику"""
        self.daily_pnl += pnl
        self.daily_trades += 1

    def reset_daily_stats(self):
        """Сброс дневной статистики (вызывать в начале нового дня)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def calculate_commission(self, trade_amount: float) -> float:
        """Расчет комиссии"""
        return trade_amount * (self.config.commission_pct / 100)

    def get_risk_status(self, balance: float) -> dict:
        """Получить текущий статус рисков"""
        return {
            'balance': balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'risk_level': self._calculate_risk_level()
        }

    def _calculate_risk_level(self) -> str:
        """Вычислить текущий уровень риска"""
        if self.current_drawdown >= self.config.max_drawdown_pct * 0.8:
            return "CRITICAL"
        elif self.current_drawdown >= self.config.max_drawdown_pct * 0.5:
            return "HIGH"
        elif self.current_drawdown >= self.config.max_drawdown_pct * 0.3:
            return "MEDIUM"
        else:
            return "LOW"


class DynamicRiskManager(RiskManager):
    """
    Динамический риск-менеджер, который адаптируется к условиям рынка
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        super().__init__(config)
        self.win_streak = 0
        self.loss_streak = 0
        self.recent_trades = []

    def adjust_position_size_by_performance(self, base_size: float) -> float:
        """
        Адаптация размера позиции на основе текущей производительности
        """
        if len(self.recent_trades) < 5:
            return base_size

        # Вычисляем win rate за последние N сделок
        recent_wins = sum(1 for t in self.recent_trades[-10:] if t > 0)
        win_rate = recent_wins / min(10, len(self.recent_trades))

        # Адаптация размера
        if win_rate > 0.6:
            # Хорошая производительность - увеличиваем размер
            adjustment = 1.2
        elif win_rate < 0.4:
            # Плохая производительность - уменьшаем размер
            adjustment = 0.7
        else:
            adjustment = 1.0

        # При серии убытков агрессивно уменьшаем размер
        if self.loss_streak >= 3:
            adjustment *= 0.5
        elif self.loss_streak >= 5:
            adjustment *= 0.3

        return base_size * adjustment

    def record_trade_result(self, pnl: float):
        """Записать результат сделки"""
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)

        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        self.update_daily_stats(pnl)


def main():
    """Пример использования"""
    config = RiskConfig(
        max_position_size_pct=10.0,
        default_stop_loss_pct=2.0,
        default_take_profit_pct=4.0,
        max_risk_per_trade_pct=1.0
    )

    rm = DynamicRiskManager(config)

    balance = 10000
    current_price = 67000

    # Расчет размера позиции
    position_size = rm.calculate_position_size(
        balance=balance,
        current_price=current_price,
        signal_strength=0.8,
        volatility=0.02
    )

    print(f"💰 Баланс: ${balance}")
    print(f"📊 Рекомендуемый размер позиции: ${position_size:.2f}")

    # Расчет stop-loss и take-profit
    sl, tp = rm.calculate_stop_loss_take_profit(
        entry_price=current_price,
        is_long=True,
        atr=1000
    )

    print(f"🛑 Stop-Loss: ${sl:.2f}")
    print(f"🎯 Take-Profit: ${tp:.2f}")

    # Проверка возможности открытия сделки
    can_trade, reason = rm.can_open_trade(balance, current_exposure=0)
    print(f"\n✅ Можно открыть сделку: {can_trade}")
    if not can_trade:
        print(f"   Причина: {reason}")


if __name__ == "__main__":
    main()
