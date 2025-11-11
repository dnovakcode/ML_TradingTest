#!/usr/bin/env python3

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_position_size_pct: float = 10.0
    max_total_exposure_pct: float = 30.0

    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 4.0
    trailing_stop_activation_pct: float = 3.0
    trailing_stop_distance_pct: float = 1.0

    max_risk_per_trade_pct: float = 1.0

    max_daily_loss_pct: float = 5.0
    max_daily_trades: int = 20

    max_drawdown_pct: float = 15.0

    commission_pct: float = 0.1


class RiskManager:

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0

    def calculate_position_size(self, balance: float, current_price: float,
                               signal_strength: float = 1.0,
                               volatility: Optional[float] = None) -> float:
        base_size = balance * (self.config.max_position_size_pct / 100)

        adjusted_size = base_size * signal_strength

        if volatility and volatility > 0:
            vol_adjustment = 1.0 / (1.0 + volatility)
            adjusted_size *= vol_adjustment

        max_risk_amount = balance * (self.config.max_risk_per_trade_pct / 100)
        stop_loss_distance = self.config.default_stop_loss_pct / 100

        max_size_by_risk = max_risk_amount / stop_loss_distance

        position_size = min(adjusted_size, max_size_by_risk)

        min_trade = 100.0
        if position_size < min_trade:
            return 0.0

        return position_size

    def calculate_stop_loss_take_profit(self, entry_price: float,
                                        is_long: bool,
                                        atr: Optional[float] = None
                                        ) -> Tuple[float, float]:
        if atr:
            sl_distance = atr * 1.5
            tp_distance = atr * 3.0
        else:
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
        if is_long and current_price <= stop_loss:
            return True, "stop_loss"
        if not is_long and current_price >= stop_loss:
            return True, "stop_loss"

        if is_long and current_price >= take_profit:
            return True, "take_profit"
        if not is_long and current_price <= take_profit:
            return True, "take_profit"

        activation_price = entry_price * (1 + self.config.trailing_stop_activation_pct / 100)
        if is_long and current_price >= activation_price:
            trailing_stop = current_price * (1 - self.config.trailing_stop_distance_pct / 100)
            if current_price <= trailing_stop:
                return True, "trailing_stop"

        return False, ""

    def can_open_trade(self, balance: float, current_exposure: float) -> Tuple[bool, str]:
        if balance > self.peak_balance:
            self.peak_balance = balance

        if self.peak_balance > 0:
            self.current_drawdown = ((self.peak_balance - balance) / self.peak_balance) * 100
        else:
            self.current_drawdown = 0.0

        if self.current_drawdown >= self.config.max_drawdown_pct:
            return False, f"max_drawdown_exceeded ({self.current_drawdown:.1f}%)"

        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl / balance) * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                return False, f"daily_loss_limit_exceeded ({daily_loss_pct:.1f}%)"

        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"daily_trade_limit_exceeded ({self.daily_trades} trades)"

        exposure_pct = (current_exposure / balance) * 100 if balance > 0 else 0
        if exposure_pct >= self.config.max_total_exposure_pct:
            return False, f"max_exposure_exceeded ({exposure_pct:.1f}%)"

        return True, ""

    def update_daily_stats(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1

    def reset_daily_stats(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def calculate_commission(self, trade_amount: float) -> float:
        return trade_amount * (self.config.commission_pct / 100)

    def get_risk_status(self, balance: float) -> dict:
        return {
            'balance': balance,
            'peak_balance': self.peak_balance,
            'current_drawdown_pct': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'risk_level': self._calculate_risk_level()
        }

    def _calculate_risk_level(self) -> str:
        if self.current_drawdown >= self.config.max_drawdown_pct * 0.8:
            return "CRITICAL"
        elif self.current_drawdown >= self.config.max_drawdown_pct * 0.5:
            return "HIGH"
        elif self.current_drawdown >= self.config.max_drawdown_pct * 0.3:
            return "MEDIUM"
        else:
            return "LOW"


class DynamicRiskManager(RiskManager):

    def __init__(self, config: Optional[RiskConfig] = None):
        super().__init__(config)
        self.win_streak = 0
        self.loss_streak = 0
        self.recent_trades = []

    def adjust_position_size_by_performance(self, base_size: float) -> float:
        if len(self.recent_trades) < 5:
            return base_size

        recent_wins = sum(1 for t in self.recent_trades[-10:] if t > 0)
        win_rate = recent_wins / min(10, len(self.recent_trades))

        if win_rate > 0.6:
            adjustment = 1.2
        elif win_rate < 0.4:
            adjustment = 0.7
        else:
            adjustment = 1.0

        if self.loss_streak >= 3:
            adjustment *= 0.5
        elif self.loss_streak >= 5:
            adjustment *= 0.3

        return base_size * adjustment

    def record_trade_result(self, pnl: float):
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
    config = RiskConfig(
        max_position_size_pct=10.0,
        default_stop_loss_pct=2.0,
        default_take_profit_pct=4.0,
        max_risk_per_trade_pct=1.0
    )

    rm = DynamicRiskManager(config)

    balance = 10000
    current_price = 67000

    position_size = rm.calculate_position_size(
        balance=balance,
        current_price=current_price,
        signal_strength=0.8,
        volatility=0.02
    )

    print(f"💰 Баланс: ${balance}")
    print(f"📊 Рекомендуемый размер позиции: ${position_size:.2f}")

    sl, tp = rm.calculate_stop_loss_take_profit(
        entry_price=current_price,
        is_long=True,
        atr=1000
    )

    print(f"🛑 Stop-Loss: ${sl:.2f}")
    print(f"🎯 Take-Profit: ${tp:.2f}")

    can_trade, reason = rm.can_open_trade(balance, current_exposure=0)
    print(f"\n✅ Можно открыть сделку: {can_trade}")
    if not can_trade:
        print(f"   Причина: {reason}")


if __name__ == "__main__":
    main()
