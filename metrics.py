#!/usr/bin/env python3
"""
Профессиональные метрики для оценки торговой стратегии
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Структура для хранения метрик производительности"""
    # Основные метрики доходности
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Риск
    max_drawdown: float
    max_drawdown_pct: float
    volatility: float

    # Sharpe и Sortino ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Торговые метрики
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Средняя длительность сделок
    avg_trade_duration: float

    # Expectancy
    expectancy: float

    # Recovery factor
    recovery_factor: float


class TradingMetricsCalculator:
    """Калькулятор профессиональных торговых метрик"""

    def __init__(self, initial_balance: float = 10000, risk_free_rate: float = 0.02):
        """
        Args:
            initial_balance: Начальный баланс
            risk_free_rate: Безрисковая ставка (годовая), по умолчанию 2%
        """
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self, portfolio_values: List[float],
                         closed_trades: List[Dict],
                         days_traded: Optional[int] = None) -> PerformanceMetrics:
        """
        Вычислить все метрики производительности

        Args:
            portfolio_values: История значений портфеля
            closed_trades: Список закрытых сделок
            days_traded: Количество дней торговли (для аннуализации)

        Returns:
            PerformanceMetrics
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return self._empty_metrics()

        portfolio_values = np.array(portfolio_values)
        final_value = portfolio_values[-1]

        # Основные метрики доходности
        total_return = final_value - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100

        # Аннуализированная доходность
        if days_traded and days_traded > 0:
            years = days_traded / 365
            annualized_return = (((final_value / self.initial_balance) ** (1 / years)) - 1) * 100
        else:
            annualized_return = 0.0

        # Drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown(portfolio_values)

        # Волатильность
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Аннуализированная

        # Sharpe Ratio
        if volatility > 0:
            excess_returns = returns - (self.risk_free_rate / 252)
            sharpe_ratio = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (использует только downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            if downside_std > 0:
                sortino_ratio = ((np.mean(returns) - self.risk_free_rate / 252) /
                               (downside_std / np.sqrt(252)))
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = sharpe_ratio

        # Calmar Ratio (annualized return / max drawdown)
        if abs(max_dd_pct) > 0:
            calmar_ratio = annualized_return / abs(max_dd_pct)
        else:
            calmar_ratio = 0.0

        # Торговые метрики
        trade_metrics = self._calculate_trade_metrics(closed_trades)

        # Recovery Factor
        if abs(max_dd) > 0:
            recovery_factor = total_return / abs(max_dd)
        else:
            recovery_factor = 0.0

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            volatility=volatility * 100,  # В процентах
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            recovery_factor=recovery_factor,
            **trade_metrics
        )

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> tuple:
        """Вычислить максимальную просадку"""
        cummax = np.maximum.accumulate(portfolio_values)
        drawdowns = portfolio_values - cummax
        max_dd = np.min(drawdowns)
        max_dd_pct = (max_dd / cummax[np.argmin(drawdowns)]) * 100
        return max_dd, max_dd_pct

    def _calculate_trade_metrics(self, closed_trades: List[Dict]) -> Dict:
        """Вычислить метрики по закрытым сделкам"""
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'avg_trade_duration': 0.0,
                'expectancy': 0.0
            }

        total_trades = len(closed_trades)

        # Разделяем на прибыльные и убыточные
        winning_trades = [t for t in closed_trades if t.get('pnl_after_commission', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl_after_commission', 0) <= 0]

        n_wins = len(winning_trades)
        n_losses = len(losing_trades)

        win_rate = (n_wins / total_trades) * 100 if total_trades > 0 else 0.0

        # Средние прибыль и убыток
        avg_win = np.mean([t['pnl_after_commission'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['pnl_after_commission'] for t in losing_trades]) if losing_trades else 0.0

        # Profit Factor
        total_wins = sum(t['pnl_after_commission'] for t in winning_trades)
        total_losses = abs(sum(t['pnl_after_commission'] for t in losing_trades))

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = float('inf') if total_wins > 0 else 0.0

        # Средняя длительность сделки
        durations = [t.get('duration', 0) for t in closed_trades]
        avg_duration = np.mean(durations) if durations else 0.0

        # Expectancy (математическое ожидание)
        if total_trades > 0:
            expectancy = ((win_rate / 100) * avg_win) - ((1 - win_rate / 100) * abs(avg_loss))
        else:
            expectancy = 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': n_wins,
            'losing_trades': n_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_duration,
            'expectancy': expectancy
        }

    def _empty_metrics(self) -> PerformanceMetrics:
        """Пустые метрики при отсутствии данных"""
        return PerformanceMetrics(
            total_return=0.0,
            total_return_pct=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            avg_trade_duration=0.0,
            expectancy=0.0,
            recovery_factor=0.0
        )

    def print_metrics(self, metrics: PerformanceMetrics):
        """Красиво вывести метрики"""
        print(f"""
{'='*80}
📊 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ
{'='*80}

💰 ДОХОДНОСТЬ:
  Общая прибыль:          ${metrics.total_return:,.2f} ({metrics.total_return_pct:+.2f}%)
  Аннуализированная:      {metrics.annualized_return:+.2f}%

⚠️  РИСК:
  Max Drawdown:           ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2f}%)
  Волатильность:          {metrics.volatility:.2f}%

📈 РИСК-СКОРРЕКТИРОВАННЫЕ МЕТРИКИ:
  Sharpe Ratio:           {metrics.sharpe_ratio:.3f}  {'✅' if metrics.sharpe_ratio > 1 else '⚠️' if metrics.sharpe_ratio > 0 else '❌'}
  Sortino Ratio:          {metrics.sortino_ratio:.3f}
  Calmar Ratio:           {metrics.calmar_ratio:.3f}
  Recovery Factor:        {metrics.recovery_factor:.3f}

📊 ТОРГОВЫЕ МЕТРИКИ:
  Всего сделок:           {metrics.total_trades}
  Прибыльных:             {metrics.winning_trades} ({metrics.win_rate:.1f}%)
  Убыточных:              {metrics.losing_trades} ({100-metrics.win_rate:.1f}%)

  Средняя прибыль:        ${metrics.avg_win:.2f}
  Средний убыток:         ${metrics.avg_loss:.2f}
  Profit Factor:          {metrics.profit_factor:.3f}  {'✅' if metrics.profit_factor > 1.5 else '⚠️' if metrics.profit_factor > 1 else '❌'}

  Expectancy:             ${metrics.expectancy:.2f}  {'✅' if metrics.expectancy > 0 else '❌'}
  Ср. длительность:       {metrics.avg_trade_duration/3600:.1f} часов

{'='*80}

💡 ОЦЕНКА:
  Sharpe > 1.0:           {'✅ Хорошо' if metrics.sharpe_ratio > 1 else '❌ Плохо'}
  Profit Factor > 1.5:    {'✅ Хорошо' if metrics.profit_factor > 1.5 else '❌ Плохо'}
  Win Rate > 50%:         {'✅ Хорошо' if metrics.win_rate > 50 else '❌ Плохо'}
  Max DD < 20%:           {'✅ Хорошо' if abs(metrics.max_drawdown_pct) < 20 else '❌ Плохо'}

🎯 ГОТОВНОСТЬ К ПРОДАКШНУ:
  {'✅ ГОТОВ К ТОРГОВЛЕ' if self._is_production_ready(metrics) else '❌ ТРЕБУЕТСЯ ДОРАБОТКА'}

{'='*80}
        """)

    def _is_production_ready(self, metrics: PerformanceMetrics) -> bool:
        """Проверка готовности к продакшн"""
        return (
            metrics.sharpe_ratio > 1.0 and
            metrics.profit_factor > 1.5 and
            metrics.win_rate > 45 and
            abs(metrics.max_drawdown_pct) < 30 and
            metrics.total_trades >= 30 and
            metrics.expectancy > 0
        )


def main():
    """Пример использования"""
    calculator = TradingMetricsCalculator(initial_balance=10000)

    # Пример данных
    portfolio_values = [10000, 10200, 10150, 10300, 10100, 10500, 10800, 10600, 11000]

    closed_trades = [
        {'pnl_after_commission': 200, 'duration': 3600},
        {'pnl_after_commission': -50, 'duration': 1800},
        {'pnl_after_commission': 150, 'duration': 7200},
        {'pnl_after_commission': -200, 'duration': 3600},
        {'pnl_after_commission': 400, 'duration': 10800},
    ]

    metrics = calculator.calculate_metrics(portfolio_values, closed_trades, days_traded=30)
    calculator.print_metrics(metrics)


if __name__ == "__main__":
    main()
