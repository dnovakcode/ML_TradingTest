#!/usr/bin/env python3
"""
🎯 Walk-Forward Validation для реалистичной оценки торговой стратегии

Walk-Forward подход:
1. Разбиваем данные на периоды (например, по 3 месяца)
2. Обучаем модель на train window
3. Тестируем на следующем периоде (validation window)
4. Сдвигаем окно и повторяем
5. Получаем реалистичную оценку на out-of-sample данных

Это предотвращает overfitting и дает честную оценку!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import SAC
import torch
import torch.nn as nn

from data_fetcher import HistoricalDataFetcher
from trading_bot_pro import ProductionTradingEnvironment
from risk_manager import RiskConfig
from metrics import TradingMetricsCalculator, PerformanceMetrics


class WalkForwardValidator:
    """Walk-Forward валидатор"""

    def __init__(self, df: pd.DataFrame, train_window_days: int = 180,
                 test_window_days: int = 30, step_days: int = 30):
        """
        Args:
            df: Полный датасет
            train_window_days: Размер обучающего окна (дней)
            test_window_days: Размер тестового окна (дней)
            step_days: Шаг сдвига окна (дней)
        """
        self.df = df.sort_values('timestamp') if 'timestamp' in df.columns else df
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days

        # Предполагаем 1h свечи
        self.hours_per_day = 24
        self.train_window_size = train_window_days * self.hours_per_day
        self.test_window_size = test_window_days * self.hours_per_day
        self.step_size = step_days * self.hours_per_day

    def generate_folds(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Генерация фолдов для walk-forward валидации

        Returns:
            List[(train_df, test_df)]
        """
        folds = []
        start_idx = 0

        while start_idx + self.train_window_size + self.test_window_size <= len(self.df):
            train_end_idx = start_idx + self.train_window_size
            test_end_idx = train_end_idx + self.test_window_size

            train_df = self.df.iloc[start_idx:train_end_idx].copy()
            test_df = self.df.iloc[train_end_idx:test_end_idx].copy()

            folds.append((train_df, test_df))

            # Сдвигаем окно
            start_idx += self.step_size

        print(f"📊 Сгенерировано {len(folds)} фолдов для walk-forward валидации")
        return folds

    def run_validation(self, folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
                      training_steps: int = 50000, initial_balance: float = 10000
                      ) -> List[PerformanceMetrics]:
        """
        Запуск walk-forward валидации

        Returns:
            List[PerformanceMetrics] для каждого фолда
        """
        all_metrics = []
        all_test_results = []

        risk_config = RiskConfig(
            max_position_size_pct=15.0,
            max_risk_per_trade_pct=2.0,
            default_stop_loss_pct=3.0,
            default_take_profit_pct=6.0
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for fold_idx, (train_df, test_df) in enumerate(folds):
            print(f"\n{'='*80}")
            print(f"🔄 ФОЛД {fold_idx + 1}/{len(folds)}")
            print(f"{'='*80}")
            print(f"  📊 Train: {len(train_df)} свечей")
            print(f"  📊 Test:  {len(test_df)} свечей")

            # Создаем окружение для обучения
            train_env = ProductionTradingEnvironment(
                train_df, initial_balance=initial_balance, risk_config=risk_config
            )

            # Создаем новую модель для каждого фолда (реалистично!)
            policy_kwargs = dict(
                activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 128], qf=[256, 128]),
            )

            model = SAC(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=2000,
                batch_size=256,
                tau=0.005,
                gamma=0.995,
                ent_coef='auto',
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0
            )

            # Обучение
            print(f"\n  🎓 Обучение на {training_steps} шагов...")
            model.learn(total_timesteps=training_steps, progress_bar=False)

            # Тестирование
            print(f"  🧪 Тестирование на out-of-sample данных...")
            test_env = ProductionTradingEnvironment(
                test_df, initial_balance=initial_balance, risk_config=risk_config
            )

            obs, _ = test_env.reset()
            done = False
            truncated = False

            while not done and not truncated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)

            # Вычисляем метрики
            calculator = TradingMetricsCalculator(initial_balance=initial_balance)
            metrics = calculator.calculate_metrics(
                portfolio_values=test_env.portfolio_history,
                closed_trades=test_env.closed_trades,
                days_traded=len(test_df) / self.hours_per_day
            )

            all_metrics.append(metrics)
            all_test_results.append({
                'fold': fold_idx,
                'portfolio_history': test_env.portfolio_history,
                'closed_trades': test_env.closed_trades,
                'final_balance': test_env.balance,
                'metrics': metrics
            })

            # Краткий отчет
            print(f"\n  📊 Результаты фолда {fold_idx + 1}:")
            print(f"    ROI: {metrics.total_return_pct:+.2f}%")
            print(f"    Sharpe: {metrics.sharpe_ratio:.3f}")
            print(f"    Win Rate: {metrics.win_rate:.1f}%")
            print(f"    Max DD: {metrics.max_drawdown_pct:.2f}%")
            print(f"    Сделок: {metrics.total_trades}")

        return all_metrics, all_test_results

    def print_summary(self, all_metrics: List[PerformanceMetrics]):
        """Вывод сводки по всем фолдам"""

        print(f"\n{'='*80}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА WALK-FORWARD VALIDATION")
        print(f"{'='*80}")

        # Собираем метрики
        returns = [m.total_return_pct for m in all_metrics]
        sharpes = [m.sharpe_ratio for m in all_metrics]
        win_rates = [m.win_rate for m in all_metrics]
        max_dds = [m.max_drawdown_pct for m in all_metrics]
        profit_factors = [m.profit_factor for m in all_metrics]

        print(f"\n💰 ДОХОДНОСТЬ:")
        print(f"  Средний ROI:       {np.mean(returns):+.2f}% (±{np.std(returns):.2f}%)")
        print(f"  Медианный ROI:     {np.median(returns):+.2f}%")
        print(f"  Лучший ROI:        {np.max(returns):+.2f}%")
        print(f"  Худший ROI:        {np.min(returns):+.2f}%")
        print(f"  Прибыльных фолдов: {sum(1 for r in returns if r > 0)}/{len(returns)}")

        print(f"\n📈 SHARPE RATIO:")
        print(f"  Средний:           {np.mean(sharpes):.3f}")
        print(f"  Медианный:         {np.median(sharpes):.3f}")
        print(f"  Лучший:            {np.max(sharpes):.3f}")
        print(f"  Худший:            {np.min(sharpes):.3f}")

        print(f"\n🎯 WIN RATE:")
        print(f"  Средний:           {np.mean(win_rates):.1f}%")
        print(f"  Медианный:         {np.median(win_rates):.1f}%")

        print(f"\n⚠️  MAX DRAWDOWN:")
        print(f"  Средний:           {np.mean(max_dds):.2f}%")
        print(f"  Худший:            {np.min(max_dds):.2f}%")

        print(f"\n💎 PROFIT FACTOR:")
        print(f"  Средний:           {np.mean(profit_factors):.2f}")
        print(f"  Медианный:         {np.median(profit_factors):.2f}")

        # Оценка готовности к продакшену
        print(f"\n{'='*80}")
        print(f"🎯 ОЦЕНКА ГОТОВНОСТИ К ПРОДАКШЕНУ")
        print(f"{'='*80}")

        # Критерии
        criteria = {
            'Средний ROI > 0%': np.mean(returns) > 0,
            'Средний Sharpe > 0.5': np.mean(sharpes) > 0.5,
            'Прибыльность > 60%': sum(1 for r in returns if r > 0) / len(returns) > 0.6,
            'Средний Win Rate > 45%': np.mean(win_rates) > 45,
            'Средний Max DD < 25%': abs(np.mean(max_dds)) < 25,
            'Все фолды Max DD < 40%': all(abs(dd) < 40 for dd in max_dds),
            'Средний Profit Factor > 1.2': np.mean(profit_factors) > 1.2
        }

        passed = 0
        for criterion, result in criteria.items():
            status = '✅' if result else '❌'
            print(f"  {status} {criterion}")
            if result:
                passed += 1

        print(f"\n{'='*80}")
        if passed >= 6:
            print(f"✅ РЕЗУЛЬТАТ: ГОТОВ К ПРОДАКШЕНУ ({passed}/7 критериев)")
            print(f"💡 Бот показывает стабильные результаты на разных периодах!")
        elif passed >= 4:
            print(f"⚠️  РЕЗУЛЬТАТ: ТРЕБУЕТСЯ ДОРАБОТКА ({passed}/7 критериев)")
            print(f"💡 Бот показывает потенциал, но нужна оптимизация параметров")
        else:
            print(f"❌ РЕЗУЛЬТАТ: НЕ ГОТОВ К ПРОДАКШЕНУ ({passed}/7 критериев)")
            print(f"💡 Требуется переработка стратегии или дополнительное обучение")
        print(f"{'='*80}\n")

        return passed >= 6


def main():
    """Главная функция"""

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       🎯 WALK-FORWARD VALIDATION SYSTEM 🎯              ║
    ║                                                           ║
    ║  Честная оценка производительности бота на реальных      ║
    ║  данных без подгонки под будущее!                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Загружаем данные
    print("\n📥 ЗАГРУЗКА ИСТОРИЧЕСКИХ ДАННЫХ...")
    fetcher = HistoricalDataFetcher(symbol='BTC/USDT', timeframe='1h')
    df = fetcher.fetch_data(days=365, force_refresh=False)
    df = fetcher.add_technical_indicators(df)

    print(f"✅ Загружено {len(df)} свечей")
    print(f"   Период: {df.index[0]} → {df.index[-1]}")

    # Создаем валидатор
    validator = WalkForwardValidator(
        df=df,
        train_window_days=120,  # 4 месяца обучения
        test_window_days=30,    # 1 месяц тестирования
        step_days=30           # Сдвиг на 1 месяц
    )

    # Генерируем фолды
    folds = validator.generate_folds()

    # Запускаем валидацию
    print(f"\n🚀 ЗАПУСК WALK-FORWARD VALIDATION...")
    print(f"⚠️  Это займет некоторое время (обучение + тестирование каждого фолда)...\n")

    all_metrics, all_results = validator.run_validation(
        folds=folds,
        training_steps=30000,  # Меньше шагов для каждого фолда (быстрее)
        initial_balance=10000
    )

    # Выводим итоговую статистику
    is_production_ready = validator.print_summary(all_metrics)

    # Сохраняем результаты
    results_dir = "./validation_results"
    os.makedirs(results_dir, exist_ok=True)

    # Сохраняем детальные результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/walk_forward_{timestamp}.txt"

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Walk-Forward Validation Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total Folds: {len(all_metrics)}\n")
        f.write(f"Production Ready: {is_production_ready}\n\n")

        for idx, metrics in enumerate(all_metrics):
            f.write(f"\nFold {idx + 1}:\n")
            f.write(f"  ROI: {metrics.total_return_pct:+.2f}%\n")
            f.write(f"  Sharpe: {metrics.sharpe_ratio:.3f}\n")
            f.write(f"  Win Rate: {metrics.win_rate:.1f}%\n")
            f.write(f"  Max DD: {metrics.max_drawdown_pct:.2f}%\n")

    print(f"💾 Результаты сохранены: {results_file}")
    print(f"\n✅ WALK-FORWARD VALIDATION ЗАВЕРШЕНА!")


if __name__ == "__main__":
    main()
