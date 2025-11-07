#!/usr/bin/env python3
"""
test_bot.py - Быстрый тест бота без реальной торговли
"""

import numpy as np
from main import TradingEnvironment, TradingAgent


def test_environment():
    """Тест окружения"""
    print("🧪 Тестирование окружения...")
    
    env = TradingEnvironment(10000)
    obs, _ = env.reset()
    
    print(f"✅ Observation shape: {obs.shape}")
    print(f"✅ Action space: {env.action_space}")
    
    # Тест случайных действий
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        
        print(f"Шаг {i+1}: reward={reward:.1f}, portfolio=${info['portfolio_value']:.2f}")
        
    print(f"🎯 Общая награда: {total_reward:.1f}")
    return True


def test_agent():
    """Тест агента"""
    print("\n🤖 Тестирование агента...")
    
    env = TradingEnvironment(10000)
    agent = TradingAgent(env)
    
    obs, _ = env.reset()
    
    for i in range(5):
        action = agent.predict(obs)
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"Шаг {i+1}: action={action}, reward={reward:.1f}")
        
    print("✅ Агент работает!")
    return True


if __name__ == "__main__":
    print("🚀 ТЕСТ ТОРГОВОГО БОТА")
    print("="*40)
    
    # Тест 1: Окружение
    test_environment()
    
    # Тест 2: Агент
    test_agent()
    
    print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("🎯 Запускайте: python main.py")
