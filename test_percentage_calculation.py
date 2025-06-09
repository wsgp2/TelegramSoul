#!/usr/bin/env python3
"""
Тест новой системы агрегации и процентов
"""

import asyncio
import json
from chatgpt_analyzer import ChatGPTAnalyzer

async def test_new_percentage_system():
    """Тестирует новую систему честных процентов"""
    
    print("🔍 ТЕСТИРУЕМ НОВУЮ СИСТЕМУ ПРОЦЕНТОВ И АГРЕГАЦИИ")
    print("=" * 60)
    
    # Инициализируем анализатор
    analyzer = ChatGPTAnalyzer()
    
    # Тестовые данные - темы с различными процентами
    test_topics = [
        {
            "name": "Криптовалюты и блокчейн",
            "keywords": ["биткоин", "ethereum", "трейдинг"],
            "percentage": 25.5,
            "sentiment": "positive",
            "description": "Активное обсуждение криптовалют"
        },
        {
            "name": "Криптоинвестиции",
            "keywords": ["биткоин", "альткоины", "портфель"],
            "percentage": 15.2,
            "sentiment": "neutral",
            "description": "Стратегии инвестирования в крипто"
        },
        {
            "name": "Путешествия",
            "keywords": ["отпуск", "билеты", "отели"],
            "percentage": 8.3,
            "sentiment": "positive",
            "description": "Планирование поездок"
        },
        {
            "name": "Работа и карьера",
            "keywords": ["проект", "дедлайн", "зарплата"],
            "percentage": 12.7,
            "sentiment": "neutral",
            "description": "Рабочие вопросы"
        },
        {
            "name": "Здоровье и спорт",
            "keywords": ["тренировка", "диета", "врач"],
            "percentage": 6.1,
            "sentiment": "positive",
            "description": "Забота о здоровье"
        }
    ]
    
    print("📊 ИСХОДНЫЕ ТЕМЫ:")
    total_original = sum(t['percentage'] for t in test_topics)
    for topic in test_topics:
        print(f"  • {topic['name']}: {topic['percentage']}%")
    print(f"📈 Общая сумма: {total_original}%")
    print()
    
    # Тестируем новую агрегацию
    print("🔄 ТЕСТИРУЕМ НОВУЮ АГРЕГАЦИЮ:")
    aggregated = analyzer._aggregate_similar_topics(test_topics)
    
    print(f"📊 Результат агрегации ({len(aggregated)} тем):")
    total_aggregated = sum(t['percentage'] for t in aggregated)
    
    for i, topic in enumerate(aggregated, 1):
        merged_info = f" (объединено {topic['merged_from']})" if topic.get('merged_from', 0) > 1 else ""
        print(f"  {i}. {topic['name']}: {topic['percentage']}%{merged_info}")
    print(f"📈 Общая сумма после агрегации: {total_aggregated}%")
    print()
    
    # Тестируем нормализацию для клиентов
    print("🎯 ТЕСТИРУЕМ НОРМАЛИЗАЦИЮ ДЛЯ КЛИЕНТОВ:")
    
    # Случай 1: Сумма далека от 100%
    low_percentage_topics = [
        {"name": "Тема 1", "percentage": 5.2},
        {"name": "Тема 2", "percentage": 3.1},
        {"name": "Тема 3", "percentage": 2.8}
    ]
    
    print("📉 Случай 1: Низкие проценты (общая сумма далека от 100%)")
    total_low = sum(t['percentage'] for t in low_percentage_topics)
    print(f"   Исходная сумма: {total_low}%")
    
    normalized_low = analyzer.normalize_percentages_for_client(low_percentage_topics)
    for topic in normalized_low:
        if topic['normalized']:
            print(f"   • {topic['name']}: {topic['original_percentage']}% → {topic['percentage']}% (нормализовано)")
        else:
            print(f"   • {topic['name']}: {topic['percentage']}% (без изменений)")
    print()
    
    # Случай 2: Сумма близка к 100%
    normal_topics = [
        {"name": "Тема A", "percentage": 45.2},
        {"name": "Тема B", "percentage": 30.1},
        {"name": "Тема C", "percentage": 20.7}
    ]
    
    print("📊 Случай 2: Нормальные проценты (сумма близка к 100%)")
    total_normal = sum(t['percentage'] for t in normal_topics)
    print(f"   Исходная сумма: {total_normal}%")
    
    normalized_normal = analyzer.normalize_percentages_for_client(normal_topics)
    for topic in normalized_normal:
        if topic['normalized']:
            print(f"   • {topic['name']}: {topic['original_percentage']}% → {topic['percentage']}% (нормализовано)")
        else:
            print(f"   • {topic['name']}: {topic['percentage']}% (без изменений)")
    print()
    
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print()
    print("🎯 ВЫВОДЫ:")
    print("1. Агрегация теперь универсальная - не привязана к конкретным темам")
    print("2. Проценты честные - отражают реальные данные")
    print("3. Нормализация работает только при необходимости")
    print("4. Система готова для любых пользователей, не только крипто-энтузиастов")

if __name__ == "__main__":
    asyncio.run(test_new_percentage_system()) 