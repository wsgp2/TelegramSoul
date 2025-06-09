#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки исправлений в красивом формате
Использует проблемные данные похожие на реальные
"""

from chatgpt_analyzer import ChatGPTAnalyzer
import os

def test_fixed_format():
    """Тестирует исправленный красивый формат с проблемными данными"""
    
    # Создаем тестовые данные с проблемными процентами (как в реальных данных)
    test_topics_data = {
        "topics": [
            {
                "name": "Путешествия и жилье",
                "keywords": ["путешествия", "жилье", "переезд"],
                "percentage": 489.0,  # Проблемный процент > 100%
                "sentiment": "positive",
                "description": "Участники обсуждают свои планы по переезду и выбору жилья"
            },
            {
                "name": "Финансовые вопросы и кредиты", 
                "keywords": ["финансы", "кредиты", "ипотека"],
                "percentage": 362.0,  # Проблемный процент > 100%
                "sentiment": "negative",
                "description": "Участники обсуждают финансовые трудности, связанные с кредитами"
            },
            {
                "name": "Монетизация и инвестиции",
                "keywords": ["инвестиции", "заработок", "пирамиды"],
                "percentage": 355.5,  # Проблемный процент > 100%
                "sentiment": "neutral",
                "description": "Участники обсуждают возможности заработка через различные схемы"
            },
            {
                "name": "Личностное развитие",
                "keywords": ["развитие", "дисциплина", "привычки"],
                "percentage": 336.0,  # Проблемный процент > 100%
                "sentiment": "positive",
                "description": "Участники обсуждают важность личностного роста и дисциплины"
            },
            {
                "name": "Криптовалюты и инвестиции",
                "keywords": ["криптовалюты", "биткоин", "торговля"],
                "percentage": 311.3,  # Проблемный процент > 100%
                "sentiment": "neutral",
                "description": "Участники обсуждают возможности инвестирования в криптовалюты"
            }
        ]
    }
    
    # Тестовые данные коммерческой оценки с правильными названиями тем
    test_commercial_assessment = {
        "commercial_assessment": [
            {
                "topic": "Путешествия и жилье",
                "commercial_score": "Средний коммерческий потенциал",
                "products": [
                    {
                        "name": "Консультации по переезду",
                        "description": "Помощь в планировании переезда",
                        "revenue_potential": "средний"
                    }
                ]
            },
            {
                "topic": "Финансовые вопросы и кредиты",
                "commercial_score": "Высокий коммерческий потенциал",
                "products": [
                    {
                        "name": "Финансовые консультации",
                        "description": "Помощь в решении кредитных проблем",
                        "revenue_potential": "высокий"
                    }
                ]
            }
        ]
    }
    
    # Инициализируем анализатор
    try:
        analyzer = ChatGPTAnalyzer(api_key="test_key")
    except:
        analyzer = ChatGPTAnalyzer.__new__(ChatGPTAnalyzer)
        analyzer.output_dir = "data/reports"
        os.makedirs(analyzer.output_dir, exist_ok=True)
    
    print("🔧 ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ В КРАСИВОМ ФОРМАТЕ")
    print("=" * 60)
    
    print(f"\n📊 ИСХОДНЫЕ ПРОБЛЕМНЫЕ ДАННЫЕ:")
    total_raw = sum(t['percentage'] for t in test_topics_data['topics'])
    print(f"📈 Сумма процентов: {total_raw:.1f}% (проблема!)")
    for topic in test_topics_data['topics']:
        print(f"   • {topic['name']}: {topic['percentage']:.1f}%")
    
    print("\n" + "="*60)
    
    # Тестируем исправленный красивый формат тем
    print("\n💎 ИСПРАВЛЕННЫЙ КРАСИВЫЙ ФОРМАТ:")
    print("-" * 40)
    beautiful_topics = analyzer.generate_beautiful_topic_format(test_topics_data)
    print(beautiful_topics)
    
    print("\n" + "=" * 60)
    
    # Тестируем полный исправленный отчет
    print("\n📊 ПОЛНЫЙ ИСПРАВЛЕННЫЙ ОТЧЕТ:")
    print("-" * 40)
    beautiful_report = analyzer.generate_beautiful_client_report(
        test_topics_data, 
        test_commercial_assessment, 
        "Тестовый Клиент (Исправленный)"
    )
    print(beautiful_report)
    
    # Сохраняем исправленный отчет
    report_path = os.path.join(analyzer.output_dir, "ТЕСТ_исправленный_отчет.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(beautiful_report)
    
    print(f"\n💾 Исправленный отчет сохранен: {report_path}")
    print("\n✅ Тестирование исправлений завершено!")

if __name__ == "__main__":
    test_fixed_format() 