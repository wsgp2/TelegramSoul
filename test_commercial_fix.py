#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для проверки исправленной коммерческой оценки
"""

from chatgpt_analyzer import ChatGPTAnalyzer
import os

def test_commercial_fix():
    """Тестирует исправленную коммерческую оценку"""
    
    # Тестовые данные тем
    test_topics_data = {
        "topics": [
            {
                "name": "Путешествия и жилье",
                "keywords": ["путешествия", "жилье"],
                "percentage": 25.0,
                "sentiment": "positive",
                "description": "Обсуждение планов переезда"
            },
            {
                "name": "Финансовые вопросы и кредиты",
                "keywords": ["финансы", "кредиты"],
                "percentage": 20.0,
                "sentiment": "negative",
                "description": "Финансовые трудности"
            }
        ]
    }
    
    # Тестовые данные коммерческой оценки в НОВОМ формате (как возвращает GPT-4o)
    test_commercial_assessment = {
        "commercial_assessment": [
            {
                "topic_name": "Путешествия и жилье",
                "commercial_potential": "medium",
                "realistic_revenue": "15,000-30,000 руб/мес",
                "monetization_methods": [
                    {
                        "method": "Консультации по переезду",
                        "description": "Помощь в выборе жилья",
                        "target_audience": "Мигранты",
                        "startup_cost": "0-5,000 руб",
                        "time_to_profit": "1-2 месяца",
                        "success_probability": "70%",
                        "first_steps": ["Создать сайт", "Найти клиентов"]
                    }
                ],
                "why_this_person": "Активно обсуждает переезд"
            },
            {
                "topic_name": "Финансовые вопросы и кредиты",
                "commercial_potential": "high",
                "realistic_revenue": "50,000-100,000 руб/мес",
                "monetization_methods": [
                    {
                        "method": "Финансовые консультации",
                        "description": "Помощь в решении кредитных проблем",
                        "target_audience": "Люди с долгами",
                        "startup_cost": "10,000-20,000 руб",
                        "time_to_profit": "2-3 месяца",
                        "success_probability": "60%",
                        "first_steps": ["Получить сертификацию", "Создать портфолио"]
                    }
                ],
                "why_this_person": "Понимает финансовые проблемы"
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
    
    print("🔧 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОЙ КОММЕРЧЕСКОЙ ОЦЕНКИ")
    print("=" * 60)
    
    # Тестируем исправленный отчет
    beautiful_report = analyzer.generate_beautiful_client_report(
        test_topics_data, 
        test_commercial_assessment, 
        "Тестовый Клиент (Исправленная коммерческая оценка)"
    )
    
    print(beautiful_report)
    
    # Сохраняем отчет
    report_path = os.path.join(analyzer.output_dir, "ТЕСТ_исправленная_коммерческая_оценка.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(beautiful_report)
    
    print(f"\n💾 Отчет с исправленной коммерческой оценкой сохранен: {report_path}")
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    test_commercial_fix() 