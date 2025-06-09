#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для демонстрации нового красивого формата отображения тем
"""

from chatgpt_analyzer import ChatGPTAnalyzer
import os

def test_beautiful_format():
    """Тестирует новый красивый формат отображения"""
    
    # Создаем тестовые данные (имитируем результат анализа)
    test_topics_data = {
        "topics": [
            {
                "name": "Инвестиции и бизнес",
                "keywords": ["инвестиции", "бизнес", "стартап", "деньги"],
                "percentage": 22.3,
                "sentiment": "positive",
                "description": "Обсуждение инвестиционных возможностей, бизнес-идей и стратегий развития"
            },
            {
                "name": "Путешествия и туризм", 
                "keywords": ["путешествия", "отпуск", "страны", "отели"],
                "percentage": 18.7,
                "sentiment": "positive",
                "description": "Планирование поездок, обмен опытом путешествий, рекомендации мест"
            },
            {
                "name": "Технологии и IT",
                "keywords": ["технологии", "программирование", "AI", "софт"],
                "percentage": 15.2,
                "sentiment": "neutral",
                "description": "Обсуждение новых технологий, инструментов разработки и IT-трендов"
            },
            {
                "name": "Личное развитие",
                "keywords": ["саморазвитие", "образование", "навыки", "рост"],
                "percentage": 12.8,
                "sentiment": "positive", 
                "description": "Самообразование, развитие навыков, личностный рост"
            },
            {
                "name": "Здоровье и спорт",
                "keywords": ["здоровье", "спорт", "фитнес", "питание"],
                "percentage": 8.4,
                "sentiment": "positive",
                "description": "Обсуждение тренировок, здорового образа жизни и питания"
            },
            {
                "name": "Развлечения",
                "keywords": ["фильмы", "игры", "музыка", "досуг"],
                "percentage": 4.2,
                "sentiment": "positive",
                "description": "Рекомендации фильмов, игр, обсуждение культурных событий"
            }
        ]
    }
    
    # Тестовые данные коммерческой оценки
    test_commercial_assessment = {
        "commercial_assessment": [
            {
                "topic": "Инвестиции и бизнес",
                "commercial_score": "Высокий коммерческий потенциал",
                "products": [
                    {
                        "name": "Инвестиционные консультации",
                        "description": "Персональные консультации по инвестициям",
                        "revenue_potential": "высокий"
                    },
                    {
                        "name": "Курс по финансовой грамотности",
                        "description": "Обучающий курс по основам инвестирования",
                        "revenue_potential": "средний"
                    }
                ]
            },
            {
                "topic": "Путешествия и туризм",
                "commercial_score": "Средний коммерческий потенциал",
                "products": [
                    {
                        "name": "Планировщик путешествий",
                        "description": "Сервис для планирования поездок",
                        "revenue_potential": "средний"
                    }
                ]
            }
        ]
    }
    
    # Инициализируем анализатор (без реальных API ключей для теста)
    try:
        analyzer = ChatGPTAnalyzer(api_key="test_key")
    except:
        # Создаем базовый анализатор для тестирования
        analyzer = ChatGPTAnalyzer.__new__(ChatGPTAnalyzer)
        analyzer.output_dir = "data/reports"
        os.makedirs(analyzer.output_dir, exist_ok=True)
    
    print("🎯 ТЕСТИРОВАНИЕ КРАСИВОГО ФОРМАТА ОТОБРАЖЕНИЯ ТЕМ")
    print("=" * 60)
    
    # Тестируем красивый формат тем
    print("\n💎 КРАСИВЫЙ ФОРМАТ ТЕМ:")
    print("-" * 40)
    beautiful_topics = analyzer.generate_beautiful_topic_format(test_topics_data)
    print(beautiful_topics)
    
    print("\n" + "=" * 60)
    
    # Тестируем полный красивый отчет
    print("\n📊 ПОЛНЫЙ КРАСИВЫЙ ОТЧЕТ:")
    print("-" * 40)
    beautiful_report = analyzer.generate_beautiful_client_report(
        test_topics_data, 
        test_commercial_assessment, 
        "Тестовый Клиент"
    )
    print(beautiful_report)
    
    # Сохраняем тестовый отчет
    report_path = os.path.join(analyzer.output_dir, "ТЕСТ_красивый_отчет.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(beautiful_report)
    
    print(f"\n💾 Тестовый отчет сохранен: {report_path}")
    print("\n✅ Тестирование завершено успешно!")

if __name__ == "__main__":
    test_beautiful_format() 