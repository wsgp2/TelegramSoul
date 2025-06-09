#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Тестовый скрипт для демонстрации понятного для клиента формата
"""

from chatgpt_analyzer import ChatGPTAnalyzer
import os

def test_clear_format():
    """Тестирует понятный для клиента формат"""
    
    # Тестовые данные
    test_topics_data = {
        "topics": [
            {
                "name": "Путешествия и отдых",
                "keywords": ["путешествия", "отпуск", "отель"],
                "percentage": 28.5,
                "sentiment": "positive",
                "description": "Планирование поездок, выбор направлений, обмен впечатлениями о путешествиях"
            },
            {
                "name": "Работа и карьера",
                "keywords": ["работа", "проект", "зарплата"],
                "percentage": 22.1,
                "sentiment": "neutral",
                "description": "Обсуждение рабочих вопросов, карьерных возможностей и профессионального развития"
            },
            {
                "name": "Здоровье и спорт",
                "keywords": ["спорт", "здоровье", "тренировка"],
                "percentage": 18.3,
                "sentiment": "positive",
                "description": "Разговоры о фитнесе, здоровом образе жизни и спортивных достижениях"
            },
            {
                "name": "Семья и отношения",
                "keywords": ["семья", "дети", "отношения"],
                "percentage": 15.7,
                "sentiment": "positive",
                "description": "Обсуждение семейных вопросов, воспитания детей и личных отношений"
            },
            {
                "name": "Хобби и увлечения",
                "keywords": ["хобби", "фотография", "музыка"],
                "percentage": 15.4,
                "sentiment": "positive",
                "description": "Разговоры о творческих занятиях, увлечениях и досуге"
            }
        ]
    }
    
    # Коммерческие данные
    test_commercial_assessment = {
        "commercial_assessment": [
            {
                "topic": "Путешествия и отдых",
                "commercial_score": "Высокий коммерческий потенциал",
                "products": [
                    {
                        "name": "Планировщик путешествий",
                        "description": "Персональная помощь в организации поездок",
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
    
    print("📋 ТЕСТИРОВАНИЕ ПОНЯТНОГО ДЛЯ КЛИЕНТА ФОРМАТА")
    print("=" * 60)
    
    # Тестируем понятный формат
    beautiful_topics = analyzer.generate_beautiful_topic_format(test_topics_data)
    print(beautiful_topics)
    
    print("\n" + "=" * 60)
    
    # Полный отчет
    beautiful_report = analyzer.generate_beautiful_client_report(
        test_topics_data, 
        test_commercial_assessment, 
        "Анна Иванова"
    )
    
    # Сохраняем отчет
    report_path = os.path.join(analyzer.output_dir, "ДЕМО_понятный_отчет.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(beautiful_report)
    
    print(f"💾 Демо отчет сохранен: {report_path}")
    print("\n✅ Тестирование понятного формата завершено!")

if __name__ == "__main__":
    test_clear_format() 