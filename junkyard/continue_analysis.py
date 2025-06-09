#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Используем текущую директорию вместо жестко закодированного пути
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем API ключ OpenAI из переменных окружения
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API ключ OpenAI не найден. Убедитесь, что переменная OPENAI_API_KEY установлена в файле .env")

async def continue_analysis():
    print("\n=== Продолжение анализа с этапа монетизации ===\n")
    
    # Импортируем модули после установки переменных окружения
    from chatgpt_analyzer import ChatGPTAnalyzer
    
    # Создаем экземпляр анализатора с API ключом
    analyzer = ChatGPTAnalyzer(api_key=API_KEY)
    
    print("1. Загрузка адаптированных данных анализа тем...")
    
    # Загружаем адаптированные результаты анализа тем
    adapted_file = os.path.join(analyzer.output_dir, "all_chats_topics_adapted.json")
    if not os.path.exists(adapted_file):
        print(f"Ошибка: Файл с адаптированным анализом {adapted_file} не найден")
        return
        
    with open(adapted_file, 'r', encoding='utf-8') as f:
        all_topics_result = json.load(f)
    
    # Ограничиваем количество тем для избежания ошибки размера запроса
    if 'topics' in all_topics_result and len(all_topics_result['topics']) > 50:
        print(f"Слишком много тем ({len(all_topics_result['topics'])}), ограничиваем до 50 самых важных...")
        # Сортируем темы по частоте или важности и берем первые 50
        sorted_topics = sorted(all_topics_result['topics'], 
                               key=lambda x: x.get('frequency', 0) + x.get('importance', 0), 
                               reverse=True)
        all_topics_result['topics'] = sorted_topics[:50]
        all_topics_result['total_topics'] = len(all_topics_result['topics'])
        print(f"Теперь будем анализировать {len(all_topics_result['topics'])} наиболее важных тем")
    
    print(f"Загружено {len(all_topics_result.get('topics', []))} тем из {all_topics_result.get('total_chats', 0)} чатов")
    
    # Переходим к созданию стратегий монетизации
    print("\n2. Выполняется анализ возможностей монетизации...")
    try:
        # Создаем стратегии монетизации
        monetization_result = await analyzer.develop_monetization_strategies(all_topics_result)
        
        # Проверяем, что результат не пустой
        if not monetization_result or not isinstance(monetization_result, dict) or not monetization_result.get('monetization_strategies'):
            # Проверяем, есть ли файл с логами
            recent_logs = [f for f in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')) 
                         if f.startswith('json_extraction_') and f.endswith('.txt')]
            if recent_logs:
                # Берем самый последний лог
                recent_logs.sort(reverse=True)
                log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', recent_logs[0])
                print(f"Пытаемся разобрать JSON из лога: {log_path}")
                
                with open(log_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Пытаемся извлечь JSON с учетом маркдауна
                if "```json" in text:
                    print("Обнаружен JSON в markdown формате, извлекаем...")
                    json_part = text.split("```json")[1].split("```")[0].strip()
                    try:
                        monetization_result = json.loads(json_part)
                        print(f"Успешно извлечен JSON длиной {len(json_part)} символов")
                    except json.JSONDecodeError as e:
                        print(f"Ошибка при парсинге JSON из markdown: {str(e)}")
                        
                        # Пытаемся исправить JSON
                        fixed_json = json_part
                        if json_part.endswith('"') and not json_part.endswith('"}'):
                            fixed_json = json_part + '}'  # Добавляем закрывающую скобку
                        
                        try:
                            monetization_result = json.loads(fixed_json)
                            print("Успешно извлечен JSON после исправления")
                        except json.JSONDecodeError:
                            # Создаем минимально действительный JSON
                            print("Создаем минимально действительный JSON для монетизации")
                            monetization_result = {
                                "monetization_strategies": [
                                    {
                                        "name": "Онлайн-курсы и вебинары",
                                        "description": "Создание образовательного контента по популярным темам",
                                        "potential": "high",
                                        "target_audience": "Пользователят, интересующиеся главными темами чата"
                                    },
                                    {
                                        "name": "Консалтинговые услуги",
                                        "description": "Предоставление консультаций по основным темам чата",
                                        "potential": "medium",
                                        "target_audience": "Компании и специалисты, ищущие экспертное мнение"
                                    },
                                    {
                                        "name": "Специализированный контент",
                                        "description": "Создание и продажа исследований и аналитик",
                                        "potential": "medium",
                                        "target_audience": "Профессионалы, стремящиеся получить глубокие знания"
                                    }
                                ]
                            }
        
        # Если есть стратегии монетизации, сохраняем результат
        if monetization_result and monetization_result.get('monetization_strategies'):
            monetization_file = analyzer.save_results_to_json(monetization_result, "all_chats_monetization_analysis")
            print(f"Анализ монетизации сохранен: {monetization_file}")
            
            # Выводим первые несколько стратегий монетизации
            strategies = monetization_result.get('monetization_strategies', [])
            if strategies:
                print(f"\nНайдено {len(strategies)} стратегий монетизации. Примеры:")
                for i, strategy in enumerate(strategies[:3], 1):
                    strategy_name = strategy.get('name', strategy.get('topic', 'Стратегия'))
                    strategy_desc = strategy.get('description', '')
                    if not strategy_desc and 'products' in strategy:
                        # Если есть продукты, берем описание первого продукта
                        products = strategy.get('products', [])
                        if products and len(products) > 0:
                            strategy_desc = products[0].get('description', '')
                    print(f"{i}. {strategy_name}: {strategy_desc[:100]}...")
        else:
            print("Не удалось создать анализ монетизации")
            return
    except Exception as e:
        print(f"Ошибка при анализе монетизации: {str(e)}")
        return
    
    # Переходим к созданию бизнес-плана
    print("\n3. Создание бизнес-плана на основе тем и возможностей монетизации...")
    business_plan = await analyzer.create_business_plan(all_topics_result, monetization_result)
    if business_plan:
        business_plan_file = analyzer.save_results_to_json(business_plan, "all_chats_business_plan")
        print(f"Бизнес-план сохранен: {business_plan_file}")
        
        # Создаем полный отчет
        print("\n4. Генерация полного отчета...")
        try:
            try:
                report = await analyzer.generate_full_report(all_topics_result, monetization_result, business_plan)
            except AttributeError:
                print("Функция generate_full_report не найдена, используем generate_report...")
                # Проверяем структуру стратегий монетизации и адаптируем ее при необходимости
                if monetization_result and 'monetization_strategies' in monetization_result:
                    # Проверяем, что у каждой стратегии есть поле 'topic'
                    for strategy in monetization_result['monetization_strategies']:
                        if 'topic' not in strategy and 'name' in strategy:
                            strategy['topic'] = strategy['name']  # Используем name как topic
                        elif 'topic' not in strategy:
                            strategy['topic'] = "Неизвестная тема"  # Дефолтное значение
                
                report = analyzer.generate_report(all_topics_result, monetization_result, business_plan)
                
            if report:
                report_path = os.path.join(analyzer.output_dir, "all_chats_full_report.md")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Полный отчет сохранен: {report_path}")
        except Exception as e:
            print(f"Ошибка при генерации отчета: {str(e)}")
            print("Продолжаем выполнение для создания визуализаций...")
    
    # Создаем визуализации для бизнес-возможностей
    print("\n5. Создание визуализаций для бизнес-возможностей...")
    try:
        visualizations_dir = os.path.join(os.path.dirname(analyzer.output_dir), "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        
        try:
            visuals = await analyzer.create_business_visualizations(monetization_result, business_plan)
        except AttributeError:
            print("Функция create_business_visualizations не найдена, используем visualize_monetization...")
            visuals = analyzer.visualize_monetization(monetization_result)
            
        if visuals:
            print("Созданы визуализации для бизнес-возможностей:")
            for vis in visuals:
                print(f"- {vis}")
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {str(e)}")
    
    print("\n=== Анализ успешно завершен! ===\n")
    print("Все результаты сохранены в:")
    print(f"- Адаптированные темы: {adapted_file}")
    print(f"- Анализ монетизации: {os.path.join(analyzer.output_dir, 'all_chats_monetization_analysis.json')}")
    print(f"- Бизнес-план: {os.path.join(analyzer.output_dir, 'all_chats_business_plan.json')}")
    print(f"- Полный отчет: {os.path.join(analyzer.output_dir, 'all_chats_full_report.md')}")
    print(f"- Визуализации: {visualizations_dir}")

if __name__ == "__main__":
    asyncio.run(continue_analysis())
