#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import argparse
import json
from dotenv import load_dotenv
from pathlib import Path
from chatgpt_analyzer import ChatGPTAnalyzer

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем API ключ OpenAI из переменных окружения
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API ключ OpenAI не найден. Убедитесь, что переменная OPENAI_API_KEY установлена в файле .env")

async def analyze_chat(analyzer, chat_path, limit=1000):
    """Анализирует один чат и возвращает результаты"""
    # Определяем, является ли путь файлом или директорией
    is_file = os.path.isfile(chat_path)
    is_dir = os.path.isdir(chat_path)
    
    # Определяем имя чата
    if is_file:
        chat_name = os.path.basename(os.path.dirname(chat_path))
    else:
        chat_name = os.path.basename(chat_path)
    
    print(f"\nЗапуск анализа чата: {chat_name}")
    print(f"Максимальное количество сообщений для анализа: {limit}")
    
    try:
        # Загружаем сообщения напрямую из файла, если указан файл
        if is_file and chat_path.endswith('.json'):
            try:
                with open(chat_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    print(f"Загружено {len(messages)} сообщений из файла {chat_path}")
            except Exception as e:
                print(f"Ошибка при загрузке файла: {e}")
                return None
        # Если указана директория, используем стандартный метод загрузки
        elif is_dir:
            messages = await analyzer.load_messages_from_dir(directory=chat_path)
            if not messages:
                # Пробуем загрузить файл messages.json напрямую
                messages_file = os.path.join(chat_path, "messages.json")
                if os.path.exists(messages_file):
                    try:
                        with open(messages_file, 'r', encoding='utf-8') as f:
                            messages = json.load(f)
                            print(f"Загружено {len(messages)} сообщений из файла {messages_file}")
                    except Exception as e:
                        print(f"Ошибка при загрузке файла: {e}")
                        return None
                else:
                    print(f"Не удалось найти файл messages.json в директории {chat_path}")
                    return None
        else:
            print(f"Указанный путь не существует или не является файлом/директорией: {chat_path}")
            return None
        
        if not messages:
            print("Не удалось загрузить сообщения для анализа.")
            return None
        
        # Подготавливаем сообщения для анализа
        prepared_messages = analyzer.prepare_messages_for_analysis(messages, sample_size=limit)
        if not prepared_messages:
            print("Нет текстовых сообщений для анализа после предварительной обработки.")
            return None
            
        # Запускаем анализ тем
        try:
            topics_result = await analyzer.analyze_topics(prepared_messages)
        except Exception as e:
            import traceback
            print(f"\nОшибка при анализе тем: {str(e)}")
            print("\nПолная трассировка ошибки:")
            print(traceback.format_exc())
            return None
        
        if not topics_result or not topics_result.get('topics'):
            print("Не удалось проанализировать темы.")
            return None
        
        # Сохраняем результаты анализа тем
        topics_file = analyzer.save_results_to_json(topics_result, f"{chat_name}_topics_analysis")
        print(f"Результаты анализа тем сохранены: {topics_file}")
        
        # Анализируем стратегии монетизации
        monetization_result = await analyzer.develop_monetization_strategies(topics_result)
        if monetization_result and monetization_result.get('monetization_strategies'):
            # Сохраняем результаты анализа монетизации
            monetization_file = analyzer.save_results_to_json(monetization_result, f"{chat_name}_monetization_analysis")
            print(f"Результаты анализа монетизации сохранены: {monetization_file}")
            
            # Создаем бизнес-план
            business_plan = await analyzer.create_business_plan(topics_result, monetization_result)
            if business_plan:
                # Сохраняем бизнес-план
                business_plan_file = analyzer.save_results_to_json(business_plan, f"{chat_name}_business_plan")
                print(f"Бизнес-план сохранен: {business_plan_file}")
                
                # Генерируем отчет
                report = analyzer.generate_report(topics_result, monetization_result, business_plan)
                if report:
                    # Сохраняем отчет
                    report_path = os.path.join(analyzer.output_dir, f"{chat_name}_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"Отчет сохранен: {report_path}")
                
                # Создаем визуализации
                visualizations = analyzer.visualize_topics(topics_result)
                if visualizations:
                    print("\nСозданы визуализации:")
                    for vis in visualizations:
                        print(f"- {vis}")
        
        print(f"\nАнализ чата {chat_name} успешно завершен!")
        print(f"Найдено {len(topics_result.get('topics', []))} основных тем")
        return topics_result
    
    except Exception as e:
        import traceback
        print(f"\nОшибка при анализе чата {chat_name}: {str(e)}")
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
        return None

async def main():
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Анализ Telegram-чатов с использованием ChatGPT")
    parser.add_argument(
        "chat_path", 
        nargs="?", 
        help="Путь к файлу JSON с сообщениями чата или директория с сообщениями"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=1000, 
        help="Максимальное количество сообщений для анализа (по умолчанию: 1000)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="Показать список доступных чатов для анализа"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Анализировать все чаты в директории"
    )
    parser.add_argument(
        "--min-messages", 
        type=int, 
        default=200, 
        help="Минимальное количество сообщений в чате для анализа (по умолчанию: 200)"
    )
    parser.add_argument(
        "--max-messages", 
        type=int, 
        default=1500, 
        help="Максимальное количество сообщений в чате для анализа (по умолчанию: 1500)"
    )
    
    args = parser.parse_args()
    
    # Создаем экземпляр анализатора с API ключом
    analyzer = ChatGPTAnalyzer(api_key=API_KEY)
    
    # Получаем информацию о доступных чатах
    available_chats = []
    messages_dir = Path(analyzer.messages_dir)
    
    # Перебираем пользовательские директории и файлы с сообщениями
    for item in messages_dir.iterdir():
        if item.name.startswith("user_") and item.is_dir():
            # Проверяем наличие файла messages.json
            msg_file = item.joinpath("messages.json")
            if msg_file.exists():
                # Пытаемся загрузить JSON для получения информации о чате
                try:
                    with open(msg_file, 'r', encoding='utf-8') as f:
                        messages = json.load(f)
                        msg_count = len(messages)
                        available_chats.append({
                            "path": str(item),
                            "name": item.name,
                            "messages": msg_count
                        })
                except Exception as e:
                    print(f"Ошибка чтения файла {msg_file}: {e}")
        elif item.name.startswith("telegram_messages_") and item.suffix == ".json":
            # Проверяем файлы с сообщениями напрямую
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    msg_count = len(messages)
                    available_chats.append({
                        "path": str(item),
                        "name": item.name,
                        "messages": msg_count
                    })
            except Exception as e:
                print(f"Ошибка чтения файла {item}: {e}")
    
    # Если запрошен список чатов, показываем его и выходим
    if args.list:
        print("Доступные чаты для анализа:")
        counter = 1
        
        # Сортируем по количеству сообщений (по убыванию)
        available_chats.sort(key=lambda x: x["messages"], reverse=True)
        
        for chat in available_chats:
            print(f"{counter}. {chat['name']} - {chat['messages']} сообщений")
            counter += 1
        
        return
    
    # Анализ всех чатов
    if args.all:
        print(f"Запуск анализа всех чатов с {args.min_messages} до {args.max_messages} сообщений")
        print(f"Результаты будут сохранены в: {analyzer.output_dir}")
        print("\nАнализ может занять продолжительное время, пожалуйста, подождите...")
        
        # Фильтруем чаты по количеству сообщений
        filtered_chats = [chat for chat in available_chats 
                         if args.min_messages <= chat["messages"] <= args.max_messages]
        
        if not filtered_chats:
            print(f"Не найдено чатов с {args.min_messages}-{args.max_messages} сообщениями.")
            return
        
        print(f"Найдено {len(filtered_chats)} чатов для анализа:")
        for i, chat in enumerate(filtered_chats, 1):
            print(f"{i}. {chat['name']} - {chat['messages']} сообщений")
        
        # Сначала анализируем темы для всех чатов
        print("\n--- ЭТАП 1: Сбор тем из всех чатов ---")
        all_topics = []
        all_topics_data = []
        
        for chat in filtered_chats:
            try:
                # Загружаем сообщения
                chat_path = chat["path"]
                messages_file = os.path.join(chat_path, "messages.json")
                
                if os.path.exists(messages_file):
                    try:
                        with open(messages_file, 'r', encoding='utf-8') as f:
                            messages = json.load(f)
                            print(f"\nЗагружено {len(messages)} сообщений из {chat['name']}")
                    except Exception as e:
                        print(f"Ошибка при загрузке файла {messages_file}: {e}")
                        continue
                else:
                    print(f"Файл сообщений не найден: {messages_file}")
                    continue
                
                # Подготавливаем сообщения для анализа
                prepared_messages = analyzer.prepare_messages_for_analysis(messages, sample_size=args.limit)
                if not prepared_messages:
                    print(f"Нет текстовых сообщений для анализа в {chat['name']}")
                    continue
                
                # Анализируем темы
                topics_result = await analyzer.analyze_topics(prepared_messages)
                if not topics_result or not topics_result.get('topics'):
                    print(f"Не удалось проанализировать темы в {chat['name']}")
                    continue
                
                # Сохраняем результаты анализа тем
                topics_file = analyzer.save_results_to_json(topics_result, f"{chat['name']}_topics_analysis")
                print(f"Результаты анализа тем для {chat['name']} сохранены: {topics_file}")
                
                # Добавляем темы в общий список
                chat_topics = topics_result.get('topics', [])
                for topic in chat_topics:
                    topic['source_chat'] = chat['name']
                    all_topics.append(topic)
                
                # Сохраняем данные для последующей обработки
                all_topics_data.append({
                    "chat_name": chat["name"],
                    "message_count": chat["messages"],
                    "topics": chat_topics
                })
                
                print(f"Найдено {len(chat_topics)} тем в чате {chat['name']}")
                
            except Exception as e:
                import traceback
                print(f"\nОшибка при анализе чата {chat['name']}: {str(e)}")
                print(traceback.format_exc())
                continue
        
        # Сохраняем все найденные темы
        if all_topics:
            all_topics_result = {"topics": all_topics}
            all_topics_file = os.path.join(analyzer.output_dir, "all_chats_topics.json")
            with open(all_topics_file, 'w', encoding='utf-8') as f:
                json.dump(all_topics_result, f, ensure_ascii=False, indent=2)
            print(f"\nВсе темы ({len(all_topics)}) сохранены в: {all_topics_file}")
            
            # Сохраняем агрегированные результаты анализа тем
            all_topics_data_file = os.path.join(analyzer.output_dir, "all_chats_analysis.json")
            with open(all_topics_data_file, 'w', encoding='utf-8') as f:
                json.dump(all_topics_data, f, ensure_ascii=False, indent=2)
            print(f"Данные по всем чатам сохранены в: {all_topics_data_file}")
            
            # ЭТАП 2: Анализ монетизации на основе всех тем
            print("\n--- ЭТАП 2: Анализ монетизации на основе всех тем ---")
            monetization_result = await analyzer.develop_monetization_strategies(all_topics_result)
            if monetization_result and monetization_result.get('monetization_strategies'):
                # Сохраняем результаты анализа монетизации
                monetization_file = analyzer.save_results_to_json(monetization_result, "all_chats_monetization_analysis")
                print(f"Результаты анализа монетизации сохранены: {monetization_file}")
                
                # Создаем бизнес-план
                business_plan = await analyzer.create_business_plan(all_topics_result, monetization_result)
                if business_plan:
                    # Сохраняем бизнес-план
                    business_plan_file = analyzer.save_results_to_json(business_plan, "all_chats_business_plan")
                    print(f"Бизнес-план сохранен: {business_plan_file}")
                    
                    # Генерируем отчет
                    report = analyzer.generate_report(all_topics_result, monetization_result, business_plan)
                    if report:
                        # Сохраняем отчет
                        report_path = os.path.join(analyzer.output_dir, "all_chats_report.md")
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(report)
                        print(f"Отчет сохранен: {report_path}")
            
            # Создаем визуализации для всех тем
            visualizations = analyzer.visualize_topics(all_topics_result)
            if visualizations:
                print("\nСозданы визуализации для всех тем:")
                for vis in visualizations:
                    print(f"- {vis}")
            
            print(f"\nУспешно проанализировано {len(all_topics_data)} из {len(filtered_chats)} чатов")
            print(f"Общее количество выявленных уникальных тем: {len(all_topics)}")
        else:
            print("Не удалось проанализировать ни одну тему.")
        
        return
    
    # Если путь к чату не указан, запрашиваем его у пользователя
    chat_path = args.chat_path
    if not chat_path:
        print("Укажите путь к файлу с сообщениями чата или директорию с сообщениями.")
        print("Для просмотра доступных чатов запустите скрипт с параметром --list")
        print("Для анализа всех чатов запустите скрипт с параметром --all")
        return
    
    # Анализ одного указанного чата
    print(f"Результаты будут сохранены в: {analyzer.output_dir}")
    await analyze_chat(analyzer, chat_path, limit=args.limit)

if __name__ == "__main__":
    asyncio.run(main())
