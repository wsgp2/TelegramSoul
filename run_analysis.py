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
            
        # 🚀 ЗАПУСКАЕМ БЫСТРЫЙ АНАЛИЗ ТЕМ (ЭТАП 1)
        try:
            print("🚀 ЭТАП 1: Запускаем быстрый анализ тем...")
            topics_result = await analyzer.analyze_topics_fast(prepared_messages)
        except Exception as e:
            import traceback
            print(f"\nОшибка при быстром анализе тем: {str(e)}")
            print("\nПолная трассировка ошибки:")
            print(traceback.format_exc())
            return None
        
        if not topics_result or not topics_result.get('topics'):
            print("Не удалось выполнить быстрый анализ тем.")
            return None
        
        # Сохраняем результаты быстрого анализа тем
        topics_file = analyzer.save_results_to_json(topics_result, f"{chat_name}_fast_topics_analysis")
        print(f"🎯 Результаты быстрого анализа тем сохранены: {topics_file}")
        
        # Показываем превью найденных тем
        print("\n" + "="*70)
        print("🎯 ПРЕВЬЮ НАЙДЕННЫХ ТЕМ:")
        print("="*70)
        
        topics_data = topics_result.get('topics', [])
        if topics_data:
            print(f"📊 Найдено {len(topics_data)} тем:")
            for i, topic in enumerate(topics_data[:5], 1):  # Показываем топ 5
                name = topic.get('name', 'Неизвестная тема')
                percentage = topic.get('percentage', 0)
                sentiment = topic.get('sentiment', 'neutral')
                sentiment_emoji = {"positive": "😊", "negative": "😔", "neutral": "😐"}.get(sentiment, "😐")
                print(f"{i}. {sentiment_emoji} {name} ({percentage:.1f}%)")
        
        print(f"\n🎯 Быстрый анализ чата {chat_name} завершен!")
        print(f"📊 Найдено {len(topics_data)} тем для дальнейшего анализа")
        print("\n💡 Для получения полного отчета с монетизацией и психологией:")
        print("   1. Соберите темы из всех чатов пользователя")
        print("   2. Запустите полный Soul Analysis")
        
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
    
    # 🚀 Список API ключей для параллельной обработки из переменных окружения
    api_keys = []
    
    # Сначала пробуем загрузить OPENAI_API_KEY1, OPENAI_API_KEY2, etc.
    for i in range(1, 6):  # OPENAI_API_KEY1 до OPENAI_API_KEY5
        key = os.getenv(f'OPENAI_API_KEY{i}')
        if key:
            api_keys.append(key)
    
    # Если нет, пробуем API_KEY_1, API_KEY_2, etc.
    if not api_keys:
        for i in range(1, 6):  # API_KEY_1 до API_KEY_5
            key = os.getenv(f'API_KEY_{i}')
            if key:
                api_keys.append(key)
    
    # Если нет дополнительных ключей, используем основной
    if not api_keys:
        main_key = os.getenv('OPENAI_API_KEY')
        if main_key:
            api_keys = [main_key]
        else:
            print("❌ Ошибка: Не найдены API ключи OpenAI!")
            print("💡 Установите переменные окружения OPENAI_API_KEY1, OPENAI_API_KEY2, ... или OPENAI_API_KEY")
            return
    
    # Создаем экземпляр анализатора с множественными API ключами
    analyzer = ChatGPTAnalyzer(api_keys=api_keys)
    
    # Получаем информацию о доступных чатах
    available_chats = []
    messages_dir = Path(analyzer.messages_dir)
    
    # Ищем оптимизированные файлы данных
    for item in messages_dir.rglob("*.json"):
        if item.name.startswith("all_messages_"):
            # Файл в новом оптимизированном формате
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
                    msg_count = len(messages)
                    # Получаем имя клиента из пути
                    client_name = item.parent.name
                    available_chats.append({
                        "path": str(item.parent),
                        "name": f"Клиент {client_name} (оптимизированный формат)",
                        "messages": msg_count,
                        "file": str(item)
                    })
            except Exception as e:
                print(f"Ошибка чтения файла {item}: {e}")
    
    # Перебираем пользовательские директории (старый формат)
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
                
                # Проверяем новый оптимизированный формат
                if "file" in chat:
                    # Новый формат с прямой ссылкой на файл
                    messages_file = chat["file"]
                    try:
                        with open(messages_file, 'r', encoding='utf-8') as f:
                            messages = json.load(f)
                            print(f"\nЗагружено {len(messages)} сообщений из {chat['name']} (оптимизированный формат)")
                    except Exception as e:
                        print(f"Ошибка при загрузке файла {messages_file}: {e}")
                        continue
                else:
                    # Старый формат
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
                
                # Анализируем темы с checkpoint поддержкой
                checkpoint_name = f"{client_name}_topics_checkpoint"
                topics_result = await analyzer.analyze_topics(prepared_messages, checkpoint_base=checkpoint_name)
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
            
            # ЭТАП 2: Оценка коммерческого потенциала на основе всех тем
            print("\n--- ЭТАП 2: Оценка коммерческого потенциала на основе всех тем ---")
            commercial_assessment = await analyzer.assess_commercial_potential(all_topics_result)
            if commercial_assessment and commercial_assessment.get('commercial_assessment'):
                # Сохраняем результаты оценки коммерческого потенциала
                assessment_file = analyzer.save_results_to_json(commercial_assessment, "all_chats_commercial_assessment")
                print(f"Результаты оценки коммерческого потенциала сохранены: {assessment_file}")
                
                # Генерируем отчет
                report = analyzer.generate_report(all_topics_result, commercial_assessment)
                if report:
                    # Сохраняем отчет
                    report_path = os.path.join(analyzer.output_dir, "all_chats_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"Отчет сохранен: {report_path}")
                    
                # Генерируем исполнительное резюме
                executive_summary = analyzer.generate_executive_summary(all_topics_result, commercial_assessment, all_topics_data)
                if executive_summary:
                    summary_path = os.path.join(analyzer.output_dir, "EXECUTIVE_SUMMARY.md")
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(executive_summary)
                    print(f"Исполнительное резюме сохранено: {summary_path}")
                
                # 🎉 ГЕНЕРИРУЕМ ГОТОВЫЙ ОТЧЕТ ДЛЯ КЛИЕНТА (все чаты)
                print("Создаем готовый отчет для клиента по всем чатам...")
                client_report = analyzer.generate_comprehensive_client_report(all_topics_result, commercial_assessment, "ВСЕ_ЧАТЫ")
                
                # Сохраняем готовый отчет для клиента
                client_report_path = os.path.join(analyzer.output_dir, "ВСЕ_ЧАТЫ_ГОТОВЫЙ_ОТЧЕТ_ДЛЯ_КЛИЕНТА.md")
                with open(client_report_path, 'w', encoding='utf-8') as f:
                    f.write(client_report)
                print(f"🎉 ГОТОВЫЙ ОТЧЕТ ПО ВСЕМ ЧАТАМ сохранен: {client_report_path}")
            
            # Создаем простое резюме для всех тем
            summary = analyzer.create_simple_summary(all_topics_result)
            if summary:
                print("\n" + summary)
            
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
