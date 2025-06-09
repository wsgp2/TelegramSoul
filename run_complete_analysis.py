#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import argparse
from dotenv import load_dotenv
from pathlib import Path
from chatgpt_analyzer import ChatGPTAnalyzer

# Загружаем переменные окружения из .env файла
load_dotenv()

async def run_complete_soul_analysis(analyzer, chat_paths, user_name="Пользователь", limit=1000):
    """
    Запускает полный двухэтапный Soul Analysis:
    ЭТАП 1: Быстрый анализ тем для каждого чата (10-15 тем)
    ЭТАП 2: Глубокий анализ монетизации и психологии на основе всех тем
    """
    
    print(f"\n🌟 ЗАПУСК ПОЛНОГО SOUL ANALYSIS ДЛЯ ПОЛЬЗОВАТЕЛЯ: {user_name}")
    print("=" * 70)
    
    all_topics = []
    successful_chats = []
    
    # 🚀 ЭТАП 1: Быстрый анализ тем для каждого чата
    print("🚀 ЭТАП 1: БЫСТРЫЙ АНАЛИЗ ТЕМ ДЛЯ КАЖДОГО ЧАТА")
    print("-" * 50)
    
    for i, chat_path in enumerate(chat_paths, 1):
        chat_name = os.path.basename(chat_path.rstrip('/'))
        print(f"\n📁 [{i}/{len(chat_paths)}] Анализируем чат: {chat_name}")
        
        try:
            # Быстрый анализ тем для одного чата
            chat_result = await analyzer.run_fast_topic_analysis(
                chat_name=chat_name,
                messages_limit=limit,
                save_results=True
            )
            
            if chat_result and chat_result.get('topics'):
                topics_count = len(chat_result['topics'])
                print(f"   ✅ Найдено {topics_count} тем")
                
                # Добавляем метаинформацию о чате к каждой теме
                for topic in chat_result['topics']:
                    topic['source_chat'] = chat_name
                
                all_topics.extend(chat_result['topics'])
                successful_chats.append(chat_name)
            else:
                print(f"   ❌ Не удалось проанализировать чат {chat_name}")
                
        except Exception as e:
            print(f"   ❌ Ошибка при анализе чата {chat_name}: {e}")
    
    if not all_topics:
        print("\n❌ Не удалось проанализировать ни одного чата!")
        return None
    
    print(f"\n🎯 ЭТАП 1 ЗАВЕРШЕН!")
    print(f"📊 Успешно проанализировано чатов: {len(successful_chats)}")
    print(f"📈 Общее количество найденных тем: {len(all_topics)}")
    print(f"📝 Чаты: {', '.join(successful_chats)}")
    
    # Агрегируем и нормализуем темы
    print("\n🔄 Агрегация и нормализация тем...")
    aggregated_topics = analyzer._aggregate_similar_topics(all_topics)
    
    # Нормализуем проценты
    total_percentage = sum(topic.get('percentage', 0) for topic in aggregated_topics)
    if total_percentage > 100:
        normalization_factor = 100 / total_percentage
        for topic in aggregated_topics:
            topic['percentage'] = topic.get('percentage', 0) * normalization_factor
    
    aggregated_data = {"topics": aggregated_topics}
    
    print(f"📊 После агрегации осталось {len(aggregated_topics)} уникальных тем")
    
    # 🎯 ЭТАП 2: Глубокий анализ монетизации и психологии
    print("\n" + "=" * 70)
    print("🎯 ЭТАП 2: ГЛУБОКИЙ АНАЛИЗ МОНЕТИЗАЦИИ И ПСИХОЛОГИИ")
    print("-" * 50)
    
    try:
        complete_result = await analyzer.run_complete_soul_analysis(
            all_topics_data=aggregated_data,
            user_name=user_name,
            save_results=True
        )
        
        if complete_result:
            print("\n🚀 ПОЛНЫЙ SOUL ANALYSIS ЗАВЕРШЕН УСПЕШНО!")
            
            # Показываем краткую статистику
            topics_count = len(complete_result.get('topics', []))
            monetization_count = len(complete_result.get('monetization_analysis', []))
            psychological_data = complete_result.get('psychological_analysis', {})
            patterns_count = len(psychological_data.get('behavior_patterns', []))
            
            print(f"📊 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   🎯 Тем: {topics_count}")
            print(f"   💰 Возможностей монетизации: {monetization_count}")
            print(f"   🧠 Психологических паттернов: {patterns_count}")
            
            # Показываем топ темы с коммерческим потенциалом
            monetization_analysis = complete_result.get('monetization_analysis', [])
            high_potential = [m for m in monetization_analysis if m.get('commercial_score') == 'high']
            
            if high_potential:
                print(f"\n🔥 ТОП ВОЗМОЖНОСТИ ДЛЯ ЗАРАБОТКА:")
                for analysis in high_potential[:3]:
                    topic = analysis.get('topic', 'Неизвестная тема')
                    revenue = analysis.get('realistic_revenue', 'не оценено')
                    print(f"   💵 {topic}: {revenue}")
            
            # Показываем ключ трансформации
            transformation_key = psychological_data.get('transformation_key', '')
            if transformation_key:
                print(f"\n✨ КЛЮЧ К ТРАНСФОРМАЦИИ:")
                print(f"   {transformation_key}")
            
            if complete_result.get('report_path'):
                print(f"\n📄 ПОЛНЫЙ ОТЧЕТ СОХРАНЕН: {complete_result['report_path']}")
            
            return complete_result
        else:
            print("❌ Не удалось выполнить глубокий анализ")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка при глубоком анализе: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Полный двухэтапный Soul Analysis всех чатов пользователя")
    parser.add_argument(
        "chat_paths",
        nargs="+",
        help="Пути к директориям с чатами для анализа"
    )
    parser.add_argument(
        "--user-name",
        default="Пользователь",
        help="Имя пользователя для отчета (по умолчанию: Пользователь)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Максимальное количество сообщений для анализа каждого чата (по умолчанию: 1000)"
    )
    
    args = parser.parse_args()
    
    # Получаем API ключи
    api_keys = []
    for i in range(1, 6):
        key = os.getenv(f'OPENAI_API_KEY{i}')
        if key:
            api_keys.append(key)
    
    if not api_keys:
        main_key = os.getenv('OPENAI_API_KEY')
        if main_key:
            api_keys = [main_key]
        else:
            raise ValueError("API ключи OpenAI не найдены. Убедитесь, что переменные OPENAI_API_KEY или OPENAI_API_KEY1-5 установлены в файле .env")
    
    print(f"🔑 Используется {len(api_keys)} API ключей для параллельной обработки")
    
    # Создаем анализатор
    analyzer = ChatGPTAnalyzer(api_keys=api_keys)
    
    # Проверяем существование путей к чатам
    valid_chat_paths = []
    for chat_path in args.chat_paths:
        if os.path.exists(chat_path):
            valid_chat_paths.append(chat_path)
            print(f"✅ Найден чат: {os.path.basename(chat_path.rstrip('/'))}")
        else:
            print(f"❌ Не найден путь: {chat_path}")
    
    if not valid_chat_paths:
        print("❌ Не найдено ни одного валидного пути к чатам!")
        return
    
    # Запускаем полный анализ
    result = await run_complete_soul_analysis(
        analyzer=analyzer,
        chat_paths=valid_chat_paths,
        user_name=args.user_name,
        limit=args.limit
    )
    
    if result:
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    else:
        print("\n😞 АНАЛИЗ ЗАВЕРШИЛСЯ С ОШИБКАМИ")

if __name__ == "__main__":
    asyncio.run(main()) 