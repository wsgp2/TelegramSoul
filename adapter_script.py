#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем API ключ OpenAI из переменных окружения
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API ключ OpenAI не найден. Убедитесь, что переменная OPENAI_API_KEY установлена в файле .env")

def adapt_analysis_format():
    print("\n=== Адаптация формата данных анализа для последующих этапов ===\n")
    
    # Пути к файлам - исправляем жестко заданный путь /tmp/TelegramSoul на относительный путь
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(repo_dir, "data", "reports")
    analysis_file = os.path.join(reports_dir, "all_chats_analysis.json")
    adapted_file = os.path.join(reports_dir, "all_chats_topics_adapted.json")
    
    # Проверяем наличие файла с анализом
    if not os.path.exists(analysis_file):
        print(f"Ошибка: Файл с анализом {analysis_file} не найден")
        return False
    
    # Загружаем данные анализа
    print(f"Загрузка данных из {analysis_file}...")
    with open(analysis_file, 'r', encoding='utf-8') as f:
        all_chats_analysis = json.load(f)
    
    print(f"Загружено {len(all_chats_analysis)} объектов анализа чатов")
    
    # Собираем все темы из всех чатов в один список
    all_topics = []
    for chat_analysis in all_chats_analysis:
        chat_name = chat_analysis.get('chat_name', 'unknown')
        message_count = chat_analysis.get('message_count', 0)
        topics = chat_analysis.get('topics', [])
        
        print(f"Чат {chat_name}: {len(topics)} тем, {message_count} сообщений")
        
        # Добавляем темы в общий список
        for topic in topics:
            if 'source_chat' not in topic:
                topic['source_chat'] = chat_name
            all_topics.append(topic)
    
    # Создаем адаптированный формат
    adapted_analysis = {
        "topics": all_topics,
        "total_topics": len(all_topics),
        "total_chats": len(all_chats_analysis)
    }
    
    # Сохраняем адаптированный формат
    print(f"\nСоздаю адаптированный формат с {len(all_topics)} темами...")
    with open(adapted_file, 'w', encoding='utf-8') as f:
        json.dump(adapted_analysis, f, ensure_ascii=False, indent=2)
    
    print(f"Адаптированный файл сохранен: {adapted_file}")
    return adapted_file

if __name__ == "__main__":
    adapt_analysis_format()
