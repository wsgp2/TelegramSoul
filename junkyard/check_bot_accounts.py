#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для проверки наличия ботов в собранных данных Telegram.
Анализирует собранные сообщения и ищет потенциальные чаты с ботами.
"""

import os
import json
import re
from pathlib import Path
import argparse
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_bot_by_username(username):
    """Проверяет, является ли username предположительно ботом по паттерну"""
    if not username:
        return False
        
    # Типичные окончания для имен ботов
    bot_patterns = [
        r'[_]?bot$',
        r'Bot$',
        r'[_]?robot$',
        r'_official$',
        r'assistant$'
    ]
    
    for pattern in bot_patterns:
        if re.search(pattern, username):
            return True
    
    return False

def analyze_messages_dir(base_dir, fix_mode=False):
    """
    Анализ директории с сообщениями для поиска потенциальных ботов
    
    Args:
        base_dir (str): Путь к директории с данными
        fix_mode (bool): Если True, удаляет директории с сообщениями от ботов
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.error(f"❌ Директория {base_dir} не существует")
        return
    
    # Статистика
    total_dirs = 0
    potential_bot_dirs = 0
    empty_message_dirs = 0
    removed_dirs = 0
    
    # Ищем все пользовательские директории
    user_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('user_')]
    total_dirs = len(user_dirs)
    
    logger.info(f"📊 Найдено {total_dirs} пользовательских директорий")
    
    potential_bots = []
    
    for user_dir in user_dirs:
        user_id = user_dir.name.replace('user_', '')
        msg_file = user_dir / 'messages.json'
        
        # Проверяем наличие файла сообщений
        if not msg_file.exists():
            logger.warning(f"⚠️ Директория {user_dir} не содержит файла messages.json")
            empty_message_dirs += 1
            continue
        
        try:
            # Загружаем сообщения
            with open(msg_file, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            
            if not messages:
                logger.warning(f"⚠️ Файл {msg_file} содержит пустой список сообщений")
                empty_message_dirs += 1
                continue
                
            # Анализируем содержимое сообщений на предмет признаков бота
            bot_score = 0
            
            # Проверяем шаблоны сообщений, характерные для ботов
            for msg in messages:
                if not msg.get('message'):
                    continue
                
                message_text = msg.get('message', '')
                
                # Характерные для ботов признаки в сообщениях
                if re.search(r'(Welcome|Добро пожаловать|Выберите команду|Нажмите кнопку|Отправьте команду)', message_text):
                    bot_score += 1
                    
                if re.search(r'(/start|/help|/settings|/menu)', message_text):
                    bot_score += 1
                
                # Проверяем на типичные шаблоны ответов ботов
                if re.search(r'(Для справки используйте команду|I can help you with|I am a bot)', message_text):
                    bot_score += 2
            
            # Если есть признаки бота
            if bot_score >= 3:
                bot_info = {
                    'id': user_id,
                    'dir': str(user_dir),
                    'message_count': len(messages),
                    'bot_score': bot_score
                }
                potential_bots.append(bot_info)
                potential_bot_dirs += 1
                
                logger.warning(f"🤖 Обнаружен потенциальный бот: {user_dir} (score: {bot_score})")
                
                # В режиме исправления удаляем директорию с ботом
                if fix_mode:
                    import shutil
                    shutil.rmtree(user_dir)
                    logger.info(f"🗑️ Удалена директория с ботом: {user_dir}")
                    removed_dirs += 1
                    
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {msg_file}: {e}")
    
    # Сохраняем информацию о потенциальных ботах
    if potential_bots:
        with open('potential_bots.json', 'w', encoding='utf-8') as f:
            json.dump(potential_bots, f, ensure_ascii=False, indent=2)
    
    # Выводим итоговую статистику
    logger.info("\n============= ИТОГИ АНАЛИЗА =============")
    logger.info(f"📁 Всего пользовательских директорий: {total_dirs}")
    logger.info(f"🤖 Потенциальных ботов: {potential_bot_dirs}")
    logger.info(f"📭 Директорий без сообщений: {empty_message_dirs}")
    
    if fix_mode:
        logger.info(f"🗑️ Удалено директорий с ботами: {removed_dirs}")
    
    return potential_bots

def main():
    parser = argparse.ArgumentParser(description="Анализ директории с сообщениями на наличие ботов")
    parser.add_argument('--dir', type=str, default='data/messages', help='Директория с сообщениями')
    parser.add_argument('--fix', action='store_true', help='Удалять обнаруженные директории с ботами')
    
    args = parser.parse_args()
    
    logger.info(f"🔍 Начинаем анализ директории: {args.dir}")
    logger.info(f"🔧 Режим исправления: {'Включен' if args.fix else 'Выключен'}")
    
    analyze_messages_dir(args.dir, args.fix)

if __name__ == "__main__":
    main()
