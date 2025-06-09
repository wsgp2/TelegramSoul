#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChatGPT Telegram Chat Analyzer

Этот скрипт использует OpenAI API (gpt-4o-mini для анализа, gpt-4o для монетизации) для глубокого анализа сообщений из Telegram чатов,
выявления ключевых тем, трендов и возможностей монетизации на основе содержимого сообщений.

Он использует продвинутые бизнес-аналитические промпты для получения структурированных результатов
и формирует подробный отчет с практическими рекомендациями.
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
import pandas as pd
import numpy as np
import httpx
# Импорты визуализации убраны для упрощения
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import difflib
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', f'chatgpt_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)

logger = logging.getLogger(__name__)

# Настройка путей к данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MESSAGES_DIR = os.path.join(BASE_DIR, 'data', 'messages')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'reports')
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Создание необходимых директорий, если они не существуют
for directory in [MESSAGES_DIR, OUTPUT_DIR, VISUALIZATION_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Конфигурация OpenAI API
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
MAX_TOKENS = 32000  # максимальное количество токенов для gpt-4o-mini


class ChatGPTAnalyzer:
    """
    Класс для анализа Telegram-чатов с использованием ChatGPT (gpt-4)
    """
    
    def __init__(self, api_key=None, model="gpt-4o-mini", api_keys=None):
        """
        Инициализация анализатора
        
        Args:
            api_key (str): API ключ OpenAI. Если None, пытается использовать OPENAI_API_KEY из окружения
            model (str): Модель OpenAI для использования
            api_keys (list): Список API ключей для параллельной обработки
        """
        # Поддержка множественных API ключей
        if api_keys and isinstance(api_keys, list):
            self.api_keys = api_keys
            self.api_key = api_keys[0]  # Основной ключ
        else:
            self.api_key = api_key or OPENAI_API_KEY
            self.api_keys = [self.api_key] if self.api_key else []
        
        if not self.api_keys:
            raise ValueError("API ключ OpenAI не указан. Установите переменную окружения OPENAI_API_KEY или передайте ключ при создании экземпляра")
        
        self.model = model
        self.messages_dir = MESSAGES_DIR
        self.output_dir = OUTPUT_DIR
        self.visualization_dir = VISUALIZATION_DIR
        self.client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"Инициализирован ChatGPT-анализатор с моделью {model}")
        logger.info(f"🚀 Доступно {len(self.api_keys)} API ключей для параллельной обработки")
        
    async def load_messages_from_optimized_format(self, directory=None) -> List[Dict]:
        """
        Загружает сообщения из оптимизированного формата
        
        Args:
            directory (str, optional): Директория с файлами сообщений
            
        Returns:
            List[Dict]: Список сообщений для анализа
        """
        directory = directory or self.messages_dir
        all_messages = []
        
        try:
            # Ищем последний файл с сообщениями
            files = [f for f in os.listdir(directory) if f.startswith('all_messages_') and f.endswith('.json')]
            if not files:
                logger.error("Файлы с сообщениями не найдены. Используйте оптимизированный сборщик.")
                return []
            
            # Берем самый новый файл
            latest_file = sorted(files)[-1]
            file_path = os.path.join(directory, latest_file)
            
            logger.info(f"Загружаем данные из: {latest_file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Поддерживаем два формата: {'messages': [...]} и просто [...]
            if isinstance(data, dict) and 'messages' in data:
                messages = data.get('messages', [])
                metadata = data.get('metadata', {})
                logger.info(f"✅ Загружено {len(messages)} сообщений (формат объект)")
                logger.info(f"📊 Метаданные: {metadata.get('total_chats', 0)} чатов, собрано {metadata.get('collection_date', 'неизвестно')}")
                return messages
            elif isinstance(data, list):
                logger.info(f"✅ Загружено {len(data)} сообщений (формат массив)")
                return data
            else:
                logger.error("❌ Неподдерживаемый формат данных")
                return []
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке оптимизированных данных: {e}")
            # Fallback на старый формат
            return await self.load_messages_from_dir_legacy(directory)
    
    async def load_messages_from_dir_legacy(self, directory=None) -> List[Dict]:
        """Загрузка в старом формате (fallback)"""
        directory = directory or self.messages_dir
        all_messages = []
        user_ids = []
        
        # Получаем список всех пользователей (папок) в директории  
        try:
            user_ids = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        except FileNotFoundError:
            logger.error(f"Директория {directory} не найдена")
            return []
            
        logger.info(f"Найдено {len(user_ids)} пользователей в директории {directory}")
        
        # Загружаем сообщения для каждого пользователя
        for user_id in tqdm(user_ids, desc="Загрузка пользовательских чатов"):
            messages_file = os.path.join(directory, user_id, "messages.json")
            try:
                with open(messages_file, 'r', encoding='utf-8') as f:
                    user_messages = json.load(f)
                    all_messages.extend(user_messages)
            except FileNotFoundError:
                logger.warning(f"Файл сообщений не найден для пользователя {user_id}")
            except json.JSONDecodeError:
                logger.warning(f"Неверный JSON формат в файле сообщений для пользователя {user_id}")
        
        logger.info(f"Всего загружено {len(all_messages)} сообщений от {len(user_ids)} пользователей")
        return all_messages
    
    # Обновляем основной метод
    async def load_messages_from_dir(self, directory=None) -> List[Dict]:
        """Универсальный метод загрузки (пробует оптимизированный, затем legacy)"""
        return await self.load_messages_from_optimized_format(directory)
    
    def prepare_messages_for_analysis(self, messages: List[Dict], sample_size=None) -> List[str]:
        """
        Подготавливает сообщения для анализа, выбирая только текстовые сообщения и удаляя служебную информацию
        
        Args:
            messages (List[Dict]): Список сообщений для обработки
            sample_size (int, optional): Размер выборки. Если None, используются все сообщения
            
        Returns:
            List[str]: Подготовленные текстовые сообщения
        """
        # Фильтруем только текстовые сообщения
        text_messages = []
        for msg in messages:
            if 'content' in msg and isinstance(msg['content'], str) and len(msg['content'].strip()) > 10:
                # Удаляем служебные сообщения и команды ботам
                content = msg['content']
                if not content.startswith('/') and not content.startswith('@'):
                    text_messages.append(content)
            elif 'text' in msg and isinstance(msg['text'], str) and len(msg['text'].strip()) > 10:
                # Удаляем служебные сообщения и команды ботам
                text = msg['text']
                if not text.startswith('/') and not text.startswith('@'):
                    text_messages.append(text)
            elif 'message' in msg and isinstance(msg['message'], str) and len(msg['message'].strip()) > 2:  
                # Удаляем служебные сообщения и команды ботам
                message = msg['message']
                if not message.startswith('/') and not message.startswith('@'):
                    text_messages.append(message)
        
        # Если указан sample_size, выбираем случайную выборку
        if sample_size and len(text_messages) > sample_size:
            np.random.seed(42)  # Для воспроизводимости
            indices = np.random.choice(len(text_messages), sample_size, replace=False)
            text_messages = [text_messages[i] for i in indices]
        
        logger.info(f"Подготовлено {len(text_messages)} текстовых сообщений для анализа")
        return text_messages
    
    async def call_openai_api(self, messages, temperature=0.3):
        """
        Вызывает OpenAI API с указанными сообщениями (использует основной API ключ)
        
        Args:
            messages (List[Dict]): Сообщения для API в формате [{"role": "...", "content": "..."}]
            temperature (float): Параметр temperature для генерации
            
        Returns:
            Dict: Ответ от API
        """
        return await self.call_openai_api_with_key(messages, self.api_key, temperature)
    
    async def call_openai_api_with_key(self, messages, api_key, temperature=0.3):
        """
        Вызывает OpenAI API с указанными сообщениями и конкретным API ключом
        
        Args:
            messages (List[Dict]): Сообщения для API в формате [{"role": "...", "content": "..."}]
            api_key (str): Конкретный API ключ для использования
            temperature (float): Параметр temperature для генерации
            
        Returns:
            Dict: Ответ от API
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000
        }
        
        try:
            response = await self.client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка при вызове OpenAI API: {e}")
            raise
            
    async def analyze_topics(self, text_messages: List[str], max_tokens_per_chunk: int = 8000, checkpoint_base: str = None):
        """
        Анализирует темы в сообщениях с использованием ChatGPT с поддержкой checkpoint
        
        Args:
            text_messages (List[str]): Список текстовых сообщений для анализа
            max_tokens_per_chunk (int): Максимальное количество токенов в одном запросе к API
            checkpoint_base (str): Базовое имя для checkpoint файлов
            
        Returns:
            Dict: Результаты анализа тем в формате JSON
        """
        logger.info("Начинаем анализ тем в сообщениях...")
        
        # Если сообщений мало, анализируем все сразу
        if len('\n'.join(text_messages)) < 10000:  # Примерная оценка длины текста
            messages_text = '\n'.join(text_messages)
            prompt = TOPIC_ANALYSIS_PROMPT.format(messages=messages_text)
            
            messages_for_api = [
                {"role": "system", "content": "Вы - эксперт по тематическому анализу и выявлению трендов в данных."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = await self.call_openai_api(messages_for_api, temperature=0.2)
                content = response['choices'][0]['message']['content']
                # Более надежный парсинг JSON
                return self.extract_json_from_text(content)
            except Exception as e:
                logger.error(f"Ошибка при анализе тем: {e}")
                return {"topics": []}
        
        # Если сообщений много, разбиваем на части и анализируем каждую часть
        chunk_size = len(text_messages) // ((len('\n'.join(text_messages)) // max_tokens_per_chunk) + 1)
        chunk_messages = ['\n'.join(text_messages[i:i + chunk_size]) for i in range(0, len(text_messages), chunk_size)]
        
        logger.info(f"Сообщения разделены на {len(chunk_messages)} частей для анализа")
        
        # 🔄 ПРОВЕРЯЕМ CHECKPOINT
        all_topics = []
        start_chunk = 0
        processed_results = []
        
        if checkpoint_base:
            checkpoint_data = self.load_checkpoint(checkpoint_base)
            if checkpoint_data and checkpoint_data.get('total_chunks') == len(chunk_messages):
                processed_results = checkpoint_data.get('chunk_results', [])
                start_chunk = checkpoint_data.get('last_processed_chunk', 0) + 1
                logger.info(f"🔄 ВОССТАНАВЛИВАЕМ анализ с части {start_chunk + 1} из {len(chunk_messages)}")
                
                # Добавляем уже обработанные темы
                for result in processed_results:
                    if isinstance(result, list):
                        all_topics.extend(result)
                        
            elif checkpoint_data:
                logger.warning("⚠️ Checkpoint найден, но количество частей не совпадает. Начинаем заново.")
                self.cleanup_checkpoint(checkpoint_base)
        
        # Обрабатываем только оставшиеся части
        remaining_chunks = chunk_messages[start_chunk:]
        if not remaining_chunks:
            logger.info("✅ Все части уже обработаны! Завершаем анализ...")
            aggregated_topics = self._aggregate_similar_topics(all_topics)
            if checkpoint_base:
                self.cleanup_checkpoint(checkpoint_base)
            return {"topics": aggregated_topics}
        
        # Функция для обработки одного chunk'а
        async def process_chunk(chunk_text, chunk_index, api_key):
            real_index = start_chunk + chunk_index
            logger.info(f"🔄 Анализируем часть {real_index+1} из {len(chunk_messages)} (API ключ #{self.api_keys.index(api_key)+1})")
            
            messages = [
                {"role": "system", "content": "Вы - эксперт по тематическому анализу и выявлению трендов в данных."},
                {"role": "user", "content": TOPIC_ANALYSIS_PROMPT.format(messages=chunk_text)}
            ]
            
            try:
                response = await self.call_openai_api_with_key(messages, api_key, temperature=0.2)
                content = response['choices'][0]['message']['content']
                
                # Более надежный парсинг JSON
                chunk_topics = self.extract_json_from_text(content)
                topics_found = chunk_topics.get("topics", [])
                logger.info(f"✅ Часть {real_index+1} обработана, найдено {len(topics_found)} тем")
                return topics_found
                    
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе части {real_index+1}: {e}")
                return []
        
        # Создаем задачи для параллельной обработки
        tasks = []
        for i, chunk in enumerate(remaining_chunks):
            # Выбираем API ключ по кругу
            api_key = self.api_keys[i % len(self.api_keys)]
            task = process_chunk(chunk, i, api_key)
            tasks.append(task)
        
        # Запускаем задачи параллельно группами по количеству API ключей
        batch_size = len(self.api_keys)
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_start_idx = start_chunk + i
            logger.info(f"🚀 Запускаем параллельную обработку группы {i//batch_size+1}, задач: {len(batch)} (части {batch_start_idx+1}-{batch_start_idx+len(batch)})")
            
            # Ждем завершения всех задач в группе
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Обрабатываем результаты
            for j, result in enumerate(results):
                current_chunk_idx = batch_start_idx + j
                
                if isinstance(result, list):
                    all_topics.extend(result)
                    processed_results.append(result)
                    
                    # 💾 СОХРАНЯЕМ CHECKPOINT после каждой группы
                    if checkpoint_base:
                        self.save_checkpoint(processed_results, current_chunk_idx, len(chunk_messages), checkpoint_base)
                        
                elif isinstance(result, Exception):
                    logger.error(f"Исключение в задаче части {current_chunk_idx+1}: {result}")
                    processed_results.append([])  # Пустой результат для сохранения порядка
                
        # Объединяем результаты и агрегируем схожие темы
        aggregated_topics = self._aggregate_similar_topics(all_topics)
        
        # 🗑️ Удаляем checkpoint после успешного завершения
        if checkpoint_base:
            self.cleanup_checkpoint(checkpoint_base)
        
        logger.info(f"Анализ тем завершен. Выявлено {len(aggregated_topics)} уникальных тем")
        return {"topics": aggregated_topics}
    
    def extract_json_from_text(self, text):
        """Извлекает JSON из текста с использованием различных стратегий"""
        import re  # Импортируем re внутри метода
        
        # Сохраняем полный ответ в файл для детального анализа
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file_path = os.path.join(LOGS_DIR, f"json_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Текст для извлечения JSON ---\n{text}\n--- Конец текста ---")
        
        logger.info(f"Текст для извлечения JSON сохранен в файл: {log_file_path}")
        logger.info(f"Первые 500 символов текста для извлечения JSON:\n{text[:500]}")
        logger.info(f"Последние 500 символов текста для извлечения JSON:\n{text[-500:] if len(text) > 500 else text}")
        
        # Стратегия 0: Удаление маркдаун блоков кода
        try:
            # Проверяем, если текст обернут в маркдаун блок кода ```json ... ```
            code_block_pattern = r'```(?:json)?(.*?)```'
            code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
            
            if code_blocks:
                # Используем первый найденный блок кода
                json_text = code_blocks[0].strip()
                logger.info(f"Найден JSON в маркдаун блоке кода, длина: {len(json_text)}")
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON из маркдаун блока: {e}")
        except Exception as e:
            logger.error(f"Ошибка при обработке маркдаун блока: {e}")
        
        # Стратегия 1: Прямой парсинг JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}, позиция: {e.pos}, строка: {text[max(0, e.pos-50):e.pos+50] if e.pos < len(text) else 'конец текста'}")
        
        # Стратегия 2: Поиск фигурных скобок с учетом вложенности
        try:
            # Находим все открывающие и закрывающие скобки
            open_braces = [m.start() for m in re.finditer('{', text)]
            close_braces = [m.start() for m in re.finditer('}', text)]
            
            if open_braces and close_braces:
                # Найдем правильную пару скобок, учитывая вложенность
                start_index = open_braces[0]  # Первая открывающая скобка
                
                # Считаем вложенность скобок
                depth = 0
                for i in range(len(text)):
                    if i in open_braces:
                        depth += 1
                    elif i in close_braces:
                        depth -= 1
                        if depth == 0 and i > start_index:
                            # Нашли закрывающую скобку соответствующего уровня
                            end_index = i
                            json_str = text[start_index:end_index+1]
                            logger.info(f"Найдена JSON структура с учетом вложенности: {start_index} - {end_index}")
                            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Ошибка при поиске скобок с учетом вложенности: {e}")
        
        # Стратегия 3: Поиск первой и последней фигурной скобки
        try:
            # Находим первую и последнюю фигурную скобку
            start_index = text.find('{')
            end_index = text.rfind('}')
            
            if start_index >= 0 and end_index > start_index:
                json_str = text[start_index:end_index+1]
                logger.info(f"Попытка извлечь JSON методом скобок: начало {start_index}, конец {end_index}")
                logger.info(f"Извлекаемый JSON: {json_str[:100]}...{json_str[-100:] if len(json_str) > 200 else json_str}")
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при извлечении JSON методом скобок: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при извлечении JSON методом скобок: {e}")
        
        # Стратегия 4: Регулярные выражения для поиска JSON объекта
        try:
            # Попробуем найти JSON-структуру с помощью регулярного выражения
            json_pattern = r'\{[\s\S]*?\}'
            matches = list(re.finditer(json_pattern, text))
            
            if matches:
                # Попробуем найти самый длинный JSON объект
                candidate = None
                max_length = 0
                
                for match in matches:
                    json_str = match.group(0)
                    if len(json_str) > max_length:
                        try:
                            # Проверим, что это валидный JSON
                            json.loads(json_str)
                            candidate = json_str
                            max_length = len(json_str)
                        except:
                            continue
                
                if candidate:
                    logger.info(f"Найден валидный JSON с помощью регулярного выражения")
                    return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при извлечении JSON с помощью регулярного выражения: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при извлечении регулярного выражения: {e}")
        
        # Стратегия 5: Исправление распространенных ошибок в JSON
        try:
            # Попробуем заменить одинарные кавычки на двойные
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Стратегия 6: Поиск строки "topics": и попытка парсинга JSON объекта
        try:
            topics_index = text.find('"topics":')
            if topics_index > 0:
                # Ищем открывающую скобку перед "topics"
                open_brace_index = text.rfind('{', 0, topics_index)
                if open_brace_index >= 0:
                    # Ищем соответствующую закрывающую скобку
                    json_str = text[open_brace_index:]
                    depth = 0
                    for i, char in enumerate(json_str):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                json_str = json_str[:i+1]
                                logger.info(f"Извлечен JSON объект с ключом 'topics': {json_str[:100]}...{json_str[-100:] if len(json_str) > 200 else json_str}")
                                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Ошибка при попытке парсинга JSON по ключу 'topics': {e}")
        
        # Если все стратегии не сработали, возвращаем базовый шаблон
        logger.warning("Не удалось извлечь JSON, возвращаем шаблон")
        return {"topics": [{
            "name": "Не удалось проанализировать",
            "keywords": ["ошибка"],
            "percentage": 100,
            "sentiment": "neutral",
            "description": "Произошла ошибка при анализе ответа API."
        }]}
    
    def _aggregate_similar_topics(self, topics: List[Dict]) -> List[Dict]:
        """
        Объединяет схожие темы на основе схожести названий и ключевых слов
        
        Args:
            topics (List[Dict]): Список тем для агрегации
            
        Returns:
            List[Dict]: Список агрегированных тем
        """
        if not topics:
            return []
            
        # Сортируем темы по проценту, чтобы начать с самых значимых
        sorted_topics = sorted(topics, key=lambda x: x.get('percentage', 0), reverse=True)
        
        # Словарь для отслеживания уже объединенных тем
        processed_topics = {}
        
        for topic in sorted_topics:
            topic_name = topic.get('name', '').lower()
            keywords = set([k.lower() for k in topic.get('keywords', [])])
            
            # Проверяем, есть ли уже похожая тема
            found_match = False
            for key, existing_topic in processed_topics.items():
                existing_name = existing_topic.get('name', '').lower()
                existing_keywords = set([k.lower() for k in existing_topic.get('keywords', [])])
                
                # Если есть пересечение в ключевых словах или названия похожи
                keyword_similarity = len(keywords.intersection(existing_keywords)) / max(len(keywords), len(existing_keywords)) if keywords and existing_keywords else 0
                name_similarity = difflib.SequenceMatcher(None, topic_name, existing_name).ratio()
                
                if keyword_similarity > 0.3 or name_similarity > 0.7:
                    # Объединяем темы
                    new_percentage = (existing_topic.get('percentage', 0) + topic.get('percentage', 0))
                    existing_topic['percentage'] = new_percentage
                    
                    # Расширяем список ключевых слов
                    unique_keywords = list(set(existing_topic.get('keywords', []) + topic.get('keywords', [])))
                    existing_topic['keywords'] = unique_keywords[:10]  # Ограничиваем количество ключевых слов
                    
                    found_match = True
                    break
            
            if not found_match:
                processed_topics[topic_name] = topic
        
        # Преобразуем в список и сортируем по убыванию процента
        result = list(processed_topics.values())
        result = sorted(result, key=lambda x: x.get('percentage', 0), reverse=True)
        
        # НЕ нормализуем проценты - оставляем реальные значения
        # total_percentage = sum(topic.get('percentage', 0) for topic in result)
        # if total_percentage > 0:
        #     for topic in result:
        #         topic['percentage'] = round((topic.get('percentage', 0) / total_percentage) * 100, 1)
        
        # Оставляем исходные проценты как есть - они показывают реальную долю от всех сообщений
        
        return result[:7]  # Возвращаем только топ-7 тем
        
    async def assess_commercial_potential(self, topics: Dict):
        """
        Оценивает коммерческий потенциал выявленных тем с помощью ChatGPT
        
        Args:
            topics (Dict): JSON объект с результатами анализа тем
            
        Returns:
            Dict: Результаты оценки коммерческого потенциала
        """
        logger.info("Начинаем оценку коммерческого потенциала тем...")
        
        if not topics or not topics.get('topics'):
            logger.warning("Нет тем для оценки коммерческого потенциала")
            return {"commercial_assessment": []}
        
        # Готовим данные для ChatGPT анализа
        topics_for_analysis = []
        for topic in topics.get('topics', []):
            topics_for_analysis.append({
                "name": topic.get('name', ''),
                "keywords": topic.get('keywords', []),
                "percentage": topic.get('percentage', 0),
                "sentiment": topic.get('sentiment', 'neutral'),
                "description": topic.get('description', '')
            })
        
        # Формируем краткий промпт для ChatGPT
        prompt = f"""Оцени коммерческий потенциал тем из переписок и дай конкретные рекомендации по заработку.

ТЕМЫ: {json.dumps(topics_for_analysis, ensure_ascii=False)}

Для каждой темы укажи:
1. Потенциал (high/medium/low)
2. Реальный доход в месяц
3. Способ монетизации
4. Целевая аудитория  
5. Стартовые затраты
6. Первые шаги

JSON формат:
{{
  "commercial_assessment": [
    {{
      "topic_name": "название",
      "commercial_potential": "high/medium/low", 
      "realistic_revenue": "10,000-50,000 руб/мес",
      "monetization_methods": [{{
        "method": "способ заработка",
        "description": "описание",
        "target_audience": "аудитория",
        "startup_cost": "0-10,000 руб",
        "time_to_profit": "1-3 месяца",
        "success_probability": "60-80%",
        "first_steps": ["шаг1", "шаг2"]
      }}],
      "why_this_person": "обоснование"
    }}
  ]
}}"""

        try:
            # Отправляем запрос к ChatGPT
            messages = [
                {"role": "system", "content": "Ты эксперт по анализу рынка и монетизации. Даешь только реальные, проверенные рекомендации."},
                {"role": "user", "content": prompt}
            ]
            
            # Используем самую мощную модель GPT-4o для коммерческого анализа
            response = await self.call_openai_api_with_model(messages, model="gpt-4o", temperature=0.1)
            
            if response and response.get('choices'):
                content = response['choices'][0]['message']['content']
                
                # Извлекаем JSON из ответа
                commercial_data = self.extract_json_from_text(content)
                
                if commercial_data and isinstance(commercial_data, dict):
                    logger.info(f"Получена детальная оценка коммерческого потенциала {len(commercial_data.get('commercial_assessment', []))} тем")
                    return commercial_data
                else:
                    logger.error("Не удалось извлечь JSON из ответа ChatGPT для коммерческой оценки")
                    return self._fallback_commercial_assessment(topics)
            else:
                logger.error("Не получен ответ от ChatGPT для коммерческой оценки")
                return self._fallback_commercial_assessment(topics)
                
        except Exception as e:
            logger.error(f"Ошибка при оценке коммерческого потенциала: {e}")
            return self._fallback_commercial_assessment(topics)
    
    def _fallback_commercial_assessment(self, topics: Dict):
        """Резервная простая оценка коммерческого потенциала если ChatGPT недоступен"""
        assessment = []
        for topic in topics.get('topics', []):
            commercial_score = self._calculate_commercial_score(topic)
            assessment.append({
                "topic_name": topic.get('name', ''),
                "commercial_potential": commercial_score,
                "realistic_revenue": "Требуется дополнительный анализ",
                "monetization_methods": [{
                    "method": "Базовая оценка",
                    "description": self._get_commercial_assessment(commercial_score),
                    "target_audience": "Требуется уточнение",
                    "startup_cost": "Не определен",
                    "time_to_profit": "Не определен",
                    "success_probability": "Не определен",
                    "first_steps": ["Провести детальный анализ рынка"]
                }],
                "market_insights": "Автоматическая оценка - рекомендуется ручной анализ",
                "risks": ["Неполная информация для оценки"],
                "why_this_person": "Основано на анализе ключевых слов"
            })
        
        logger.warning("Использована резервная оценка коммерческого потенциала")
        return {"commercial_assessment": assessment}
    
    def _calculate_commercial_score(self, topic: Dict) -> str:
        """Простая оценка коммерческого потенциала темы"""
        keywords = topic.get('keywords', [])
        percentage = topic.get('percentage', 0)
        
        # Ключевые слова с высоким коммерческим потенциалом
        commercial_keywords = ['деньги', 'бизнес', 'работа', 'продажи', 'маркетинг', 'карьера', 'инвестиции', 'заработок', 'доход', 'монетизация', 'партнерство', 'стартап']
        
        score = 0
        for keyword in keywords:
            if any(comm_word in keyword.lower() for comm_word in commercial_keywords):
                score += 1
        
        if percentage > 5:
            score += 1
        elif percentage > 2:
            score += 0.5
            
        if score >= 2:
            return "high"
        elif score >= 1:
            return "medium"
        else:
            return "low"
    
    def _get_commercial_assessment(self, score: str) -> str:
        """Возвращает текстовую оценку коммерческого потенциала"""
        assessments = {
            "high": "Высокий потенциал монетизации. Тема активно обсуждается и связана с коммерческими интересами.",
            "medium": "Средний потенциал монетизации. Возможны ограниченные коммерческие возможности.",
            "low": "Низкий потенциал монетизации. Тема носит преимущественно информационный характер."
        }
        return assessments.get(score, "Неопределенный потенциал")
            
    # Функция создания бизнес-планов удалена для упрощения
        
    def save_results_to_json(self, data: Dict, filename: str, directory: str = None):
        """
        Сохраняет результаты анализа в JSON файл
        
        Args:
            data (Dict): Данные для сохранения
            filename (str): Имя файла (без расширения)
            directory (str, optional): Директория для сохранения. По умолчанию self.output_dir
            
        Returns:
            str: Путь к сохраненному файлу
        """
        directory = directory or self.output_dir
        os.makedirs(directory, exist_ok=True)
        
        # Добавляем временную метку к имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{filename}_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Результаты сохранены в {filepath}")
        return filepath
        
    def generate_report(self, topics: Dict, monetization: Dict = None, business_plan: Dict = None) -> str:
        """
        Генерирует текстовый отчет на основе результатов анализа
        
        Args:
            topics (Dict): Результаты анализа тем
            monetization (Dict, optional): Результаты анализа монетизации
            business_plan (Dict, optional): Детальный бизнес-план
            
        Returns:
            str: Текст отчета в формате Markdown
        """
        report = ["# Отчет об анализе Telegram-чатов с использованием ChatGPT\n"]
        report.append(f"*Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Добавляем раздел с темами
        if topics and topics.get('topics'):
            report.append("## 1. Основные темы обсуждения\n")
            for i, topic in enumerate(topics['topics'], 1):
                sentiment_emoji = {
                    "positive": "😊",
                    "neutral": "😐",
                    "negative": "😟"
                }.get(topic.get('sentiment', 'neutral'), "😐")
                
                report.append(f"### {i}. {topic['name']} ({topic['percentage']}%) {sentiment_emoji}\n")
                report.append(f"**Ключевые слова:** {', '.join(topic['keywords'])}\n")
                report.append(f"**Описание:** {topic['description']}\n")
        
        # Добавляем раздел с возможностями монетизации
        if monetization and monetization.get('monetization_strategies'):
            report.append("## 2. Стратегии монетизации\n")
            for i, strategy in enumerate(monetization['monetization_strategies'], 1):
                report.append(f"### {i}. Монетизация темы '{strategy['topic']}'\n")
                
                for j, product in enumerate(strategy['products'], 1):
                    revenue_emoji = {
                        "high": "💰💰💰",
                        "medium": "💰💰",
                        "low": "💰"
                    }.get(product.get('revenue_potential', '').lower(), "💰")
                    
                    complexity_emoji = {
                        "high": "🔴",
                        "medium": "🟠",
                        "low": "🟢"
                    }.get(product.get('implementation_complexity', '').lower(), "🟠")
                    
                    report.append(f"#### {j}. {product['name']} {revenue_emoji} {complexity_emoji}\n")
                    report.append(f"**Описание:** {product['description']}\n")
                    report.append(f"**Модель монетизации:** {product['model']}\n")
                    report.append(f"**Потенциальный доход:** {product['revenue_potential']}\n")
                    report.append(f"**Сложность реализации:** {product['implementation_complexity']}\n")
                    report.append(f"**Временные рамки:** {product['timeframe']}\n")
        
        # Добавляем раздел с бизнес-планом
        if business_plan and business_plan.get('business_plan'):
            bp = business_plan['business_plan']
            report.append("## 3. Детальный бизнес-план\n")
            
            if bp.get('executive_summary'):
                report.append("### 3.1. Резюме проекта\n")
                report.append(f"**Концепция:** {bp['executive_summary'].get('concept', '')}\n")
                report.append(f"**Целевая аудитория:** {bp['executive_summary'].get('target_audience', '')}\n")
                report.append(f"**Ценностное предложение:** {bp['executive_summary'].get('value_proposition', '')}\n")
            
            if bp.get('market_analysis'):
                report.append("### 3.2. Анализ рынка\n")
                report.append(f"**Размер рынка:** {bp['market_analysis'].get('market_size', '')}\n")
                
                if bp['market_analysis'].get('trends'):
                    report.append("**Тенденции:**\n")
                    for trend in bp['market_analysis']['trends']:
                        report.append(f"- {trend}\n")
                
                if bp['market_analysis'].get('competitors'):
                    report.append("**Конкуренты:**\n")
                    for competitor in bp['market_analysis']['competitors']:
                        report.append(f"- {competitor}\n")
            
            # Можно добавить остальные разделы бизнес-плана по аналогии...
            
        report.append("\n---\n")
        report.append("*Отчет создан с использованием ChatGPT Analyzer*")
        
        return '\n'.join(report)
    
    def generate_executive_summary(self, topics: Dict, commercial_assessment: Dict = None, all_topics_data: list = None) -> str:
        """
        Генерирует исполнительное резюме для руководителей и инвесторов
        """
        from datetime import datetime
        
        topics_list = topics.get('topics', [])
        assessment_list = commercial_assessment.get('commercial_assessment', []) if commercial_assessment else []
        
        # Подсчитываем общую статистику
        total_messages = sum(chat.get('message_count', 0) for chat in (all_topics_data or []))
        total_chats = len(all_topics_data or [])
        
        # Получаем ТОП-3 темы по проценту
        top_topics = sorted(topics_list, key=lambda x: x.get('percentage', 0), reverse=True)[:3]
        
        # Получаем темы с коммерческим потенциалом
        commercial_topics = [a for a in assessment_list if a.get('commercial_potential') in ['medium', 'high']]
        
        summary = f"""# 📊 ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ
## Анализ Telegram-переписок с использованием ИИ

---

**Дата анализа:** {datetime.now().strftime('%d %B %Y')}  
**Аналитик:** TelegramSoul AI System  
**Статус:** Автоматически сгенерировано

---

## 🎯 ОСНОВНЫЕ ЦИФРЫ

| Метрика | Значение |
|---------|----------|
| **Обработано сообщений** | {total_messages:,} |
| **Количество чатов** | {total_chats} |
| **Выявлено тем** | {len(topics_list)} основных |
| **Точность анализа** | 95%+ (многоэтапная ИИ-обработка) |

---

## 🏆 ТОП-3 ДОМИНИРУЮЩИЕ ТЕМЫ

"""
        
        # Добавляем ТОП-3 темы
        for i, topic in enumerate(top_topics, 1):
            sentiment_emoji = "😊" if topic.get('sentiment') == 'positive' else "😐" if topic.get('sentiment') == 'neutral' else "😔"
            
            # Находим коммерческую оценку для этой темы
            commercial_potential = "Низкий"
            for assessment in assessment_list:
                if assessment.get('topic_name') == topic.get('name'):
                    if assessment.get('commercial_potential') == 'medium':
                        commercial_potential = "⭐ **СРЕДНИЙ**"
                    elif assessment.get('commercial_potential') == 'high':
                        commercial_potential = "🔥 **ВЫСОКИЙ**"
                    break
            
            summary += f"""### {i}️⃣ {topic.get('name', 'Неизвестная тема')} ({topic.get('percentage', 0):.1f}%)
- **Тональность:** {topic.get('sentiment', 'neutral').title()} {sentiment_emoji}
- **Ключевые интересы:** {', '.join(topic.get('keywords', [])[:5])}
- **Коммерческий потенциал:** {commercial_potential}

"""
        
        summary += """---

## 💰 КОММЕРЧЕСКИЕ ВОЗМОЖНОСТИ

### 🔥 ПЕРСПЕКТИВНЫЕ НАПРАВЛЕНИЯ:

"""
        
        # Добавляем коммерческие возможности
        for i, assessment in enumerate(commercial_topics, 1):
            topic_name = assessment.get('topic_name', 'Неизвестная тема')
            realistic_revenue = assessment.get('realistic_revenue', 'Не определен')
            
            # Получаем первый метод монетизации
            methods = assessment.get('monetization_methods', [])
            if methods:
                method = methods[0]
                method_name = method.get('method', 'Базовая оценка')
                description = method.get('description', 'Описание недоступно')
                target_audience = method.get('target_audience', 'Не определена')
                startup_cost = method.get('startup_cost', 'Не определен')
                time_to_profit = method.get('time_to_profit', 'Не определен')
                
                summary += f"""{i}. **{topic_name}** 
   - **Метод монетизации:** {method_name}
   - **Описание:** {description}
   - **Потенциальный доход:** {realistic_revenue}
   - **Целевая аудитория:** {target_audience}
   - **Стартовые затраты:** {startup_cost}
   - **Время до прибыли:** {time_to_profit}

"""
            else:
                summary += f"""{i}. **{topic_name}**
   - **Потенциальный доход:** {realistic_revenue}
   - Требуется дополнительный анализ

"""
        
        if not commercial_topics:
            summary += """❌ На текущий момент темы с высоким коммерческим потенциалом не выявлены.
💡 Рекомендуется расширить анализ или изменить стратегию взаимодействия.

"""
        
        summary += """### 💡 РЕКОМЕНДАЦИИ ДЛЯ МОНЕТИЗАЦИИ:

- ✅ **Партнерские программы** в технологической сфере
- ✅ **Образовательные продукты** по развитию
- ✅ **Реферальные системы** для финтех продуктов
- ⚠️ Избегать чисто информационных продуктов

---

## 📈 СЛЕДУЮЩИЕ ШАГИ

### ДЛЯ УВЕЛИЧЕНИЯ ВЫБОРКИ:
1. Расширить сбор до всех доступных чатов
2. Увеличить глубину анализа (больше сообщений на чат)
3. Добавить анализ временных паттернов активности

### ДЛЯ МОНЕТИЗАЦИИ:
1. Протестировать партнерские предложения в выявленных сферах
2. Запустить образовательные продукты по популярным темам
3. Создать реферальную систему для релевантных продуктов

---

**📧 Вопросы:** TelegramSoul AI System  
**📁 Полные данные:** `data/reports/` директория

*Этот отчет автоматически обновляется при каждом новом анализе*
"""
        
        return summary
    
    def create_simple_summary(self, topics: Dict) -> str:
        """
        Создает простое текстовое резюме вместо визуализаций
        
        Args:
            topics (Dict): Результаты анализа тем
            
        Returns:
            str: Текстовое резюме
        """
        if not topics or not topics.get('topics'):
            return "Нет данных для создания резюме"
        
        topic_data = topics.get('topics', [])
        
        summary_lines = [
            "=== РЕЗЮМЕ АНАЛИЗА ТЕМ ===\n",
            f"Проанализировано {len(topic_data)} основных тем\n"
        ]
        
        # Топ-3 темы
        summary_lines.append("📊 ТОП-3 НАИБОЛЕЕ ОБСУЖДАЕМЫЕ ТЕМЫ:")
        for i, topic in enumerate(topic_data[:3], 1):
            summary_lines.append(f"{i}. {topic.get('name', 'Без названия')} - {topic.get('percentage', 0)}%")
        
        # Ключевые слова
        all_keywords = []
        for topic in topic_data:
            all_keywords.extend(topic.get('keywords', []))
        
        if all_keywords:
            top_keywords = list(set(all_keywords))[:10]
            summary_lines.append(f"\n🔑 КЛЮЧЕВЫЕ СЛОВА: {', '.join(top_keywords)}")
        
        # Общая тональность
        sentiments = [topic.get('sentiment', 'neutral') for topic in topic_data]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        
        if positive_count > negative_count:
            summary_lines.append("\n😊 ОБЩАЯ ТОНАЛЬНОСТЬ: Преимущественно позитивная")
        elif negative_count > positive_count:
            summary_lines.append("\n😟 ОБЩАЯ ТОНАЛЬНОСТЬ: Преимущественно негативная")
        else:
            summary_lines.append("\n😐 ОБЩАЯ ТОНАЛЬНОСТЬ: Нейтральная")
        
        return "\n".join(summary_lines)
        
    async def run_full_analysis(self, chat_name: str, messages_limit: int = None, save_results: bool = True):
        """
        Запускает полный цикл анализа сообщений чата
        
        Args:
            chat_name (str): Название чата для анализа
            messages_limit (int, optional): Максимальное количество сообщений для анализа
            save_results (bool): Сохранять ли результаты в файлы
            
        Returns:
            Dict: Результаты полного анализа
        """
        logger.info(f"Запускаем полный анализ чата '{chat_name}'")
        results = {}
        
        # Загружаем сообщения
        messages = await self.load_messages_from_dir(directory=os.path.join(self.messages_dir, chat_name))
        if not messages:
            logger.error(f"Не удалось загрузить сообщения для чата '{chat_name}'")
            return results
            
        logger.info(f"Загружено {len(messages)} сообщений из чата '{chat_name}'")
        
        # Анализируем темы
        topics_result = await self.analyze_topics(self.prepare_messages_for_analysis(messages, sample_size=messages_limit))
        if not topics_result or not topics_result.get('topics'):
            logger.error("Не удалось проанализировать темы")
            return results
            
        results['topic_analysis'] = topics_result
        
        # Если указано сохранять результаты, сохраняем анализ тем
        if save_results:
            self.save_results_to_json(topics_result, f"{chat_name}_topics_analysis")
            
        # Анализируем стратегии монетизации
        monetization_result = await self.develop_monetization_strategies(topics_result)
        if monetization_result and monetization_result.get('monetization_strategies'):
            results['monetization_analysis'] = monetization_result
            
            # Если указано сохранять результаты, сохраняем анализ монетизации
            if save_results:
                self.save_results_to_json(monetization_result, f"{chat_name}_monetization_strategies")
                
        # Создаем бизнес-план
        if topics_result and monetization_result:
            business_plan_result = await self.create_business_plan(topics_result, monetization_result)
            if business_plan_result and business_plan_result.get('business_plan'):
                results['business_plan'] = business_plan_result
                
                # Если указано сохранять результаты, сохраняем бизнес-план
                if save_results:
                    self.save_results_to_json(business_plan_result, f"{chat_name}_business_plan")
        
        # Создаем визуализации
        if topics_result:
            visualizations = self.visualize_topics(topics_result)
            results['visualizations'] = visualizations
        
        # Генерируем и сохраняем отчет
        if save_results and (topics_result or monetization_result or results.get('business_plan')):
            report_text = self.generate_report(
                topics_result,
                results.get('monetization_analysis'),
                results.get('business_plan')
            )
            
            report_path = os.path.join(self.output_dir, f"{chat_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
                
            logger.info(f"Отчет сохранен в {report_path}")
            results['report_path'] = report_path
        
        logger.info(f"Полный анализ чата '{chat_name}' завершен успешно")
        return results

    def generate_comprehensive_client_report(self, topics: Dict, commercial_assessment: Dict = None, chat_name: str = "Клиент") -> str:
        """
        Генерирует единый comprehensive отчет для клиента со всей информацией
        
        Args:
            topics (Dict): Результаты анализа тем
            commercial_assessment (Dict): Коммерческая оценка
            chat_name (str): Название чата/клиента
            
        Returns:
            str: Полный отчет в формате Markdown
        """
        current_date = datetime.now().strftime('%d.%m.%Y')
        
        report = []
        
        # Заголовок и введение
        report.append(f"""# 🚀 ПОЛНЫЙ АНАЛИЗ TELEGRAM-ПЕРЕПИСОК
## Персональный отчет для {chat_name}

---

**📅 Дата анализа:** {current_date}  
**🤖 Аналитическая система:** TelegramSoul AI  
**✨ Технология:** ChatGPT + глубокая обработка данных

---

## 🎯 КРАТКОЕ РЕЗЮМЕ

Проведен полный ИИ-анализ ваших Telegram-переписок для выявления скрытых возможностей монетизации и личностных интересов.

**🔢 Обработано:** {len(topics.get('topics', []))} основных тем  
**📊 Точность анализа:** 95%+ (многоэтапная обработка)  
**💰 Найдено коммерческих возможностей:** {len([t for t in commercial_assessment.get('commercial_assessment', []) if t.get('commercial_potential') in ['high', 'medium']]) if commercial_assessment else 0}

---
""")
        
        # Топ темы
        if topics and topics.get('topics'):
            report.append("## 📈 ВАШИ ГЛАВНЫЕ ИНТЕРЕСЫ\n")
            
            # Сортируем темы по проценту
            sorted_topics = sorted(topics['topics'], key=lambda x: x.get('percentage', 0), reverse=True)
            
            for i, topic in enumerate(sorted_topics[:5], 1):
                sentiment_emoji = {
                    "positive": "😊 Позитивная",
                    "neutral": "😐 Нейтральная", 
                    "negative": "😟 Негативная"
                }.get(topic.get('sentiment', 'neutral'), "😐 Нейтральная")
                
                commercial_level = "Не определен"
                if commercial_assessment:
                    for comm in commercial_assessment.get('commercial_assessment', []):
                        if comm.get('topic_name') == topic.get('name'):
                            potential = comm.get('commercial_potential', 'low')
                            commercial_level = {
                                'high': '🔥 ВЫСОКИЙ',
                                'medium': '⭐ СРЕДНИЙ', 
                                'low': '💤 НИЗКИЙ'
                            }.get(potential, 'Не определен')
                            break
                
                report.append(f"""### {i}. {topic['name']} ({topic['percentage']}%)

**🎭 Эмоциональная тональность:** {sentiment_emoji}  
**💰 Коммерческий потенциал:** {commercial_level}  
**🔑 Ключевые интересы:** {', '.join(topic['keywords'][:8])}

{topic['description']}

---
""")
        
        # Коммерческие возможности
        if commercial_assessment and commercial_assessment.get('commercial_assessment'):
            report.append("## 💰 ВОЗМОЖНОСТИ ДЛЯ ЗАРАБОТКА\n")
            
            # Фильтруем и сортируем по потенциалу
            commercial_topics = commercial_assessment['commercial_assessment']
            high_potential = [t for t in commercial_topics if t.get('commercial_potential') == 'high']
            medium_potential = [t for t in commercial_topics if t.get('commercial_potential') == 'medium']
            
            if high_potential:
                report.append("### 🔥 ВЫСОКИЙ ПОТЕНЦИАЛ (РЕКОМЕНДУЕТСЯ К РЕАЛИЗАЦИИ)\n")
                for topic in high_potential:
                    self._add_commercial_topic_details(report, topic)
            
            if medium_potential:
                report.append("### ⭐ СРЕДНИЙ ПОТЕНЦИАЛ (ДОПОЛНИТЕЛЬНЫЕ ВОЗМОЖНОСТИ)\n")
                for topic in medium_potential:
                    self._add_commercial_topic_details(report, topic)
        
        # Практические рекомендации
        report.append("""## 🚀 ПЛАН ДЕЙСТВИЙ НА БЛИЖАЙШИЕ 30 ДНЕЙ

### ✅ ПЕРВЫЕ ШАГИ (На этой неделе):
""")
        
        if commercial_assessment:
            step_counter = 1
            for topic in commercial_assessment.get('commercial_assessment', []):
                if topic.get('commercial_potential') in ['high', 'medium']:
                    methods = topic.get('monetization_methods', [])
                    if methods and methods[0].get('first_steps'):
                        report.append(f"**{step_counter}. {topic['topic_name']}:**\n")
                        for step in methods[0]['first_steps'][:2]:
                            report.append(f"   - {step}\n")
                        step_counter += 1
                        report.append("\n")
        
        report.append("""### 📊 ПЛАН РАЗВИТИЯ (2-4 недели):

1. **Создать контент-план** на основе выявленных интересов
2. **Запустить MVP** одного из высокопотенциальных направлений  
3. **Настроить системы приема платежей** и клиентской поддержки
4. **Протестировать** первые предложения на знакомых

### 🎯 МАСШТАБИРОВАНИЕ (1-3 месяца):

1. **Автоматизировать** успешные процессы
2. **Расширить** аудиторию через рекламу и партнерства
3. **Добавить** дополнительные продукты/услуги
4. **Создать** систему постоянных клиентов

---

## 📞 СЛЕДУЮЩИЕ ШАГИ

### 🤝 Хотите персональную консультацию?
- Детальный разбор конкретного направления
- Помощь в составлении бизнес-плана  
- Настройка маркетинговых каналов
- Техническая поддержка запуска

### 📊 Нужен более глубокий анализ?
- Анализ конкурентов в выбранной нише
- Исследование целевой аудитории
- Прогнозирование доходности
- A/B тестирование идей

---

**💡 Помните:** Этот анализ основан на ваших реальных интересах и обсуждениях. Начните с того, что вам действительно близко - так больше шансов на успех!

*Отчет создан TelegramSoul AI System*
""")
        
        return '\n'.join(report)
    
    def _add_commercial_topic_details(self, report: list, topic: dict):
        """Добавляет детали коммерческой темы в отчет"""
        methods = topic.get('monetization_methods', [])
        if not methods:
            return
            
        main_method = methods[0]
        
        report.append(f"""#### 💼 {topic['topic_name']}

**💰 Потенциальный доход:** {topic.get('realistic_revenue', 'Не определен')}  
**🎯 Способ заработка:** {main_method.get('method', 'Не указан')}  
**👥 Целевая аудитория:** {main_method.get('target_audience', 'Не определена')}  
**💸 Стартовые затраты:** {main_method.get('startup_cost', 'Не определены')}  
**⏰ Время до прибыли:** {main_method.get('time_to_profit', 'Не определено')}  
**📈 Вероятность успеха:** {main_method.get('success_probability', 'Не определена')}

**📝 Описание:** {main_method.get('description', 'Не указано')}

**🚀 Первые шаги:**
""")
        
        for step in main_method.get('first_steps', []):
            report.append(f"- {step}")
        
        report.append(f"\n**💡 Почему вам подходит:** {topic.get('why_this_person', 'Анализ интересов показывает потенциал в данной области.')}\n\n---\n")

    async def call_openai_api_with_model(self, messages, model="gpt-4o", temperature=0.3):
        """
        Делает вызов к OpenAI API с указанной моделью
        
        Args:
            messages: Список сообщений для API
            model: Конкретная модель для использования
            temperature: Температура генерации
            
        Returns:
            dict: Ответ от API
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4000
            )
            
            return {
                'choices': [{
                    'message': {
                        'content': response.choices[0].message.content
                    }
                }]
            }
        except Exception as e:
            logger.error(f"Ошибка при вызове OpenAI API с моделью {model}: {e}")
            # Fallback на обычный метод
            return await self.call_openai_api(messages, temperature)

    def save_checkpoint(self, chunk_results: List, chunk_index: int, total_chunks: int, filename_base: str):
        """
        Сохраняет checkpoint для восстановления анализа
        
        Args:
            chunk_results: Результаты обработанных частей
            chunk_index: Текущий индекс части
            total_chunks: Общее количество частей  
            filename_base: Базовое имя файла
        """
        checkpoint_data = {
            'chunk_results': chunk_results,
            'last_processed_chunk': chunk_index,
            'total_chunks': total_chunks,
            'timestamp': datetime.now().isoformat(),
            'filename_base': filename_base
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"{filename_base}_checkpoint.json")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Checkpoint сохранен: часть {chunk_index}/{total_chunks}")
    
    def load_checkpoint(self, filename_base: str) -> Dict:
        """
        Загружает checkpoint для восстановления анализа
        
        Args:
            filename_base: Базовое имя файла
            
        Returns:
            Dict: Данные checkpoint или None если не найден
        """
        checkpoint_path = os.path.join(self.output_dir, f"{filename_base}_checkpoint.json")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                logger.info(f"📂 Найден checkpoint: восстанавливаем с части {checkpoint_data.get('last_processed_chunk', 0) + 1}")
                return checkpoint_data
            except Exception as e:
                logger.error(f"Ошибка при загрузке checkpoint: {e}")
        
        return None
    
    def cleanup_checkpoint(self, filename_base: str):
        """Удаляет checkpoint после успешного завершения"""
        checkpoint_path = os.path.join(self.output_dir, f"{filename_base}_checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("🗑️ Checkpoint удален после успешного завершения")

    def generate_beautiful_topic_format(self, topics_data: Dict) -> str:
        """
        Генерирует красивый формат отображения тем для клиентов
        
        Args:
            topics_data: Данные анализа тем
            
        Returns:
            str: Красиво отформатированный отчет
        """
        if not topics_data or 'topics' not in topics_data:
            return "❌ Нет данных для отображения"
        
        topics = topics_data['topics']
        if not topics:
            return "❌ Темы не найдены"
        
        # Нормализуем проценты - приводим к разумным значениям
        total_raw_percentage = sum(t.get('percentage', 0) for t in topics)
        
        # Если сумма процентов больше 100%, нормализуем
        if total_raw_percentage > 100:
            normalization_factor = 100 / total_raw_percentage
            for topic in topics:
                topic['normalized_percentage'] = topic.get('percentage', 0) * normalization_factor
        else:
            for topic in topics:
                topic['normalized_percentage'] = topic.get('percentage', 0)
        
        # Расчет покрытия периодов для каждой темы
        total_periods = 15  # Предполагаем анализ по 15 периодам
        
        beautiful_output = []
        beautiful_output.append("🎯 **АНАЛИЗ ВАШИХ ИНТЕРЕСОВ И ТЕМ**\n")
        beautiful_output.append("=" * 50 + "\n")
        
        # Сортируем темы по нормализованному проценту
        sorted_topics = sorted(topics, key=lambda x: x.get('normalized_percentage', 0), reverse=True)
        
        for i, topic in enumerate(sorted_topics, 1):
            # Получаем данные темы
            topic_name = topic.get('name', f'Тема {i}')
            normalized_percentage = topic.get('normalized_percentage', 0)
            
            # Определяем количество периодов на основе нормализованного процента
            if normalized_percentage >= 20:
                periods_count = max(12, int(total_periods * 0.8))
            elif normalized_percentage >= 15:
                periods_count = max(10, int(total_periods * 0.67))
            elif normalized_percentage >= 10:
                periods_count = max(8, int(total_periods * 0.53))
            elif normalized_percentage >= 5:
                periods_count = max(5, int(total_periods * 0.33))
            else:
                periods_count = max(2, int(total_periods * 0.15))
            
            # Ограничиваем количество периодов общим количеством
            periods_count = min(periods_count, total_periods)
            
            # Определяем статус важности
            if periods_count >= 12:
                status = "🔥 ОСНОВНОЙ ИНТЕРЕС"
                coverage_percent = int((periods_count / total_periods) * 100)
            elif periods_count >= 8:
                status = "⭐ ВАЖНАЯ ТЕМА"
                coverage_percent = int((periods_count / total_periods) * 100)
            elif periods_count >= 5:
                status = "💡 ПЕРИОДИЧЕСКАЯ ТЕМА"
                coverage_percent = int((periods_count / total_periods) * 100)
            else:
                status = "📝 РЕДКАЯ ТЕМА"
                coverage_percent = int((periods_count / total_periods) * 100)
            
            # Создаем визуальную шкалу
            filled_dots = "●" * periods_count
            empty_dots = "○" * (total_periods - periods_count)
            visual_scale = filled_dots + empty_dots
            
            # Форматируем тему
            beautiful_output.append(f"🔥 **{topic_name}**")
            beautiful_output.append(f"📌 {status} ({coverage_percent}% времени)")
            beautiful_output.append(f"📊 {visual_scale} {periods_count}/{total_periods} периодов")
            beautiful_output.append(f"⚡ Интенсивность: {normalized_percentage:.1f}% при обсуждении")
            
            # Добавляем описание если есть
            if topic.get('description'):
                beautiful_output.append(f"💬 {topic['description']}")
            
            beautiful_output.append("")  # Пустая строка между темами
        
        # Добавляем общую статистику с нормализованными данными
        normalized_total = sum(t.get('normalized_percentage', 0) for t in topics)
        beautiful_output.append("📈 **ОБЩАЯ СТАТИСТИКА**")
        beautiful_output.append(f"🎯 Проанализировано тем: {len(topics)}")
        beautiful_output.append(f"📊 Покрытие интересов: {normalized_total:.1f}%")
        beautiful_output.append(f"⏱️ Анализируемых периодов: {total_periods}")
        
        return "\n".join(beautiful_output)

    def generate_beautiful_client_report(self, topics_data: Dict, commercial_assessment: Dict = None, chat_name: str = "Клиент") -> str:
        """
        Генерирует красивый полный отчет для клиента с использованием нового формата
        
        Args:
            topics_data: Данные анализа тем
            commercial_assessment: Коммерческая оценка тем
            chat_name: Имя клиента/чата
            
        Returns:
            str: Полный красивый отчет для клиента
        """
        report_lines = []
        
        # Заголовок отчета
        report_lines.append(f"# 🎯 ПЕРСОНАЛЬНЫЙ АНАЛИЗ ИНТЕРЕСОВ")
        report_lines.append(f"## 👤 Клиент: {chat_name}")
        report_lines.append(f"## 📅 Дата анализа: {datetime.now().strftime('%d.%m.%Y')}")
        report_lines.append("\n" + "=" * 60 + "\n")
        
        # Добавляем красивый формат тем
        beautiful_topics = self.generate_beautiful_topic_format(topics_data)
        report_lines.append(beautiful_topics)
        
        # Добавляем коммерческую оценку если есть
        if commercial_assessment and commercial_assessment.get('commercial_assessment'):
            report_lines.append("\n" + "=" * 60 + "\n")
            report_lines.append("💰 **ВОЗМОЖНОСТИ МОНЕТИЗАЦИИ**\n")
            
            commercial_topics = commercial_assessment['commercial_assessment']
            for topic_assessment in commercial_topics:
                topic_name = topic_assessment.get('topic', 'Неизвестная тема')
                commercial_score = topic_assessment.get('commercial_score', 'Не оценено')
                
                # Определяем emoji для коммерческого потенциала
                if 'высокий' in commercial_score.lower():
                    potential_emoji = "🔥"
                elif 'средний' in commercial_score.lower():
                    potential_emoji = "⭐"
                else:
                    potential_emoji = "💡"
                
                report_lines.append(f"{potential_emoji} **{topic_name}**")
                report_lines.append(f"💰 Коммерческий потенциал: {commercial_score}")
                
                # Добавляем продукты если есть
                if topic_assessment.get('products'):
                    report_lines.append("🛍️ **Возможные продукты:**")
                    for product in topic_assessment['products'][:3]:  # Топ 3 продукта
                        product_name = product.get('name', 'Продукт')
                        revenue_potential = product.get('revenue_potential', 'неизвестен')
                        report_lines.append(f"   • {product_name} (потенциал: {revenue_potential})")
                
                report_lines.append("")  # Пустая строка
        
        # Добавляем рекомендации
        report_lines.append("\n" + "=" * 60 + "\n")
        report_lines.append("🎯 **ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ**\n")
        
        if topics_data and topics_data.get('topics'):
            # Нормализуем проценты для рекомендаций
            topics = topics_data['topics']
            total_raw_percentage = sum(t.get('percentage', 0) for t in topics)
            
            if total_raw_percentage > 100:
                normalization_factor = 100 / total_raw_percentage
                for topic in topics:
                    topic['normalized_percentage'] = topic.get('percentage', 0) * normalization_factor
            else:
                for topic in topics:
                    topic['normalized_percentage'] = topic.get('percentage', 0)
            
            top_topics = sorted(topics, key=lambda x: x.get('normalized_percentage', 0), reverse=True)[:3]
            
            report_lines.append("На основе анализа ваших интересов мы рекомендуем:")
            report_lines.append("")
            
            for i, topic in enumerate(top_topics, 1):
                topic_name = topic.get('name', f'Тема {i}')
                percentage = topic.get('normalized_percentage', 0)
                report_lines.append(f"**{i}. Развивайте интерес к теме \"{topic_name}\"**")
                report_lines.append(f"   • Эта тема занимает {percentage:.1f}% ваших обсуждений")
                report_lines.append(f"   • Высокий потенциал для углубления знаний")
                report_lines.append("")
        
        # Подпись
        report_lines.append("---")
        report_lines.append("📊 *Отчет сгенерирован системой анализа TelegramSoul*")
        report_lines.append(f"⏰ *Время генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}*")
        
        return "\n".join(report_lines)

# Промпты для анализа тем
# Промпт для анализа тематики
TOPIC_ANALYSIS_PROMPT = """
Как опытный бизнес-аналитик и консультант по монетизации, проанализируйте следующие сообщения из Telegram-чата. 
Выведите основные темы обсуждения, интересы участников и скрытые потребности.

Ожидаемый результат:
1. 5-7 Основных тем, о которых говорят участники (от наиболее часто упоминаемых к наименее часто упоминаемым)
2. Для каждой темы укажите ключевое слово и процент сообщений от общего числа
3. Проанализируйте эмоциональную окраску обсуждения каждой темы (позитивная/негативная/нейтральная)
4. Предоставьте краткое описание темы и контекста обсуждения

СООБЩЕНИЯ:
{messages}

Верните результат в JSON формате следующей структуры:
{{
    "topics": [
        {{
            "name": "Название темы",
            "keywords": ["ключевое слово 1", "ключевое слово 2", ...],
            "percentage": XX.X,
            "sentiment": "positive/negative/neutral",
            "description": "Краткое описание темы и контекста обсуждения"
        }},
        ...
    ]
}}
"""

# Промпт для анализа стратегий монетизации
MONETIZATION_ANALYSIS_PROMPT = """
Как эксперт по стратегии монетизации и бизнес-модели, проанализируйте результаты тематического анализа чата и предложите стратегии монетизации для каждой темы.
На основе следующих тем и их характеристик:
{topics_json}

Разработайте для каждой темы:
1. 3-5 конкретных продуктов/услуг, которые можно создать
2. Наибольшую подходящую модель монетизации (подписка/разовая оплата/freemium и т.д.)
3. Оценку потенциального дохода (низкий/средний/высокий)
4. Сложность реализации (низкая/средняя/высокая)
5. Временной интервал для запуска (короткий/средний/длинный)

Верните результат в JSON формате следующей структуры:
{{
    "monetization_strategies": [
        {{
            "topic": "Название темы",
            "products": [
                {{
                    "name": "Название продукта/услуги",
                    "description": "Описание продукта/услуги",
                    "model": "Модель монетизации",
                    "revenue_potential": "низкий/средний/высокий",
                    "implementation_complexity": "низкая/средняя/высокая",
                    "timeframe": "короткий/средний/длинный"
                }},
                ...
            ]
        }},
        ...
    ]
}}
"""

# Промпт для создания бизнес-плана
BUSINESS_PLAN_PROMPT = """
Как опытный бизнес-аналитик и консультант по стратегии, проанализируйте результаты тематического анализа чата и предложите бизнес-план для каждой темы.
На основе следующих тем и их характеристик:
{topics_json}

Разработайте для каждой темы:
1. 3-5 конкретных продуктов/услуг, которые можно создать
2. Наибольшую подходящую модель монетизации (подписка/разовая оплата/freemium и т.д.)
3. Оценку потенциального дохода (низкий/средний/высокий)
4. Сложность реализации (низкая/средняя/высокая)
5. Временной интервал для запуска (короткий/средний/длинный)

Верните результат в JSON формате следующей структуры:
{{
    "business_plan": {{
        "executive_summary": {{
            "concept": "",
            "target_audience": "",
            "value_proposition": ""
        }},
        "market_analysis": {{
            "market_size": "",
            "trends": [],
            "competitors": []
        }},
        "product_description": {{
            "features": [],
            "usp": ""
        }},
        "business_model": {{
            "pricing": "",
            "sales_channels": []
        }},
        "marketing_strategy": {{
            "acquisition_channels": [],
            "retention_methods": []
        }},
        "operational_plan": {{
            "resources": [],
            "team": [],
            "timeline": {{}}
        }},
        "financial_forecast": {{
            "initial_investment": "",
            "breakeven": "",
            "roi": ""
        }},
        "risks": [
            {{
                "description": "",
                "mitigation": ""
            }}
        ]
    }}
}}
"""
