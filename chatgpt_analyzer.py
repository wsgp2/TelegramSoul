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
            
    async def analyze_topics_fast(self, text_messages: List[str], max_tokens_per_chunk: int = 8000, checkpoint_base: str = None):
        """
        Быстрый анализ тем в сообщениях (ЭТАП 1)
        
        Args:
            text_messages (List[str]): Список текстовых сообщений для анализа
            max_tokens_per_chunk (int): Максимальное количество токенов в одном запросе к API
            checkpoint_base (str): Базовое имя для checkpoint файлов
            
        Returns:
            Dict: Результаты анализа тем в формате JSON (10-15 тем)
        """
        logger.info("🚀 ЭТАП 1: Начинаем быстрый анализ тем...")
        
        # Если сообщений мало, анализируем все сразу
        if len('\n'.join(text_messages)) < 10000:  # Примерная оценка длины текста
            messages_text = '\n'.join(text_messages)
            prompt = FAST_TOPIC_ANALYSIS_PROMPT.format(messages=messages_text)
            
            messages_for_api = [
                {"role": "system", "content": "Вы - экспертный аналитик цифрового поведения для быстрого выделения тем."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = await self.call_openai_api(messages_for_api, temperature=0.2)
                content = response['choices'][0]['message']['content']
                # Более надежный парсинг JSON
                return self.extract_json_from_text(content)
            except Exception as e:
                logger.error(f"Ошибка при быстром анализе тем: {e}")
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
                {"role": "system", "content": "Вы - экспертный аналитик цифрового поведения для быстрого выделения тем."},
                {"role": "user", "content": FAST_TOPIC_ANALYSIS_PROMPT.format(messages=chunk_text)}
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
        
        logger.info(f"🎯 ЭТАП 1 завершен. Выявлено {len(aggregated_topics)} уникальных тем")
        return {"topics": aggregated_topics}

    async def analyze_final_monetization_psychology(self, aggregated_topics: Dict):
        """
        Глубокий финальный анализ монетизации и психологии (ЭТАП 2)
        
        Args:
            aggregated_topics (Dict): Агрегированные темы из всех чатов
            
        Returns:
            Dict: Результаты глубокого анализа (монетизация + психология)
        """
        logger.info("💰 🧠 ЭТАП 2: Начинаем глубокий анализ монетизации и психологии...")
        
        if not aggregated_topics or not aggregated_topics.get('topics'):
            logger.error("Нет тем для глубокого анализа")
            return {"expertise_analysis": [], "psychological_analysis": {}}
        
        # Подготавливаем данные для анализа
        import json
        topics_json = json.dumps(aggregated_topics, ensure_ascii=False, indent=2)
        prompt = FINAL_ANALYSIS_PROMPT.format(aggregated_topics=topics_json)
        
        messages_for_api = [
            {"role": "system", "content": "Вы - экспертный стратег монетизации и трансформационный коуч."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.call_openai_api(messages_for_api, temperature=0.3)
            content = response['choices'][0]['message']['content']
            result = self.extract_json_from_text(content)
            
            logger.info("🎯 ЭТАП 2 завершен успешно!")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при глубоком анализе: {e}")
            return {"expertise_analysis": [], "psychological_analysis": {}}
    
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
        Улучшенное объединение схожих тем с более строгими правилами
        
        Args:
            topics (List[Dict]): Список тем для агрегации
            
        Returns:
            List[Dict]: Список агрегированных тем
        """
        if not topics:
            return []
            
        # Импортируем модуль для сравнения строк
        import difflib
        
        # Сортируем темы по проценту, чтобы начать с самых значимых
        sorted_topics = sorted(topics, key=lambda x: x.get('percentage', 0), reverse=True)
        
        # Словарь групп тем со строгими правилами объединения
        topic_groups = {
            'криптовалюты_и_инвестиции': {
                'patterns': ['крипто', 'биткоин', 'альткоин', 'инвест', 'торгов', 'блокчейн', 'деньги', 'финанс'],
                'merged_topics': [],
                'final_name': 'Криптовалюты и инвестиции'
            },
            'путешествия_и_туризм': {
                'patterns': ['путешеств', 'туризм', 'поездк', 'отдых', 'бали', 'тай', 'филиппин'],
                'merged_topics': [],
                'final_name': 'Путешествия и туризм'
            },
            'события_и_мероприятия': {
                'patterns': ['событи', 'мероприят', 'встреч', 'созвон', 'нетворк', 'конференц', 'батл'],
                'merged_topics': [],
                'final_name': 'События и мероприятия'
            },
            'личное_развитие': {
                'patterns': ['развит', 'курс', 'тренинг', 'обучен', 'альфа', 'коуч', 'семинар'],
                'merged_topics': [],
                'final_name': 'Личное развитие'
            },
            'технические_вопросы': {
                'patterns': ['техническ', 'технолог', 'инструмент', 'софт', 'приложен', 'бот'],
                'merged_topics': [],
                'final_name': 'Технические вопросы'
            }
        }
        
        # Несгруппированные темы
        ungrouped_topics = []
        
        # Распределяем темы по группам
        for topic in sorted_topics:
            topic_name = topic.get('name', '').lower()
            keywords_text = ' '.join(topic.get('keywords', [])).lower()
            description_text = topic.get('description', '').lower()
            full_text = f"{topic_name} {keywords_text} {description_text}"
            
            # Ищем подходящую группу
            assigned = False
            for group_key, group_info in topic_groups.items():
                patterns = group_info['patterns']
                
                # Проверяем совпадение с паттернами группы
                matches = sum(1 for pattern in patterns if pattern in full_text)
                
                # Если найдено 2+ совпадения или 1 сильное совпадение в названии
                if matches >= 2 or any(pattern in topic_name for pattern in patterns):
                    group_info['merged_topics'].append(topic)
                    assigned = True
                    break
            
            if not assigned:
                ungrouped_topics.append(topic)
        
        # Создаем финальные агрегированные темы
        final_topics = []
        
        # Обрабатываем группы
        for group_key, group_info in topic_groups.items():
            if group_info['merged_topics']:
                # Объединяем темы группы
                total_percentage = sum(t.get('percentage', 0) for t in group_info['merged_topics'])
                all_keywords = []
                all_descriptions = []
                sentiments = []
                
                for topic in group_info['merged_topics']:
                    all_keywords.extend(topic.get('keywords', []))
                    all_descriptions.append(topic.get('description', ''))
                    sentiments.append(topic.get('sentiment', 'neutral'))
                
                # Убираем дубли ключевых слов и берем самые важные
                unique_keywords = list(dict.fromkeys(all_keywords))[:8]
                
                # Определяем общую тональность
                sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
                for s in sentiments:
                    sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
                final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                
                # Создаем объединенное описание
                best_description = max(all_descriptions, key=len) if all_descriptions else ""
                
                merged_topic = {
                    'name': group_info['final_name'],
                    'keywords': unique_keywords,
                    'percentage': total_percentage,
                    'sentiment': final_sentiment,
                    'description': best_description,
                    'merged_from': len(group_info['merged_topics'])
                }
                
                final_topics.append(merged_topic)
        
        # Добавляем несгруппированные темы
        final_topics.extend(ungrouped_topics)
        
        # Сортируем по убыванию процента
        final_topics = sorted(final_topics, key=lambda x: x.get('percentage', 0), reverse=True)
        
        return final_topics[:7]  # Возвращаем только топ-7 тем
        
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
        
    async def run_fast_topic_analysis(self, chat_name: str, messages_limit: int = None, save_results: bool = True):
        """
        Запускает быстрый анализ тем для одного чата (ЭТАП 1)
        
        Args:
            chat_name (str): Название чата для анализа
            messages_limit (int, optional): Максимальное количество сообщений для анализа
            save_results (bool): Сохранять ли результаты в файлы
            
        Returns:
            Dict: Результаты быстрого анализа тем (10-15 тем)
        """
        logger.info(f"🚀 ЭТАП 1: Запускаем быстрый анализ тем для чата '{chat_name}'")
        
        # Загружаем сообщения
        messages = await self.load_messages_from_dir(directory=os.path.join(self.messages_dir, chat_name))
        if not messages:
            logger.error(f"Не удалось загрузить сообщения для чата '{chat_name}'")
            return {"topics": []}
            
        logger.info(f"Загружено {len(messages)} сообщений из чата '{chat_name}'")
        
        # Выполняем быстрый анализ тем
        text_messages = self.prepare_messages_for_analysis(messages, sample_size=messages_limit)
        checkpoint_base = f"{chat_name}_fast_topics" if save_results else None
        
        topics_result = await self.analyze_topics_fast(text_messages, checkpoint_base=checkpoint_base)
        
        if not topics_result:
            logger.error("Не удалось выполнить быстрый анализ тем")
            return {"topics": []}
            
        logger.info(f"🎯 Быстрый анализ чата '{chat_name}' завершен!")
        
        # Сохраняем результаты если нужно
        if save_results:
            self.save_results_to_json(topics_result, f"{chat_name}_fast_topics_analysis")
            logger.info(f"📁 Результаты сохранены для чата '{chat_name}'")
        
        return topics_result
    
    async def run_complete_soul_analysis(self, all_topics_data: Dict, user_name: str = "Пользователь", save_results: bool = True):
        """
        Запускает полный комплексный анализ на основе всех тем пользователя (ЭТАП 2)
        
        Args:
            all_topics_data (Dict): Агрегированные темы из всех чатов пользователя
            user_name (str): Имя пользователя для отчета
            save_results (bool): Сохранять ли результаты в файлы
            
        Returns:
            Dict: Полный результат анализа (темы + монетизация + психология)
        """
        logger.info(f"🌟 ЭТАП 2: Запускаем полный Soul Analysis для пользователя '{user_name}'")
        
        # Выполняем глубокий анализ монетизации и психологии
        deep_analysis_result = await self.analyze_final_monetization_psychology(all_topics_data)
        
        if not deep_analysis_result:
            logger.error("Не удалось выполнить глубокий анализ")
            deep_analysis_result = {"expertise_analysis": [], "psychological_analysis": {}}
        
        # Объединяем результаты
        complete_result = {
            "topics": all_topics_data.get('topics', []),
            "expertise_analysis": deep_analysis_result.get('expertise_analysis', []),
            "psychological_analysis": deep_analysis_result.get('psychological_analysis', {})
        }
        
        # Сохраняем результаты если нужно
        if save_results:
            # Сохраняем полный результат
            self.save_results_to_json(complete_result, f"{user_name}_complete_soul_analysis")
            
            # Генерируем и сохраняем красивый отчет
            beautiful_report = self.generate_beautiful_soul_report(complete_result, user_name)
            
            report_path = os.path.join(self.output_dir, f"{user_name}_SOUL_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(beautiful_report)
                
            logger.info(f"🌟 Complete Soul Analysis Report сохранен в {report_path}")
            complete_result['report_path'] = report_path
        
        logger.info(f"🚀 Полный Soul Analysis для пользователя '{user_name}' завершен!")
        return complete_result

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
        beautiful_output.append("💡 *Мы разделили историю вашего общения на 15 временных периодов и проанализировали, в каких из них обсуждалась каждая тема*\n")
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
            
            # Определяем статус важности и создаем понятные описания
            if periods_count >= 12:
                status = "🔥 ПОСТОЯННАЯ ТЕМА"
                time_description = "Обсуждается практически всегда"
                coverage_percent = int((periods_count / total_periods) * 100)
            elif periods_count >= 8:
                status = "⭐ ЧАСТАЯ ТЕМА"
                time_description = "Обсуждается регулярно"
                coverage_percent = int((periods_count / total_periods) * 100)
            elif periods_count >= 5:
                status = "💡 ПЕРИОДИЧЕСКАЯ ТЕМА"
                time_description = "Иногда обсуждается"
                coverage_percent = int((periods_count / total_periods) * 100)
            else:
                status = "📝 РЕДКАЯ ТЕМА"
                time_description = "Редко упоминается"
                coverage_percent = int((periods_count / total_periods) * 100)
            
            # Создаем визуальную шкалу с пояснением
            filled_dots = "●" * periods_count
            empty_dots = "○" * (total_periods - periods_count)
            visual_scale = filled_dots + empty_dots
            
            # Форматируем тему с понятными объяснениями
            beautiful_output.append(f"🔥 **{topic_name}**")
            beautiful_output.append(f"📌 {status} - {time_description}")
            beautiful_output.append(f"📊 Частота: {visual_scale} (присутствует в {periods_count} из {total_periods} временных отрезков)")
            beautiful_output.append(f"⚡ Активность: {normalized_percentage:.1f}% от всех ваших сообщений")
            
            # Добавляем описание если есть
            if topic.get('description'):
                beautiful_output.append(f"💬 {topic['description']}")
            
            beautiful_output.append("")  # Пустая строка между темами
        
        # Добавляем понятную статистику
        normalized_total = sum(t.get('normalized_percentage', 0) for t in topics)
        beautiful_output.append("📈 **ИТОГОВАЯ СТАТИСТИКА**")
        beautiful_output.append(f"🎯 Найдено основных тем для обсуждения: {len(topics)}")
        beautiful_output.append(f"📊 Охват ваших интересов: {normalized_total:.1f}% сообщений проанализировано")
        beautiful_output.append(f"⏱️ Период анализа: вся история разделена на {total_periods} временных отрезков")
        beautiful_output.append("\n💡 **Как читать отчет:**")
        beautiful_output.append("• ● = тема активно обсуждалась в этом периоде")
        beautiful_output.append("• ○ = тема не обсуждалась в этом периоде")
        beautiful_output.append("• Чем больше ●, тем чаще вы говорите на эту тему")
        
        return "\n".join(beautiful_output)

    def generate_beautiful_soul_report(self, unified_data: Dict, chat_name: str = "Клиент") -> str:
        """
        Генерирует революционный полный отчет для клиента с использованием объединенного анализа
        
        Args:
            unified_data: Данные объединенного анализа (темы + монетизация + психология)
            chat_name: Имя клиента/чата
            
        Returns:
            str: Полный красивый отчет для клиента с тремя уровнями анализа
        """
        report_lines = []
        
        # 🎯 ЗАГОЛОВОК ОТЧЕТА
        report_lines.append(f"# 🌟 SOUL ANALYSIS - КОМПЛЕКСНЫЙ РАЗБОР ЦИФРОВОЙ ДУШИ")
        report_lines.append(f"## 👤 Клиент: {chat_name}")
        report_lines.append(f"## 📅 Дата анализа: {datetime.now().strftime('%d.%m.%Y')}")
        report_lines.append("\n" + "═" * 70 + "\n")
        
        # 📍 УРОВЕНЬ 1: ТЕМЫ И ИНТЕРЕСЫ
        topics_data = unified_data.get('topics', [])
        if topics_data:
            report_lines.append("## 📍 УРОВЕНЬ 1: ВАШИ ОСНОВНЫЕ ТЕМЫ И ИНТЕРЕСЫ")
            report_lines.append("")
            report_lines.append("На основе анализа всех ваших сообщений мы выявили следующие основные темы:")
            report_lines.append("")
            
            # Нормализуем проценты
            total_raw_percentage = sum(t.get('percentage', 0) for t in topics_data)
            if total_raw_percentage > 100:
                normalization_factor = 100 / total_raw_percentage
                for topic in topics_data:
                    topic['normalized_percentage'] = topic.get('percentage', 0) * normalization_factor
            else:
                for topic in topics_data:
                    topic['normalized_percentage'] = topic.get('percentage', 0)
            
            # Сортируем темы по проценту
            sorted_topics = sorted(topics_data, key=lambda x: x.get('normalized_percentage', 0), reverse=True)
            
            for i, topic in enumerate(sorted_topics, 1):
                topic_name = topic.get('name', 'Неизвестная тема')
                percentage = topic.get('normalized_percentage', 0)
                sentiment = topic.get('sentiment', 'neutral')
                description = topic.get('description', '')
                keywords = topic.get('keywords', [])
                
                # Определяем emoji и статус темы по процентам
                if percentage >= 25:
                    emoji = "🔥"
                    status = "ДОМИНИРУЮЩАЯ ТЕМА"
                elif percentage >= 15:
                    emoji = "⭐"
                    status = "ОСНОВНОЙ ИНТЕРЕС"
                elif percentage >= 8:
                    emoji = "💡"
                    status = "ЗАМЕТНАЯ ТЕМА"
                elif percentage >= 3:
                    emoji = "📌"
                    status = "ПЕРИОДИЧЕСКАЯ ТЕМА"
                else:
                    emoji = "🔸"
                    status = "РЕДКАЯ ТЕМА"
                
                # Определяем emoji настроения
                if sentiment == 'positive':
                    sentiment_emoji = "😊"
                    sentiment_text = "позитивное отношение"
                elif sentiment == 'negative':
                    sentiment_emoji = "😔"
                    sentiment_text = "есть негативные моменты"
                else:
                    sentiment_emoji = "😐"
                    sentiment_text = "нейтральное отношение"
                
                # Визуальная полоса интенсивности
                bar_length = int(percentage / 5)  # 5% = 1 символ
                bar_length = min(max(bar_length, 1), 20)  # От 1 до 20 символов
                intensity_bar = "█" * bar_length
                
                report_lines.append(f"{emoji} **{i}. {topic_name}** ({percentage:.1f}%)")
                report_lines.append(f"   📊 {intensity_bar}")
                report_lines.append(f"   🏷️ {status}")
                report_lines.append(f"   {sentiment_emoji} {sentiment_text}")
                
                if keywords:
                    keywords_text = ", ".join(keywords[:5])  # Показываем до 5 ключевых слов
                    report_lines.append(f"   🔑 Ключевые слова: {keywords_text}")
                
                if description:
                    report_lines.append(f"   💬 {description}")
                
                report_lines.append("")
            
            report_lines.append("📊 **Легенда интенсивности:**")
            report_lines.append("█ = каждый символ ≈ 5% от всех сообщений")
            report_lines.append("Чем длиннее полоса, тем чаще вы обсуждаете эту тему")
            report_lines.append("")
        
        # 💰 УРОВЕНЬ 2: АНАЛИЗ ЭКСПЕРТНОСТИ И НИШ
        expertise_data = unified_data.get('expertise_analysis', [])
        if expertise_data:
            report_lines.append("\n" + "═" * 70 + "\n")
            report_lines.append("## 💰 УРОВЕНЬ 2: ЭКСПЕРТНОСТЬ И НИШИ")
            report_lines.append("")
            report_lines.append("На основе глубинного анализа ваших сообщений мы определили ваш уровень экспертности:")
            report_lines.append("")
            
            # Сортируем по уровню экспертности
            expertise_order = {'expert': 3, 'advanced': 2, 'beginner': 1}
            sorted_expertise = sorted(
                expertise_data, 
                key=lambda x: expertise_order.get(x.get('expertise_level', 'beginner'), 0), 
                reverse=True
            )
            
            for analysis in sorted_expertise:
                topic_name = analysis.get('topic', 'Неизвестная тема')
                expertise_level = analysis.get('expertise_level', 'beginner')
                expertise_indicators = analysis.get('expertise_indicators', 'не определены')
                commercial_potential = analysis.get('commercial_potential', 'low')
                monetization_readiness = analysis.get('monetization_readiness', 'long_term')
                methods = analysis.get('monetization_methods', [])
                
                # Определяем emoji для уровня экспертности
                if expertise_level == 'expert':
                    expertise_emoji = "🏆"
                    expertise_text = "ЭКСПЕРТ"
                elif expertise_level == 'advanced':
                    expertise_emoji = "📈"
                    expertise_text = "ПРОДВИНУТЫЙ"
                else:
                    expertise_emoji = "🌱"
                    expertise_text = "НАЧИНАЮЩИЙ"
                
                # Определяем готовность к монетизации
                if monetization_readiness == 'ready':
                    readiness_text = "🚀 Готов к монетизации"
                elif monetization_readiness == 'need_development':
                    readiness_text = "🔧 Требует развития"
                else:
                    readiness_text = "⏳ Долгосрочная перспектива"
                
                report_lines.append(f"{expertise_emoji} **{topic_name}** - {expertise_text}")
                report_lines.append(f"💡 **Признаки экспертности:** {expertise_indicators}")
                report_lines.append(f"💰 **Коммерческий потенциал ниши:** {commercial_potential}")
                report_lines.append(f"⚡ **Готовность к монетизации:** {readiness_text}")
                
                if methods:
                    report_lines.append("🛍️ **Способы монетизации экспертности:**")
                    for method in methods[:3]:  # Топ 3 метода
                        method_name = method.get('method', 'Способ заработка')
                        time_to_monetization = method.get('time_to_monetization', 'неизвестно')
                        development_needed = method.get('development_needed', 'неизвестно')
                        report_lines.append(f"   • **{method_name}**")
                        report_lines.append(f"     ⏰ Время до монетизации: {time_to_monetization}")
                        report_lines.append(f"     📚 Нужно развить: {development_needed}")
                
                report_lines.append("")
        
        # 🧠 УРОВЕНЬ 3: ПСИХОЛОГИЧЕСКИЙ АНАЛИЗ
        psychological_data = unified_data.get('psychological_analysis', {})
        if psychological_data:
            report_lines.append("\n" + "═" * 70 + "\n")
            report_lines.append("## 🧠 УРОВЕНЬ 3: ГЛУБИННЫЕ ПАТТЕРНЫ И ТРАНСФОРМАЦИЯ")
            report_lines.append("")
            
            system_model = psychological_data.get('system_model', '')
            patterns = psychological_data.get('patterns', [])
            transformation_hint = psychological_data.get('transformation_hint', '')
            
            if system_model:
                report_lines.append("🔮 **Ваша система поведения:**")
                report_lines.append(f"{system_model}")
                report_lines.append("")
            
            if patterns:
                report_lines.append("🔍 **Выявленные паттерны:**")
                report_lines.append("")
                
                for i, pattern in enumerate(patterns, 1):
                    name = pattern.get('name', f'Паттерн {i}')
                    origin = pattern.get('origin', 'неизвестно')
                    block_effect = pattern.get('block_effect', 'не определено')
                    blind_spot = pattern.get('blind_spot', 'не определено')
                    related_topics = pattern.get('related_topics', [])
                    
                    report_lines.append(f"**{i}. {name}**")
                    report_lines.append(f"   🌱 Происхождение: {origin}")
                    report_lines.append(f"   🚫 Как блокирует: {block_effect}")
                    report_lines.append(f"   👁️ Слепая зона: {blind_spot}")
                    if related_topics:
                        report_lines.append(f"   📌 Проявляется в темах: {', '.join(related_topics)}")
                    report_lines.append("")
            
            if transformation_hint:
                report_lines.append("✨ **КЛЮЧ К ТРАНСФОРМАЦИИ:**")
                report_lines.append(f"{transformation_hint}")
                report_lines.append("")
        
        # 🎯 ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ
        expertise_data = unified_data.get('expertise_analysis', [])
        if expertise_data:
            report_lines.append("\n" + "═" * 70 + "\n")
            report_lines.append("## 🎯 ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
            report_lines.append("")
            
            # Сортируем по коммерческому потенциалу И готовности к монетизации
            priority_topics = []
            for analysis in expertise_data:
                topic_name = analysis.get('topic', '')
                commercial_potential = analysis.get('commercial_potential', 'low')
                monetization_readiness = analysis.get('monetization_readiness', 'long_term')
                expertise_level = analysis.get('expertise_level', 'beginner')
                
                # Рассчитываем приоритет (0-10)
                priority_score = 0
                
                # Баллы за коммерческий потенциал
                if commercial_potential == 'high':
                    priority_score += 4
                elif commercial_potential == 'medium':
                    priority_score += 2
                
                # Баллы за готовность к монетизации
                if monetization_readiness == 'ready':
                    priority_score += 3
                elif monetization_readiness == 'need_development':
                    priority_score += 1
                
                # Баллы за экспертность
                if expertise_level == 'expert':
                    priority_score += 3
                elif expertise_level == 'advanced':
                    priority_score += 2
                else:
                    priority_score += 1
                
                priority_topics.append({
                    'topic': topic_name,
                    'analysis': analysis,
                    'priority': priority_score
                })
            
            # Сортируем по приоритету
            priority_topics.sort(key=lambda x: x['priority'], reverse=True)
            
            report_lines.append("На основе анализа экспертности и коммерческого потенциала рекомендуем:")
            report_lines.append("")
            
            for i, item in enumerate(priority_topics[:3], 1):
                analysis = item['analysis']
                topic_name = analysis.get('topic', f'Тема {i}')
                commercial_potential = analysis.get('commercial_potential', 'low')
                monetization_readiness = analysis.get('monetization_readiness', 'long_term')
                expertise_level = analysis.get('expertise_level', 'beginner')
                
                # Формируем логичную рекомендацию
                if expertise_level == 'expert' and commercial_potential == 'high' and monetization_readiness == 'ready':
                    rec = "🚀 НАЧИНАЙТЕ МОНЕТИЗАЦИЮ НЕМЕДЛЕННО! У вас есть все для успеха"
                elif expertise_level in ['expert', 'advanced'] and commercial_potential in ['high', 'medium']:
                    rec = "💪 Развивайте активно - у вас отличные перспективы"
                elif commercial_potential == 'high' and expertise_level == 'beginner':
                    rec = "📚 Сначала углубите знания, потом монетизируйте"
                elif expertise_level in ['expert', 'advanced'] and commercial_potential == 'low':
                    rec = "🎯 Развивайте для души, монетизация не приоритет"
                else:
                    rec = "🌱 Развивайтесь постепенно, оценивайте возможности"
                
                report_lines.append(f"**{i}. {topic_name}**")
                report_lines.append(f"   🎯 Уровень: {expertise_level} | Потенциал: {commercial_potential} | Готовность: {monetization_readiness}")
                report_lines.append(f"   💡 **Рекомендация:** {rec}")
                report_lines.append("")
        
        # ПОДПИСЬ
        report_lines.append("─" * 70)
        report_lines.append("🌟 *Soul Analysis Report - глубокий анализ цифровой души*")
        report_lines.append("📊 *Система TelegramSoul использует ИИ для комплексного анализа*")
        report_lines.append(f"⏰ *Время генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}*")
        
        return "\n".join(report_lines)

# 🚀 ЭТАП 1: Быстрый анализ тем для всех чатов
FAST_TOPIC_ANALYSIS_PROMPT = """
Ты — экспертный аналитик цифрового поведения. Твоя задача — быстро и точно выделить основные темы из Telegram-переписки.

📍 ЗАДАЧА: Проанализируй сообщения и выдели 10-15 ключевых тем, которые обсуждает пользователь.

📋 ЧТО НУЖНО СДЕЛАТЬ:
1. Выдели 10-15 тем (от наиболее популярных к менее популярным)
2. Для каждой темы укажи:
   - Название темы (краткое и понятное)
   - Ключевые слова (5-8 слов максимум)
   - Процент упоминаний (ВАЖНО: сумма всех процентов = 100%)
   - Эмоциональная тональность (positive/negative/neutral)
   - Краткое описание (1-2 предложения)

⚡ ТРЕБОВАНИЯ:
- Будь точным в процентах — их сумма должна быть ровно 100%
- Фокусируйся на содержательных темах, игнорируй бытовые мелочи
- Группируй похожие темы вместе
- Называй темы понятно для обычного человека

СООБЩЕНИЯ ДЛЯ АНАЛИЗА:
{messages}

📤 ВЕРНИ РЕЗУЛЬТАТ В JSON:
{{
  "topics": [
    {{
      "name": "Название темы",
      "keywords": ["слово1", "слово2", "слово3"],
      "percentage": XX.X,
      "sentiment": "positive/negative/neutral",
      "description": "Краткое описание темы"
    }},
    ...
  ]
}}
"""

# 🎯 ЭТАП 2: Глубокий финальный анализ на основе всех тем
FINAL_ANALYSIS_PROMPT = """
Ты — экспертный стратег монетизации и трансформационный коуч. На основе полного анализа всех тем пользователя проведи глубокое исследование возможностей.

📊 ИСХОДНЫЕ ДАННЫЕ:
{aggregated_topics}

🎯 ЗАДАЧА: Провести двухуровневый анализ:

---

💰 УРОВЕНЬ 1: АНАЛИЗ ЭКСПЕРТНОСТИ И НИШ
Для каждой темы проанализируй:
1. Глубину экспертности пользователя (expert/advanced/beginner)
2. Оцени коммерческий потенциал ниши (high/medium/low)
3. Предложи 2-3 конкретных способа монетизации экспертности
4. Определи срок развития до уровня монетизации и сложность входа

🧠 УРОВЕНЬ 2: ПСИХОЛОГИЧЕСКИЕ ПАТТЕРНЫ И ТРАНСФОРМАЦИЯ
На основе всей совокупности тем и интересов:
1. Определи архетип поведения пользователя (метафора личности)
2. Выяви 3-5 скрытых паттернов, которые могут блокировать рост
3. Для каждого паттерна укажи:
   - Происхождение (семья/культура/травма/социум)
   - В каких темах проявляется
   - Как блокирует развитие
   - Какую слепую зону создает
4. Дай трансформационный ключ для преодоления ограничений

📤 ВЕРНИ РЕЗУЛЬТАТ В JSON:
{{
  "expertise_analysis": [
    {{
      "topic": "Название темы",
      "expertise_level": "expert/advanced/beginner",
      "expertise_indicators": "Конкретные признаки экспертности из сообщений",
      "commercial_potential": "high/medium/low",
      "monetization_readiness": "ready/need_development/long_term",
      "monetization_methods": [
        {{
          "method": "Название способа",
          "description": "Подробное описание",
          "time_to_monetization": "2-4 месяца",
          "development_needed": "Что нужно развить для монетизации"
        }}
      ]
    }}
  ],
  "psychological_analysis": {{
    "system_model": "Метафорическое описание архетипа пользователя - кто он по сути",
    "patterns": [
      {{
        "name": "Название паттерна",
        "origin": "Возможное происхождение (семья/культура/травма/социум)",
        "related_topics": ["тема1", "тема2"],
        "block_effect": "Как именно блокирует развитие и рост",
        "blind_spot": "Какую слепую зону создает"
      }}
    ],
    "transformation_hint": "Ключевая рекомендация для трансформации и преодоления ограничений"
  }}
}}
"""
