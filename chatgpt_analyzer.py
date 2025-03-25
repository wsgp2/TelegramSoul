#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ChatGPT Telegram Chat Analyzer

Этот скрипт использует OpenAI API (gpt-4o-mini) для глубокого анализа сообщений из Telegram чатов,
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
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import difflib
from wordcloud import WordCloud
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
    Класс для анализа Telegram-чатов с использованием ChatGPT (gpt-4o-mini)
    """
    
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Инициализация анализатора
        
        Args:
            api_key (str): API ключ OpenAI. Если None, пытается использовать OPENAI_API_KEY из окружения
            model (str): Модель OpenAI для использования
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("API ключ OpenAI не указан. Установите переменную окружения OPENAI_API_KEY или передайте ключ при создании экземпляра")
        
        self.model = model
        self.messages_dir = MESSAGES_DIR
        self.output_dir = OUTPUT_DIR
        self.visualization_dir = VISUALIZATION_DIR
        self.client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"Инициализирован ChatGPT-анализатор с моделью {model}")
        
    async def load_messages_from_dir(self, directory=None) -> List[Dict]:
        """
        Загружает сообщения из JSON файлов в указанной директории
        
        Args:
            directory (str, optional): Директория с файлами сообщений. По умолчанию self.messages_dir
            
        Returns:
            List[Dict]: Список сообщений
        """
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
        Вызывает OpenAI API с указанными сообщениями
        
        Args:
            messages (List[Dict]): Сообщения для API в формате [{"role": "...", "content": "..."}]
            temperature (float): Параметр temperature для генерации
            
        Returns:
            Dict: Ответ от API
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
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
            
    async def analyze_topics(self, text_messages: List[str], max_tokens_per_chunk: int = 8000):
        """
        Анализирует темы в сообщениях с использованием ChatGPT
        
        Args:
            text_messages (List[str]): Список текстовых сообщений для анализа
            max_tokens_per_chunk (int): Максимальное количество токенов в одном запросе к API
            
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
        
        # Анализируем каждую часть
        all_topics = []
        for i, chunk in enumerate(chunk_messages):
            logger.info(f"Анализируем часть {i+1} из {len(chunk_messages)}")
            messages = [
                {"role": "system", "content": "Вы - эксперт по тематическому анализу и выявлению трендов в данных."},
                {"role": "user", "content": TOPIC_ANALYSIS_PROMPT.format(messages=chunk)}
            ]
            
            try:
                response = await self.call_openai_api(messages, temperature=0.2)
                content = response['choices'][0]['message']['content']
                # Более надежный парсинг JSON
                chunk_topics = self.extract_json_from_text(content)
                all_topics.extend(chunk_topics.get("topics", []))
            except Exception as e:
                logger.error(f"Ошибка при анализе части {i+1}: {e}")
                continue
                
        # Объединяем результаты и агрегируем схожие темы
        aggregated_topics = self._aggregate_similar_topics(all_topics)
        
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
        
        # Нормализуем проценты, чтобы сумма была 100%
        total_percentage = sum(topic.get('percentage', 0) for topic in result)
        if total_percentage > 0:
            for topic in result:
                topic['percentage'] = round((topic.get('percentage', 0) / total_percentage) * 100, 1)
        
        return result[:7]  # Возвращаем только топ-7 тем
        
    async def develop_monetization_strategies(self, topics: Dict):
        """
        Разрабатывает стратегии монетизации на основе выявленных тем
        
        Args:
            topics (Dict): JSON объект с результатами анализа тем
            
        Returns:
            Dict: Результаты анализа возможностей монетизации
        """
        logger.info("Начинаем разработку стратегий монетизации...")
        
        if not topics or not topics.get('topics'):
            logger.warning("Нет тем для анализа монетизации")
            return {"monetization_strategies": []}
            
        topics_json = json.dumps(topics, ensure_ascii=False, indent=2)
        prompt = MONETIZATION_ANALYSIS_PROMPT.format(topics_json=topics_json)
        
        messages_for_api = [
            {"role": "system", "content": "Вы - опытный бизнес-консультант, эксперт по разработке стратегий монетизации и построению бизнес-моделей."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.call_openai_api(messages_for_api, temperature=0.4)
            content = response['choices'][0]['message']['content']
            # Используем extract_json_from_text вместо прямого json.loads
            result = self.extract_json_from_text(content)
            
            logger.info(f"Разработано {len(result.get('monetization_strategies', []))} стратегий монетизации")
            return result
        except Exception as e:
            logger.error(f"Ошибка при разработке стратегий монетизации: {e}")
            return {"monetization_strategies": []}
            
    async def create_business_plan(self, topics: Dict, monetization: Dict):
        """
        Создает детальный бизнес-план на основе результатов анализа
        
        Args:
            topics (Dict): JSON объект с результатами анализа тем
            monetization (Dict): JSON объект с результатами анализа монетизации
            
        Returns:
            Dict: Детальный бизнес-план
        """
        logger.info("Начинаем создание детального бизнес-плана...")
        
        if not topics or not monetization:
            logger.warning("Недостаточно данных для создания бизнес-плана")
            return {"business_plan": {}}
            
        topics_json = json.dumps(topics, ensure_ascii=False, indent=2)
        monetization_json = json.dumps(monetization, ensure_ascii=False, indent=2)
        
        prompt = BUSINESS_PLAN_PROMPT.format(
            topics_json=topics_json,
            monetization_json=monetization_json
        )
        
        messages_for_api = [
            {"role": "system", "content": "Вы - опытный бизнес-стратег и консультант по запуску стартапов с обширным опытом создания успешных бизнес-планов."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.call_openai_api(messages_for_api, temperature=0.3)
            content = response['choices'][0]['message']['content']
            # Используем extract_json_from_text вместо прямого json.loads
            result = self.extract_json_from_text(content)
            
            logger.info("Бизнес-план успешно создан")
            return result
        except Exception as e:
            logger.error(f"Ошибка при создании бизнес-плана: {e}")
            return {"business_plan": {}}
        
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
    
    def visualize_topics(self, topics: Dict, output_dir: str = None) -> List[str]:
        """
        Создает визуализации для результатов анализа тем
        
        Args:
            topics (Dict): Результаты анализа тем
            output_dir (str, optional): Директория для сохранения визуализаций
            
        Returns:
            List[str]: Список путей к сохраненным визуализациям
        """
        output_dir = output_dir or self.visualization_dir
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        if not topics or not topics.get('topics'):
            logger.warning("Нет данных для создания визуализаций тем")
            return saved_files
        
        topic_data = topics.get('topics', [])
        
        try:
            # 1. Создаем круговую диаграмму распределения тем
            plt.figure(figsize=(12, 8))
            labels = [topic['name'] for topic in topic_data]
            sizes = [topic['percentage'] for topic in topic_data]
            colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
            
            # Добавляем отступ для выделения самой большой темы
            explode = [0.1 if i == np.argmax(sizes) else 0 for i in range(len(sizes))]
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            plt.axis('equal')  # Делаем круг равным
            plt.title('Распределение тем в чате', fontsize=16, pad=20)
            
            pie_chart_path = os.path.join(output_dir, f"topics_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(pie_chart_path)
            logger.info(f"Создана круговая диаграмма распределения тем: {pie_chart_path}")
            
            # 2. Создаем облако ключевых слов
            all_keywords = []
            keyword_weights = {}
            
            for topic in topic_data:
                for keyword in topic['keywords']:
                    if keyword in keyword_weights:
                        keyword_weights[keyword] += topic['percentage']
                    else:
                        keyword_weights[keyword] = topic['percentage']
                    all_keywords.append(keyword)
            
            if all_keywords:
                plt.figure(figsize=(14, 10))
                wordcloud = WordCloud(
                    width=1000, height=600,
                    background_color='white',
                    max_words=100,
                    colormap='viridis',
                    collocations=False
                ).generate_from_frequencies(keyword_weights)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title('Облако ключевых слов из всех тем', fontsize=16)
                
                wordcloud_path = os.path.join(output_dir, f"keywords_wordcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.close()
                saved_files.append(wordcloud_path)
                logger.info(f"Создано облако ключевых слов: {wordcloud_path}")
            
            # 3. Создаем столбчатую диаграмму распределения сентиментов
            sentiment_counts = Counter([topic.get('sentiment', 'neutral') for topic in topic_data])
            sentiment_df = pd.DataFrame({
                'Сентимент': list(sentiment_counts.keys()),
                'Количество тем': list(sentiment_counts.values())
            })
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Сентимент', y='Количество тем', data=sentiment_df, palette=['green', 'gray', 'red'])
            plt.title('Распределение сентиментов по темам', fontsize=16)
            plt.xlabel('Сентимент', fontsize=12)
            plt.ylabel('Количество тем', fontsize=12)
            
            sentiment_path = os.path.join(output_dir, f"sentiment_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(sentiment_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(sentiment_path)
            logger.info(f"Создана диаграмма распределения сентиментов: {sentiment_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при создании визуализаций: {e}")
        
        return saved_files
        
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
