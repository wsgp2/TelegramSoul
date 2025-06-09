#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Анализатор данных чатов для выявления возможностей монетизации.
Использует данные из ChromaDB с векторизованными сообщениями для:
- обнаружения часто обсуждаемых тем
- выявления интересов и потребностей пользователей
- анализа эмоциональной окраски сообщений
- определения временных паттернов общения
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from tqdm import tqdm
# Добавляем новые импорты для сентимент-анализа
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка переменных окружения
load_dotenv()

# Константы
DB_DIR = os.getenv("DB_DIR", os.path.join(os.path.dirname(__file__), "data", "db"))
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './data/messages')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatMonetizationAnalyzer:
    """
    Анализатор данных чатов для выявления возможностей монетизации.
    Использует векторизованные сообщения и диалоги для анализа тематик,
    интересов пользователей и потенциальных направлений монетизации.
    """
    
    def __init__(self, db_path=DB_DIR, model_name='all-MiniLM-L6-v2'):
        """
        Инициализация анализатора данных чатов.
        
        Args:
            db_path (str): Путь к базе данных с векторами сообщений
            model_name (str): Название модели SentenceTransformer для эмбеддингов
        """
        logger.info(f"Инициализация анализатора данных чатов с базой данных: {db_path}")
        
        # Инициализация модели для эмбеддингов
        self.model = SentenceTransformer(model_name)
        
        # Подключение к базе данных
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Подключение к коллекциям
        try:
            self.message_collection = self.client.get_collection("telegram_messages")
            logger.info("Подключение к коллекции сообщений успешно")
            self.has_messages = True
        except Exception as e:
            logger.warning(f"Коллекция сообщений не найдена: {e}")
            self.has_messages = False
            
        try:
            self.conversation_collection = self.client.get_collection("telegram_conversations")
            logger.info("Подключение к коллекции диалогов успешно")
            self.has_conversations = True
        except Exception as e:
            logger.warning(f"Коллекция диалогов не найдена: {e}")
            self.has_conversations = False
            
        # Проверка наличия данных
        if not (self.has_messages or self.has_conversations):
            raise ValueError("Ни одна коллекция не найдена. Сначала необходимо собрать и векторизовать данные.")
    
    def _load_all_messages(self, limit=None):
        """
        Загрузка всех сообщений из базы данных.
        
        Args:
            limit (int, optional): Ограничение на количество загружаемых сообщений
            
        Returns:
            pd.DataFrame: DataFrame с сообщениями и их метаданными
        """
        if not self.has_messages:
            logger.warning("Коллекция сообщений не найдена")
            return pd.DataFrame()
        
        logger.info("Загрузка всех сообщений из базы данных...")
        
        try:
            # Получение всех документов из коллекции
            results = self.message_collection.get(
                limit=limit,
                include=["metadatas", "documents", "embeddings"]
            )
            
            # Создание DataFrame
            messages_data = []
            
            if results and 'metadatas' in results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    # Получаем текст сообщения
                    text = ""
                    if 'documents' in results and i < len(results['documents']):
                        text = results['documents'][i]
                    
                    # Получаем эмбеддинг
                    embedding = []
                    if 'embeddings' in results and i < len(results['embeddings']):
                        embedding = results['embeddings'][i]
                    
                    # Формируем запись
                    message_record = {
                        'text': text,
                        'embedding': embedding,
                        'message_id': metadata.get('message_id', ''),
                        'chat_id': metadata.get('chat_id', ''),
                        'from_id': metadata.get('from_id', ''),
                        'to_id': metadata.get('to_id', ''),
                        'date': metadata.get('date', ''),
                        'is_reply': metadata.get('is_reply', False),
                        'out': metadata.get('out', False),
                        'chat_type': metadata.get('chat_type', 'unknown')
                    }
                    
                    messages_data.append(message_record)
            
            # Создаем DataFrame
            df = pd.DataFrame(messages_data)
            logger.info(f"Загружено {len(df)} сообщений")
            
            # Преобразуем дату в datetime объект
            if 'date' in df.columns and not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке сообщений: {e}")
            return pd.DataFrame()
    
    def identify_topics(self, n_clusters=10, min_cluster_size=5):
        """
        Идентификация основных тем в сообщениях с использованием кластеризации.
        
        Args:
            n_clusters (int): Количество кластеров для выделения
            min_cluster_size (int): Минимальный размер кластера для рассмотрения
            
        Returns:
            dict: Словарь с информацией о темах/кластерах
        """
        logger.info(f"Идентификация основных тем в сообщениях (кластеров: {n_clusters})...")
        
        # Загрузка сообщений
        df = self._load_all_messages()
        
        if df.empty:
            logger.warning("Нет данных для анализа тем")
            return {}
        
        # Фильтрация сообщений (только с непустым текстом и эмбеддингом)
        df = df[df['text'].str.len() > 5].reset_index(drop=True)
        
        if df.empty:
            logger.warning("После фильтрации не осталось сообщений для анализа")
            return {}
        
        try:
            # Подготовка эмбеддингов для кластеризации
            embeddings = np.array(df['embedding'].tolist())
            
            # Кластеризация с помощью K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(embeddings)
            
            # Анализ кластеров
            clusters_info = {}
            for cluster_id in range(n_clusters):
                cluster_messages = df[df['cluster'] == cluster_id]
                
                if len(cluster_messages) >= min_cluster_size:
                    # Определение самых характерных сообщений для кластера
                    # (ближайших к центру кластера)
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    
                    # Вычисление расстояния от каждого сообщения до центра кластера
                    distances = np.linalg.norm(embeddings[df['cluster'] == cluster_id] - cluster_center, axis=1)
                    
                    # Выбор топ-5 сообщений, ближайших к центру
                    top_indices = np.argsort(distances)[:5]
                    representative_messages = cluster_messages.iloc[top_indices]['text'].tolist()
                    
                    # Выделение ключевых слов (простой подход - просто самые частые слова)
                    all_words = ' '.join(cluster_messages['text'].tolist()).lower()
                    word_counter = Counter(word for word in all_words.split() if len(word) > 3)
                    top_words = [word for word, count in word_counter.most_common(10)]
                    
                    # Запись информации о кластере
                    clusters_info[cluster_id] = {
                        'size': len(cluster_messages),
                        'percentage': round(len(cluster_messages) / len(df) * 100, 2),
                        'top_keywords': top_words,
                        'representative_messages': representative_messages
                    }
            
            logger.info(f"Идентифицировано {len(clusters_info)} значимых тем")
            return clusters_info
            
        except Exception as e:
            logger.error(f"Ошибка при идентификации тем: {e}")
            return {}

    def analyze_sentiment(self, messages_df=None):
        """
        Анализ эмоциональной окраски сообщений с использованием TextBlob.
        
        Args:
            messages_df (pd.DataFrame, optional): DataFrame с сообщениями или None, чтобы загрузить все сообщения
            
        Returns:
            dict: Результаты анализа эмоциональной окраски сообщений
        """
        logger.info("Проведение анализа эмоциональной окраски сообщений...")
        
        if messages_df is None:
            messages_df = self._load_all_messages()
            
        if messages_df.empty:
            logger.warning("Нет данных для анализа эмоциональной окраски")
            return {}
        
        # Фильтрация сообщений (только с непустым текстом)
        messages_df = messages_df[messages_df['text'].str.len() > 5].reset_index(drop=True)
        
        if messages_df.empty:
            logger.warning("После фильтрации не осталось сообщений для анализа эмоций")
            return {}
        
        try:
            # Вычисление полярности (от -1 до 1) и субъективности (от 0 до 1) для каждого сообщения
            sentiments = []
            for text in tqdm(messages_df['text'], desc="Анализ эмоций"):
                blob = TextBlob(text)
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            # Добавляем результаты в DataFrame
            sentiment_df = pd.DataFrame(sentiments)
            messages_df = pd.concat([messages_df, sentiment_df], axis=1)
            
            # Классификация по эмоциональной окраске
            messages_df['sentiment_category'] = pd.cut(
                messages_df['polarity'], 
                bins=[-1.0, -0.3, 0.3, 1.0], 
                labels=['negative', 'neutral', 'positive']
            )
            
            # Анализ распределения эмоций
            sentiment_distribution = messages_df['sentiment_category'].value_counts().to_dict()
            total = sum(sentiment_distribution.values())
            sentiment_percentages = {k: round(v/total*100, 2) for k, v in sentiment_distribution.items()}
            
            # Нахождение наиболее позитивных и негативных сообщений
            most_positive = messages_df.nlargest(5, 'polarity')
            most_negative = messages_df.nsmallest(5, 'polarity')
            
            # Анализ по темам с сильной эмоциональной окраской
            # Группируем сообщения по кластерам, если они были определены
            emotion_by_topic = {}
            if 'cluster' in messages_df.columns:
                for cluster in messages_df['cluster'].unique():
                    cluster_msgs = messages_df[messages_df['cluster'] == cluster]
                    emotion_by_topic[int(cluster)] = {
                        'avg_polarity': round(cluster_msgs['polarity'].mean(), 2),
                        'sentiment_distribution': cluster_msgs['sentiment_category'].value_counts().to_dict(),
                        'topic_size': len(cluster_msgs)
                    }
            
            # Формирование результатов
            sentiment_results = {
                'overall_sentiment': {
                    'average_polarity': round(messages_df['polarity'].mean(), 2),
                    'average_subjectivity': round(messages_df['subjectivity'].mean(), 2),
                    'distribution': sentiment_distribution,
                    'percentages': sentiment_percentages
                },
                'most_positive_examples': most_positive[['text', 'polarity']].to_dict('records'),
                'most_negative_examples': most_negative[['text', 'polarity']].to_dict('records'),
                'emotion_by_topic': emotion_by_topic
            }
            
            logger.info(f"Анализ эмоциональной окраски завершен. Средняя полярность: {sentiment_results['overall_sentiment']['average_polarity']}")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Ошибка при анализе эмоциональной окраски: {e}")
            return {}

    def analyze_time_patterns(self, messages_df=None):
        """
        Анализ временных паттернов в сообщениях - когда обсуждаются темы, в какое время дня/недели и т.д.
        
        Args:
            messages_df (pd.DataFrame, optional): DataFrame с сообщениями или None, чтобы загрузить все сообщения
            
        Returns:
            dict: Результаты анализа временных паттернов
        """
        logger.info("Проведение анализа временных паттернов в сообщениях...")
        
        if messages_df is None:
            messages_df = self._load_all_messages()
            
        if messages_df.empty:
            logger.warning("Нет данных для анализа временных паттернов")
            return {}
        
        # Проверка наличия даты
        if 'date' not in messages_df.columns:
            logger.warning("В данных отсутствует информация о дате сообщений")
            return {}
        
        try:
            # Преобразуем дату в datetime, если это еще не сделано
            if not pd.api.types.is_datetime64_any_dtype(messages_df['date']):
                messages_df['date'] = pd.to_datetime(messages_df['date'], errors='coerce')
            
            # Отбрасываем строки с недопустимыми датами
            messages_df = messages_df.dropna(subset=['date']).reset_index(drop=True)
            
            if messages_df.empty:
                logger.warning("После очистки дат не осталось сообщений для анализа")
                return {}
            
            # Извлечение компонентов даты/времени
            messages_df['hour'] = messages_df['date'].dt.hour
            messages_df['day_of_week'] = messages_df['date'].dt.day_name()
            messages_df['month'] = messages_df['date'].dt.month_name()
            messages_df['year'] = messages_df['date'].dt.year
            
            # Анализ активности по времени суток
            hourly_activity = messages_df['hour'].value_counts().sort_index().to_dict()
            
            # Анализ активности по дням недели
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_activity = messages_df['day_of_week'].value_counts().reindex(day_order).fillna(0).to_dict()
            
            # Анализ по месяцам
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
            monthly_activity = messages_df['month'].value_counts().reindex(month_order).fillna(0).to_dict()
            
            # Анализ по годам
            yearly_activity = messages_df['year'].value_counts().sort_index().to_dict()
            
            # Анализ тем по времени суток (если есть кластеры)
            topics_by_time = {}
            if 'cluster' in messages_df.columns:
                for hour in range(24):
                    hour_msgs = messages_df[messages_df['hour'] == hour]
                    if not hour_msgs.empty:
                        topics_by_time[hour] = hour_msgs['cluster'].value_counts().to_dict()
            
            # Анализ эмоций по времени суток (если есть данные о сентименте)
            sentiment_by_time = {}
            if 'sentiment_category' in messages_df.columns:
                for hour in range(24):
                    hour_msgs = messages_df[messages_df['hour'] == hour]
                    if not hour_msgs.empty:
                        sentiment_by_time[hour] = hour_msgs['sentiment_category'].value_counts().to_dict()
            
            # Формирование результатов
            time_patterns = {
                'hourly_activity': hourly_activity,
                'daily_activity': weekly_activity,
                'monthly_activity': monthly_activity,
                'yearly_activity': yearly_activity,
                'topics_by_time': topics_by_time,
                'sentiment_by_time': sentiment_by_time
            }
            
            # Определение пиковых часов активности
            peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            time_patterns['peak_hours'] = [{'hour': h, 'count': c} for h, c in peak_hours]
            
            # Определение самых активных дней недели
            peak_days = sorted(weekly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            time_patterns['peak_days'] = [{'day': d, 'count': c} for d, c in peak_days]
            
            logger.info("Анализ временных паттернов завершен")
            return time_patterns
            
        except Exception as e:
            logger.error(f"Ошибка при анализе временных паттернов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def identify_monetization_opportunities(self, topics=None, sentiment_results=None, time_patterns=None):
        """
        Выявление потенциальных возможностей монетизации на основе анализа данных.
        
        Args:
            topics (dict, optional): Результаты анализа тем
            sentiment_results (dict, optional): Результаты анализа эмоциональной окраски
            time_patterns (dict, optional): Результаты анализа временных паттернов
            
        Returns:
            dict: Выявленные возможности монетизации по категориям
        """
        logger.info("Выявление потенциальных возможностей монетизации...")
        
        # Если данные не переданы, выполняем анализ
        if topics is None:
            topics = self.identify_topics()
        
        if sentiment_results is None:
            sentiment_results = self.analyze_sentiment()
            
        if time_patterns is None:
            time_patterns = self.analyze_time_patterns()
        
        # Определение популярных тем с позитивным сентиментом
        positive_topics = []
        if topics and sentiment_results and 'emotion_by_topic' in sentiment_results:
            for cluster_id, info in sentiment_results['emotion_by_topic'].items():
                if cluster_id in topics and info.get('avg_polarity', 0) > 0.2:
                    positive_topics.append({
                        'topic_id': cluster_id,
                        'keywords': topics[cluster_id]['top_keywords'],
                        'size': topics[cluster_id]['size'],
                        'polarity': info['avg_polarity']
                    })
        
        # Сортировка тем по размеру (популярности)
        sorted_topics = sorted(topics.items(), key=lambda x: x[1]['size'], reverse=True) if topics else []
        top_topics = [{'topic_id': t[0], 'size': t[1]['size'], 'keywords': t[1]['top_keywords']} for t in sorted_topics[:5]]
        
        # Определение временных интервалов с высокой активностью
        active_times = []
        if time_patterns and 'peak_hours' in time_patterns:
            active_times = time_patterns['peak_hours']
        
        # Формирование рекомендаций по категориям монетизации
        monetization_opportunities = {
            'info_products': {
                'potential': 'high' if len(top_topics) >= 3 else 'medium',
                'ideas': [
                    f"Электронная книга или курс по теме: {', '.join(topic['keywords'][:3])}" 
                    for topic in top_topics[:3] if topic['size'] > 10
                ],
                'reasoning': "Популярные темы с большим количеством сообщений указывают на интерес аудитории к этим вопросам."
            },
            'consulting_services': {
                'potential': 'high' if positive_topics else 'medium',
                'ideas': [
                    f"Консультации по теме: {', '.join(topic['keywords'][:3])}" 
                    for topic in positive_topics[:3]
                ],
                'reasoning': "Темы с позитивным сентиментом могут указывать на области, где ваша экспертиза ценится."
            },
            'affiliate_marketing': {
                'potential': 'medium',
                'ideas': [
                    f"Партнерские программы в категории: {', '.join(topic['keywords'][:3])}" 
                    for topic in top_topics[:2]
                ],
                'reasoning': "Часто обсуждаемые темы могут указывать на потребительский интерес к связанным продуктам."
            },
            'premium_bot': {
                'potential': 'high' if time_patterns and active_times else 'medium',
                'ideas': [
                    f"Премиум-версия бота с расширенными ответами на тему: {', '.join(topic['keywords'][:3])}"
                    for topic in top_topics[:1]
                ],
                'timing': [
                    f"Оптимальное время для маркетинга: {hour['hour']}:00" for hour in active_times
                ] if active_times else [],
                'reasoning': "Существующая база пользователей может быть конвертирована в платных подписчиков премиум-версии бота."
            }
        }
        
        logger.info("Анализ возможностей монетизации завершен")
        return monetization_opportunities

def main():
    """
    Основная функция для запуска анализа данных чатов.
    """
    try:
        # Инициализация анализатора
        analyzer = ChatMonetizationAnalyzer()
        
        # Идентификация основных тем
        topics = analyzer.identify_topics(n_clusters=15, min_cluster_size=3)
        
        # Анализ эмоциональной окраски сообщений
        sentiment_results = analyzer.analyze_sentiment()
        
        # Анализ временных паттернов
        time_patterns = analyzer.analyze_time_patterns()
        
        # Выявление возможностей монетизации
        monetization_opportunities = analyzer.identify_monetization_opportunities(
            topics=topics,
            sentiment_results=sentiment_results,
            time_patterns=time_patterns
        )
        
        # Создание директории для отчетов, если её нет
        reports_dir = os.path.join(os.path.dirname(__file__), "data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Сохранение результатов анализа в JSON
        results = {
            'topics': topics,
            'sentiment': sentiment_results,
            'time_patterns': time_patterns,
            'monetization_opportunities': monetization_opportunities
        }
        
        with open(os.path.join(reports_dir, 'monetization_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Вывод результатов
        print("\n=== Основные темы в сообщениях ===\n")
        for cluster_id, info in topics.items():
            print(f"Тема #{cluster_id+1} ({info['size']} сообщений, {info['percentage']}%)")
            print(f"Ключевые слова: {', '.join(info['top_keywords'])}")
            print("Примеры сообщений:")
            for i, msg in enumerate(info['representative_messages'][:3]):
                print(f"  {i+1}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
            print()
            
        print("\n=== Эмоциональная окраска сообщений ===\n")
        if sentiment_results and 'overall_sentiment' in sentiment_results:
            overall = sentiment_results['overall_sentiment']
            print(f"Средняя эмоциональная окраска: {overall['average_polarity']} ({overall['average_polarity'] < 0 and 'негативная' or (overall['average_polarity'] > 0.2 and 'позитивная' or 'нейтральная')})")
            print(f"Распределение эмоций: {overall['percentages']}")
            print()
            
        print("\n=== Временные паттерны ===\n")
        if time_patterns and 'peak_hours' in time_patterns:
            print(f"Пиковые часы активности: {[h['hour'] for h in time_patterns['peak_hours']]}")
            print(f"Самые активные дни недели: {[d['day'] for d in time_patterns['peak_days']]}")
            print()
            
        print("\n=== Потенциальные возможности монетизации ===\n")
        for category, info in monetization_opportunities.items():
            print(f"{category.replace('_', ' ').title()} (потенциал: {info['potential']})")
            if 'ideas' in info and info['ideas']:
                print("Идеи:")
                for idea in info['ideas']:
                    print(f"  - {idea}")
            if 'timing' in info and info['timing']:
                print("Оптимальное время:")
                for timing in info['timing']:
                    print(f"  - {timing}")
            print(f"Обоснование: {info['reasoning']}")
            print()
            
        print(f"\nПолный отчет сохранен в: {os.path.join(reports_dir, 'monetization_analysis.json')}")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении анализа: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
