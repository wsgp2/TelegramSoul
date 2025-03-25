#!/usr/bin/env python3
import os
import json
import asyncio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import plotly.io as pio
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('topic_clustering')

# Загрузка переменных окружения
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Инициализация клиента OpenAI
client = OpenAI(api_key=API_KEY)

class TopicClusterAnalyzer:
    def __init__(self, data_dir="data/reports", output_dir="data/visualizations"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.ensure_dirs_exist()
        
    def ensure_dirs_exist(self):
        """Создание необходимых директорий"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_topics(self, filename="all_chats_topics_adapted.json"):
        """Загрузка всех тем из файла"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'topics' in data:
                    logger.info(f"Загружено {len(data['topics'])} тем из {data.get('total_chats', 0)} чатов")
                    return data
                else:
                    logger.error("В файле отсутствует ключ 'topics'")
                    return None
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {filepath}: {str(e)}")
            return None
    
    async def get_embeddings(self, topics):
        """Получение векторных представлений для тем с использованием OpenAI API"""
        try:
            # Создаем список названий тем
            topic_texts = [topic.get('name', '') for topic in topics]
            
            # Получаем эмбеддинги через OpenAI API
            embeddings = []
            
            # Обрабатываем темы в пакетах по 1000 (ограничение API)
            batch_size = 1000
            for i in range(0, len(topic_texts), batch_size):
                batch = topic_texts[i:i+batch_size]
                
                # Используем актуальную модель text-embedding-3-small
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                
                # Извлекаем эмбеддинги из ответа
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Получены эмбеддинги для пакета {i}-{i+len(batch)} из {len(topic_texts)}")
            
            logger.info(f"Получены эмбеддинги для всех {len(embeddings)} тем")
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддингов: {str(e)}")
            return None
    
    def cluster_topics(self, embeddings, topics, n_clusters=10):
        """Кластеризация тем на основе их векторных представлений"""
        try:
            # Применяем K-means для кластеризации
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Добавляем информацию о кластере к каждой теме
            for i, topic in enumerate(topics):
                topic['cluster'] = int(clusters[i])
            
            # Группируем темы по кластерам
            clustered_topics = {}
            for i, topic in enumerate(topics):
                cluster_id = topic['cluster']
                if cluster_id not in clustered_topics:
                    clustered_topics[cluster_id] = []
                clustered_topics[cluster_id].append(topic)
            
            # Сортируем кластеры по размеру (количеству тем)
            sorted_clusters = {k: v for k, v in sorted(
                clustered_topics.items(), 
                key=lambda item: len(item[1]), 
                reverse=True
            )}
            
            logger.info(f"Темы успешно кластеризованы на {n_clusters} групп")
            return sorted_clusters, topics
        except Exception as e:
            logger.error(f"Ошибка при кластеризации: {str(e)}")
            return None, topics
    
    async def assign_cluster_names(self, clustered_topics):
        """Присваивание осмысленных имен кластерам с помощью OpenAI API"""
        try:
            cluster_names = {}
            
            for cluster_id, topics in clustered_topics.items():
                # Берем названия всех тем в кластере
                topic_names = [topic.get('name', '') for topic in topics]
                topic_names_str = '\n'.join([f"- {name}" for name in topic_names[:20]])  # Ограничиваем до 20 тем
                
                # Запрос к OpenAI для определения общей тематики
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Ты эксперт по категоризации и классификации информации."},
                        {"role": "user", "content": f"На основе следующего списка тем, определи одно короткое название (3-4 слова максимум) для категории, которая объединяет их. Дай только название категории, без пояснений.\n\n{topic_names_str}"}
                    ]
                )
                
                # Получаем имя кластера из ответа
                cluster_name = response.choices[0].message.content.strip()
                cluster_names[cluster_id] = cluster_name
                logger.info(f"Кластер {cluster_id} ({len(topics)} тем) назван: {cluster_name}")
            
            return cluster_names
        except Exception as e:
            logger.error(f"Ошибка при назначении имен кластерам: {str(e)}")
            # Возвращаем базовые имена в случае ошибки
            return {cluster_id: f"Кластер {cluster_id}" for cluster_id in clustered_topics.keys()}
    
    def create_cluster_summary(self, clustered_topics, cluster_names):
        """Создание резюме по каждому кластеру"""
        summary = ["# Анализ основных тематических направлений\n"]
        
        for cluster_id, topics in clustered_topics.items():
            cluster_name = cluster_names.get(cluster_id, f"Кластер {cluster_id}")
            summary.append(f"## {cluster_name} ({len(topics)} тем)\n")
            
            # Сортируем темы внутри кластера по частоте и важности
            sorted_topics = sorted(
                topics, 
                key=lambda x: (x.get('frequency', 0) + x.get('importance', 0)), 
                reverse=True
            )
            
            # Добавляем 10 самых важных тем из кластера
            for i, topic in enumerate(sorted_topics[:10], 1):
                frequency = topic.get('frequency', 0)
                importance = topic.get('importance', 0)
                emotions = topic.get('emotions', {}).get('dominant', 'нейтральные')
                summary.append(f"{i}. **{topic.get('name', '')}** (Частота: {frequency}, Важность: {importance}, Эмоции: {emotions})")
            
            summary.append("\n")
        
        return "\n".join(summary)
    
    def visualize_topic_clusters(self, topics, embeddings, cluster_names):
        """Создание 2D визуализации кластеров тем"""
        try:
            from sklearn.decomposition import PCA
            
            # Используем PCA для снижения размерности до 2D
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Создаем DataFrame для Plotly
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'topic': [topic.get('name', '') for topic in topics],
                'cluster': [topic.get('cluster', 0) for topic in topics],
                'frequency': [topic.get('frequency', 0) for topic in topics],
                'importance': [topic.get('importance', 0) for topic in topics]
            })
            
            # Создаем новую колонку с названиями кластеров
            df['cluster_name'] = df['cluster'].apply(lambda x: cluster_names.get(x, f"Кластер {x}"))
            
            # Создаем bubble chart
            fig = px.scatter(
                df, x='x', y='y', color='cluster_name',
                size='frequency', hover_name='topic',
                hover_data=['importance', 'frequency'],
                title="Карта тематических интересов",
                size_max=40
            )
            
            # Настройка графика
            fig.update_layout(
                autosize=True, width=1200, height=800,
                xaxis_title="Первый компонент",
                yaxis_title="Второй компонент",
                legend_title="Тематическая категория"
            )
            
            # Сохраняем график
            output_path = os.path.join(self.output_dir, "topic_clusters_map.html")
            fig.write_html(output_path)
            
            logger.info(f"Визуализация кластеров тем сохранена: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при создании визуализации: {str(e)}")
            return None
    
    def create_topic_heatmap(self, topics, embeddings):
        """Создание тепловой карты семантических связей между темами"""
        try:
            # Вычисляем матрицу косинусного сходства
            similarity_matrix = cosine_similarity(embeddings)
            
            # Берем только топ-30 тем для читаемости
            sorted_topics = sorted(
                topics, 
                key=lambda x: (x.get('frequency', 0) + x.get('importance', 0)), 
                reverse=True
            )[:30]
            top_indices = [topics.index(topic) for topic in sorted_topics]
            
            # Фильтруем матрицу сходства
            filtered_matrix = similarity_matrix[np.ix_(top_indices, top_indices)]
            
            # Создаем подписи для осей
            topic_names = [topic.get('name', '') for topic in sorted_topics]
            
            # Создаем тепловую карту
            fig = go.Figure(data=go.Heatmap(
                z=filtered_matrix,
                x=topic_names,
                y=topic_names,
                colorscale='Viridis',
                zmin=0, zmax=1
            ))
            
            # Настройка графика
            fig.update_layout(
                title="Тепловая карта взаимосвязей между темами",
                autosize=True, width=1200, height=1000,
                xaxis=dict(tickangle=45),
                margin=dict(l=100, r=20, t=100, b=150)
            )
            
            # Сохраняем график
            output_path = os.path.join(self.output_dir, "topic_similarity_heatmap.html")
            fig.write_html(output_path)
            
            logger.info(f"Тепловая карта связей между темами сохранена: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Ошибка при создании тепловой карты: {str(e)}")
            return None

async def main():
    # Инициализация анализатора
    analyzer = TopicClusterAnalyzer()
    
    # Загрузка данных
    data = analyzer.load_topics()
    if not data or 'topics' not in data:
        logger.error("Не удалось загрузить данные о темах")
        return
    
    # Получаем все темы
    topics = data['topics']
    logger.info(f"Анализируем {len(topics)} тем из {data.get('total_chats', 0)} чатов")
    
    # Получаем эмбеддинги для тем
    embeddings = await analyzer.get_embeddings(topics)
    if embeddings is None:
        logger.error("Не удалось получить эмбеддинги для тем")
        return
    
    # Определяем оптимальное количество кластеров (можно настроить)
    n_clusters = min(max(10, len(topics) // 50), 20)  # От 10 до 20 кластеров
    
    # Кластеризация тем
    clustered_topics, updated_topics = analyzer.cluster_topics(embeddings, topics, n_clusters)
    if clustered_topics is None:
        logger.error("Не удалось выполнить кластеризацию тем")
        return
    
    # Присваиваем осмысленные имена кластерам
    cluster_names = await analyzer.assign_cluster_names(clustered_topics)
    
    # Создаем сводное резюме по кластерам
    summary = analyzer.create_cluster_summary(clustered_topics, cluster_names)
    with open(os.path.join(analyzer.data_dir, "topic_clusters_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Сохранено резюме по кластерам тем")
    
    # Создаем визуализацию кластеров
    vis_path = analyzer.visualize_topic_clusters(updated_topics, embeddings, cluster_names)
    
    # Создаем тепловую карту связей между темами
    heatmap_path = analyzer.create_topic_heatmap(topics, embeddings)
    
    # Сохраняем результаты кластеризации
    output = {
        "original_topics_count": len(topics),
        "clusters_count": len(clustered_topics),
        "cluster_names": cluster_names,
        "clusters": {str(k): [topic.get('name', '') for topic in v] for k, v in clustered_topics.items()},
        "total_chats": data.get('total_chats', 0)
    }
    
    with open(os.path.join(analyzer.data_dir, "topic_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info("Анализ и кластеризация тем успешно завершены")
    print("\n=== Анализ и кластеризация тем успешно завершены ===")
    print(f"Результаты сохранены в {analyzer.data_dir} и {analyzer.output_dir}")
    print(f"Визуализация кластеров: {vis_path}")
    print(f"Тепловая карта связей: {heatmap_path}")
    print(f"Резюме кластеров: {os.path.join(analyzer.data_dir, 'topic_clusters_summary.md')}")

if __name__ == "__main__":
    asyncio.run(main())
