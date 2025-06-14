# 🔍 Анализатор чатов Telegram «О чем я?»

## 🤖 Что делает этот инструмент?

Этот инструмент анализирует ваши Telegram-чаты и показывает реальную картину того, о чем вы говорите, какие темы вас волнуют, и даже как можно монетизировать ваши знания. По сути, это взгляд со стороны на ваше мышление и интересы через призму общения в Telegram.

Он состоит из нескольких основных модулей:

1. **Сборщик чатов** - автоматически собирает ваши сообщения из Telegram и сохраняет их в формате JSON.
2. **Анализатор содержимого** - исследует ваши чаты и выявляет основные темы, эмоции и возможности для монетизации.
3. **Кластеризатор тем** - группирует похожие темы в значимые кластеры для более глубокого понимания обсуждаемых вопросов.
4. **Генератор стратегий монетизации** - создает детальные рекомендации по монетизации на основе выявленных тематик.

## ⚙️ Как это работает?

1. **💬 Разбирает сообщения** – вытягивает все диалоги и ключевые фразы
2. **📊 Выявляет главные темы** – определяет, о чем вы говорите чаще всего
3. **🎭 Анализирует эмоции** – позитив, негатив, нейтрал для каждой темы
4. **🔬 Кластеризует темы** – группирует похожие темы в логические кластеры
5. **💰 Ищет возможности для монетизации** – подскажет, где можно сделать бизнес
6. **📑 Генерирует бизнес-план** – дает готовую стратегию на основе ваших обсуждений

## 🧠 Зачем это нужно?

- **Понять, о чем вы на самом деле** – часто мы говорим одно, а думаем о другом
- **Определить, куда двигаться дальше** – какие темы реально зажигают
- **Найти новые бизнес-возможности** – чтобы монетизировать свои знания
- **Посмотреть на себя со стороны** – это как зеркало для мышления

## 📋 Структура проекта

```
- run_analysis.py           # Главный скрипт для запуска анализа чатов
- chatgpt_analyzer.py       # Класс для анализа чатов с помощью ChatGPT
- chat_monetization_analyzer.py # Анализ возможностей монетизации
- topic_clustering.py       # Кластеризация тем с помощью K-means и OpenAI
- tg_message_collector.py   # Сборщик сообщений из Telegram
- check_bot_accounts.py     # Проверка учетных записей ботов
- adapter_script.py         # Адаптация данных анализа для последующих этапов
- continue_analysis.py      # Продолжение анализа с промежуточных этапов
- monetization_strategy.md  # Документ со стратегиями монетизации
- .env.example              # Пример файла с переменными окружения
- requirements.txt          # Зависимости проекта
- data/                     # Директория для данных
  - messages/               # Сохраненные сообщения
  - reports/                # Отчеты анализа
  - visualizations/         # Визуализации и графики (PNG, HTML)
- samples/                  # Директория с примерами
  - demo_chat.json          # Пример чата для демонстрации
- logs/                     # Журналы работы программы
- docs/                     # Документация
```

## 🔧 Основные файлы проекта

### run_analysis.py
**Описание**: Главный файл проекта, который запускает весь процесс анализа. Позволяет анализировать как один чат, так и все доступные чаты.

**Функционал**:
- Загрузка сообщений из указанной директории (по умолчанию из data/messages/)
- Фильтрация сообщений по различным критериям (количество, временной диапазон)
- Последовательный запуск анализаторов (тем, эмоций, монетизации)
- Формирование итоговых отчетов в формате JSON и Markdown
- Создание визуализаций (графики распределения тем, эмоций, активности)

**Параметры запуска**:
- `--chat-dir` - директория с сообщениями для анализа
- `--output-dir` - директория для сохранения результатов
- `--min-messages` - минимальное количество сообщений в чате для анализа
- `--max-messages` - максимальное количество сообщений для обработки

### chatgpt_analyzer.py
**Описание**: Класс для анализа содержимого чатов с использованием OpenAI API (ChatGPT). Это ядро аналитической части проекта.

**Функционал**:
- Обработка текстовых сообщений и выявление ключевых тем
- Анализ эмоциональной окраски сообщений и тем
- Оценка временных паттернов общения
- Идентификация доминирующих тем и интересов
- Генерация структурированных данных для дальнейшего анализа

**Ключевые методы**:
- `analyze_chat()` - выполняет комплексный анализ чата
- `extract_topics()` - извлекает основные темы из текста
- `analyze_sentiment()` - определяет эмоциональную окраску
- `generate_report()` - создает структурированный отчет

### chat_monetization_analyzer.py
**Описание**: Модуль для анализа возможностей монетизации на основе выявленных тем и интересов. Позволяет находить потенциальные бизнес-возможности.

**Функционал**:
- Анализ перспективности монетизации для каждой темы
- Выявление наиболее коммерчески привлекательных тематик
- Генерация конкретных идей продуктов и услуг
- Оценка потенциального рынка и аудитории
- Создание детальных бизнес-стратегий для монетизации знаний

**Ключевые методы**:
- `analyze_monetization_potential()` - оценивает потенциал монетизации
- `generate_monetization_ideas()` - создает идеи продуктов/услуг
- `create_business_strategies()` - разрабатывает бизнес-стратегии

### topic_clustering.py
**Описание**: Модуль для кластеризации и группировки похожих тем с использованием алгоритмов машинного обучения и OpenAI API. Позволяет выявить скрытые взаимосвязи между темами.

**Функционал**:
- Преобразование текстовых тем в векторные представления (эмбеддинги)
- Кластеризация тем с использованием алгоритма K-means
- Автоматическое определение оптимального количества кластеров
- Генерация содержательных названий для кластеров с помощью ChatGPT
- Создание интерактивных визуализаций для анализа связей между темами

**Ключевые методы**:
- `get_topic_embeddings()` - получает векторные представления тем
- `cluster_topics()` - группирует темы в кластеры
- `name_clusters()` - генерирует названия для кластеров
- `visualize_clusters()` - создает визуализации (2D-карты, тепловые карты)

### tg_message_collector.py
**Описание**: Сборщик сообщений из Telegram. Автоматически извлекает сообщения из ваших чатов с использованием Telethon API.

**Функционал**:
- Авторизация в Telegram через официальный API
- Получение списка всех доступных чатов
- Фильтрация чатов по различным параметрам
- Извлечение сообщений с метаданными (время, автор, вложения)
- Сохранение сообщений в структурированном JSON-формате

**Параметры запуска**:
- `--limit-per-chat` - максимальное количество сообщений из одного чата
- `--max-chats` - максимальное количество чатов для обработки
- `--delay` - задержка между запросами к API для избежания ограничений
- `--filter-type` - тип чатов для сбора (личные, группы, каналы)

### adapter_script.py
**Описание**: Адаптер для преобразования форматов данных анализа между различными этапами обработки. Обеспечивает совместимость данных между модулями.

**Функционал**:
- Преобразование первичных результатов анализа для использования в кластеризации
- Объединение данных из различных чатов в единый набор
- Фильтрация и нормализация данных для повышения качества анализа
- Подготовка данных для последующей генерации стратегий монетизации
- Конвертация между различными форматами данных (JSON, CSV, внутренние структуры)

**Ключевые методы**:
- `adapt_analysis_format()` - преобразует данные анализа в нужный формат
- `merge_chat_analysis()` - объединяет результаты анализа из разных чатов
- `normalize_topics_data()` - нормализует данные о темах для кластеризации

### continue_analysis.py
**Описание**: Скрипт для продолжения анализа с промежуточных этапов или перезапуска отдельных компонентов без необходимости повторения всего процесса.

**Функционал**:
- Загрузка ранее сохраненных результатов анализа
- Выборочное выполнение определенных этапов аналитического процесса
- Перегенерация отчетов и визуализаций с использованием кэшированных данных
- Обновление отдельных компонентов анализа без повторного запуска всего процесса
- Экспериментирование с различными параметрами анализа

**Ключевые команды**:
- `--regenerate-report` - пересоздает отчеты из существующих данных
- `--cluster-only` - выполняет только кластеризацию тем
- `--monetization-only` - генерирует только стратегии монетизации

### check_bot_accounts.py
**Описание**: Утилита для выявления и анализа ботов в собранных данных Telegram. Помогает отфильтровать автоматические сообщения для повышения качества анализа.

**Функционал**:
- Обнаружение потенциальных ботов по шаблонам имен пользователей
- Анализ паттернов сообщений, характерных для ботов
- Статистика по активности ботов в различных чатах
- Опциональная фильтрация или исключение сообщений ботов из анализа
- Формирование отчета о выявленных ботах

**Параметры запуска**:
- `--dir` - директория с данными для анализа
- `--fix` - автоматически удаляет чаты, где большинство сообщений от ботов
- `--report` - создает детальный отчет о найденных ботах

### monetization_strategy.md
**Описание**: Документ с детальными стратегиями монетизации, автоматически генерируемый на основе анализа тем и кластеризации.

**Содержание**:
- Обзор ключевых тематических кластеров и их потенциала
- Конкретные продукты и услуги для каждого кластера
- Анализ целевой аудитории и её потребностей
- Маркетинговые рекомендации для продвижения
- Пошаговый план реализации бизнес-возможностей
- Оценка ресурсов и временных затрат на реализацию

## 🗂️ Структура каталогов

### data/
**Описание**: Основная директория для хранения всех данных проекта: исходных сообщений, результатов анализа и визуализаций.

**Подкаталоги**:
- **messages/** - содержит собранные сообщения из Telegram в формате JSON
- **reports/** - хранит результаты анализа (JSON-файлы) и отчеты (Markdown)
- **visualizations/** - содержит все визуализации (PNG-графики и HTML-интерактивные карты)

### samples/
**Описание**: Директория с примерами и демонстрационными данными.

**Содержимое**:
- **demo_chat.json** - пример чата для тестирования функциональности
- **sample_reports/** - примеры отчетов для ознакомления с результатами

### logs/
**Описание**: Директория для хранения журналов работы программы и отладочной информации.

### docs/
**Описание**: Документация по проекту, включая руководства, схемы и обучающие материалы.

## 🚀 Как использовать

### Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/tgparser.git
cd tgparser
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Создайте файл .env на основе примера:
```bash
cp .env.example .env
```

4. Добавьте ваш API ключ OpenAI и Telegram API ключ в файл .env:
```
OPENAI_API_KEY=your_api_key_here
API_ID=your_telegram_api_id
API_HASH=your_telegram_api_hash
PHONE=your_phone_number
```

### Сбор сообщений из Telegram

**Автоматический сбор:**
```bash
python tg_message_collector.py
```

Эта команда запустит сборщик сообщений, который:
1. Авторизуется в Telegram используя данные из .env
2. Загрузит список всех ваших чатов
3. Соберет сообщения из личных чатов и сохранит в data/messages/

**Параметры сборщика:**
```bash
python tg_message_collector.py --limit-per-chat 1000 --max-chats 50
```

- `--limit-per-chat` - максимальное число сообщений из одного чата
- `--max-chats` - максимальное число чатов для обработки
- `--delay` - задержка между обработкой чатов (сек)

### Подготовка данных

Если у вас уже есть экспорт данных из Telegram Desktop, поместите его в директорию `data/raw/`.

### Запуск анализа

```bash
python run_analysis.py
```

**Параметры анализа:**
```bash
python run_analysis.py --chat-dir data/messages/ --output-dir data/reports/
```

- `--chat-dir` - директория с сообщениями
- `--output-dir` - директория для сохранения результатов
- `--min-messages N` - минимальное количество сообщений в чате
- `--max-messages N` - максимальное количество сообщений для обработки

### Кластеризация тем

После основного анализа вы можете запустить кластеризацию тем:

```bash
python topic_clustering.py
```

Это создаст интерактивные визуализации в директории `data/visualizations/`:
- Карта кластеров тем (topic_clusters_map.html)
- Тепловая карта сходства тем (topic_similarity_heatmap.html)

### Генерация стратегий монетизации

На основе кластеризации вы можете сгенерировать детальные стратегии монетизации:

```bash
python continue_analysis.py
```

Результат будет сохранен в файле `monetization_strategy.md`.

## 📝 Результаты анализа

После завершения анализа в директории `data/reports/` будут созданы следующие файлы:

1. `topic_analysis.json` - основные темы ваших обсуждений
2. `sentiment_analysis.json` - эмоциональный анализ по темам
3. `monetization_opportunities.json` - возможности для монетизации

В директории `data/visualizations/` будут созданы графики и диаграммы для визуального представления результатов.

## 🔑 Требования

- Python 3.8+
- OpenAI API ключ
- Telegram API ключ

## 💡 Примечания

- Для качественного анализа рекомендуется использовать не менее 300-500 сообщений
- Стоимость анализа зависит от количества сообщений и тарифа OpenAI

## 👨‍💻 Автор

[Sergei Dyshkant](https://t.me/sergei_dyshkant) (SergD)

## 📜 Лицензия

MIT License
