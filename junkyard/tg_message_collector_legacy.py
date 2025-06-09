#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для сбора личных сообщений из Telegram через API.
Использует библиотеку Telethon для доступа к Telegram.
Соблюдает ограничения API для предотвращения блокировок и FloodWait.
"""

import os
import sys
import json
import time
import asyncio
import logging
import random
import pickle
from datetime import datetime
from dotenv import load_dotenv
from telethon import TelegramClient, events, types
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty, InputPeerUser, InputPeerChannel, InputPeerChat, PeerUser, PeerChat, PeerChannel, User
from telethon.errors import FloodWaitError
from pathlib import Path

# Настройка подробного логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tg_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение учетных данных из переменных окружения
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/messages')

# Проверка наличия необходимых переменных окружения
if not all([API_ID, API_HASH, PHONE]):
    logger.error("❌ Ошибка: Не найдены необходимые переменные окружения (API_ID, API_HASH, PHONE)")
    sys.exit(1)

# Создание директории для выходных файлов, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"📁 Директория для хранения сообщений: {os.path.abspath(OUTPUT_DIR)}")

# Константы ограничений Telegram API (по официальной документации)
# https://core.telegram.org/bots/faq
MAX_MESSAGES_PER_SECOND_PER_CHAT = 1  # 1 сообщение в секунду в одном чате
MAX_MESSAGES_PER_MINUTE_PER_GROUP = 20  # 20 сообщений в минуту в группу
MAX_BULK_MESSAGES_PER_SECOND = 30  # ~30 сообщений в секунду при массовых уведомлениях
SAFE_LIMIT_FACTOR = 0.7  # Используем 70% от максимальных лимитов для безопасности

class TelegramMessageCollector:
    """Класс для сбора сообщений из личных чатов Telegram."""
    
    def __init__(self, api_id, api_hash, phone, data_dir='data/messages'):
        """
        Инициализация сборщика сообщений.
        
        Args:
            api_id (int): Telegram API ID
            api_hash (str): Telegram API Hash
            phone (str): Номер телефона для аутентификации
            data_dir (str): Директория для сохранения собранных сообщений
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        # Создаем папку для конкретного клиента по номеру телефона
        client_folder = phone.replace('+', '').replace('-', '').replace(' ', '')
        self.data_dir = os.path.join(data_dir, f"client_{client_folder}")
        self.checkpoint_file = "telegram_collector_checkpoint.pkl"
        
        # Создаем директорию для данных, если её нет
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        self.client = TelegramClient('session_message_collector', api_id, api_hash)
        self.me = None
        self.messages_data = []
        
        # Статистика обработки
        self.stats = {
            "chats_processed": 0,
            "messages_collected": 0,
            "conversation_pairs_collected": 0,  # Новый счетчик для пар "запрос-ответ"
            "flood_wait_errors": 0,
            "total_wait_time": 0,
            "api_calls": 0,
            "start_time": None,
            "end_time": None,
            "empty_chats": 0,
            "skipped_chats": 0,
            "bot_chats": 0
        }
        
    async def start(self):
        """Запуск клиента Telegram и аутентификация."""
        logger.info("🔄 Запуск клиента Telegram...")
        self.stats["start_time"] = datetime.now()
        
        try:
            await self.client.start(phone=self.phone)
            self.me = await self.client.get_me()
            logger.info(f"✅ Вход выполнен как: {self.me.first_name} (@{self.me.username})")
            logger.info(f"🆔 User ID: {self.me.id}")
        except Exception as e:
            logger.error(f"❌ Ошибка при авторизации: {e}")
            raise
    
    async def api_call(self, coroutine, description="API вызов"):
        """
        Обертка для вызовов API с обработкой ограничений и пауз.
        
        Args:
            coroutine: Корутина для вызова API
            description: Описание вызова для логов
            
        Returns:
            Результат вызова API
        """
        for attempt in range(1, 6):  # Максимум 5 попыток
            try:
                self.stats["api_calls"] += 1
                
                logger.debug(f"📡 Выполнение API вызова: {description}")
                return await coroutine
            except FloodWaitError as e:
                self.stats["flood_wait_errors"] += 1
                self.stats["total_wait_time"] += e.seconds
                
                logger.warning(f"⚠️ FloodWaitError: Необходимо подождать {e.seconds} секунд перед следующим запросом ({description})")
                await asyncio.sleep(e.seconds + random.uniform(0.5, 1.5))  # Добавляем случайную задержку
            except Exception as e:
                logger.error(f"❌ Ошибка при вызове API ({description}): {e}")
                # Добавляем небольшую задержку при ошибке
                await asyncio.sleep(random.uniform(3, 5))
                raise
    
    def save_checkpoint(self, state_dict):
        """
        Сохраняет текущее состояние сбора данных в файл контрольной точки.
        
        Args:
            state_dict (dict): Словарь с текущим состоянием сбора
        """
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state_dict, f)
            logger.info(f"✅ Контрольная точка сохранена в {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении контрольной точки: {e}")
    
    def load_checkpoint(self):
        """
        Загружает сохраненное состояние сбора данных из файла контрольной точки.
        
        Returns:
            dict: Словарь с сохраненным состоянием или None, если файл не найден
        """
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logger.info(f"✅ Загружена контрольная точка: {checkpoint.get('stage', 'неизвестный этап')}")
                
                # Обновляем статистику из контрольной точки, если она есть
                if 'stats' in checkpoint:
                    # Убедимся, что все ключи присутствуют
                    for key in self.stats:
                        if key not in checkpoint['stats']:
                            checkpoint['stats'][key] = 0 if isinstance(self.stats[key], int) else self.stats[key]
                    
                    self.stats = checkpoint['stats']
                
                return checkpoint
            else:
                logger.info("⚠️ Файл контрольной точки не найден")
                return None
        except Exception as e:
            logger.error(f"❌ Ошибка при восстановлении из контрольной точки: {e}")
            return None
            
    async def collect_personal_chats(self, limit_per_chat=None, max_chats=None, delay_between_chats=5):
        """
        Сбор сообщений из личных чатов с улучшенной обработкой ограничений API.
        
        Args:
            limit_per_chat (int, optional): Максимальное количество сообщений для загрузки из каждого чата.
                                           None означает загрузку всех доступных сообщений.
            max_chats (int, optional): Максимальное количество чатов для обработки за один запуск.
                                      None означает обработку всех доступных чатов.
            delay_between_chats (int): Минимальная задержка между обработкой чатов
            
        Returns:
            list: Список собранных сообщений
        """
        checkpoint = self.load_checkpoint()
        if checkpoint and 'stage' in checkpoint:
            try:
                if checkpoint['stage'] == 'loading_dialogs':
                    logger.info("🔄 Возобновление загрузки диалогов")
                    # Продолжаем загрузку с сохраненных параметров
                    dialogs = checkpoint.get('dialogs', [])
                    offset_date = checkpoint.get('offset_date')
                    offset_id = checkpoint.get('offset_id', 0)
                    chunks_loaded = checkpoint.get('chunks_loaded', 0)
                    
                    logger.info(f"📊 Загружено {len(dialogs)} диалогов из контрольной точки")
                    
                    # Вызываем загрузку оставшихся диалогов
                    return await self._continue_loading_dialogs(
                        dialogs, offset_date, offset_id, chunks_loaded,
                        limit_per_chat, max_chats, delay_between_chats
                    )
                elif checkpoint['stage'] == 'dialogs_loaded':
                    logger.info("🔄 Возобновление работы с сохраненными диалогами")
                    dialogs = checkpoint['dialogs']
                    logger.info(f"📊 Загружено {len(dialogs)} диалогов из контрольной точки")
                elif checkpoint['stage'] in ['personal_chats_filtered', 'processing_chats']:
                    logger.info("🔄 Возобновление обработки личных чатов")
                    # Проверяем наличие необходимых ключей
                    if 'personal_chats' not in checkpoint or 'processed_chats' not in checkpoint:
                        logger.warning("⚠️ В контрольной точке отсутствуют необходимые данные, начинаем сбор заново")
                        # Удаляем контрольную точку и начинаем заново
                        os.remove(self.checkpoint_file)
                        return await self.collect_personal_chats(limit_per_chat, max_chats, delay_between_chats)
                    
                    return await self._process_personal_chats(
                        checkpoint['personal_chats'], 
                        checkpoint['processed_chats'],
                        limit_per_chat, 
                        max_chats, 
                        delay_between_chats
                    )
                elif checkpoint['stage'] == 'completed':
                    logger.info("✅ Обработка всех чатов уже завершена ранее")
                    if 'messages_data' in checkpoint:
                        self.messages_data = checkpoint['messages_data']
                        logger.info(f"📊 Всего собрано {len(self.messages_data)} сообщений")
                    return self.messages_data
                else:
                    logger.warning(f"⚠️ Неизвестный этап контрольной точки: {checkpoint['stage']}, начинаем сбор заново")
                    return await self._start_collection(limit_per_chat, max_chats, delay_between_chats)
            except Exception as e:
                logger.error(f"❌ Ошибка при восстановлении из контрольной точки: {e}")
                logger.info("🔄 Начинаем сбор данных заново")
                # Если возникла ошибка при восстановлении, удаляем контрольную точку и начинаем заново
                if os.path.exists(self.checkpoint_file):
                    os.remove(self.checkpoint_file)
                return await self._start_collection(limit_per_chat, max_chats, delay_between_chats)
        else:
            logger.info("🔍 Начинаем новый сбор данных")
            return await self._start_collection(limit_per_chat, max_chats, delay_between_chats)
    
    async def _start_collection(self, limit_per_chat=None, max_chats=None, delay_between_chats=5):
        """
        Начинает сбор сообщений с нуля.
        
        Args:
            limit_per_chat (int, optional): Максимальное количество сообщений для загрузки из каждого чата.
            max_chats (int, optional): Максимальное количество чатов для обработки.
            delay_between_chats (int): Минимальная задержка между обработкой чатов.
            
        Returns:
            list: Список собранных сообщений
        """
        logger.info("🔍 Получение списка диалогов...")
        
        # Получаем диалоги с большими паузами между запросами
        dialogs = []
        offset_date = None
        offset_id = 0
        offset_peer = InputPeerEmpty()
        
        # Постепенно загружаем диалоги небольшими порциями
        chunks_loaded = 0
        chunk_size = 100  # Увеличен размер порции для получения большего количества диалогов
        
        logger.info("📊 Параметры сбора сообщений:")
        if limit_per_chat is None:
            logger.info("   - Загрузка ВСЕХ доступных сообщений из каждого чата")
        else:
            logger.info(f"   - Загрузка до {limit_per_chat} сообщений из каждого чата")
            
        if max_chats is None:
            logger.info("   - Обработка ВСЕХ доступных чатов")
        else:
            logger.info(f"   - Обработка до {max_chats} чатов")
            
        logger.info(f"   - Минимальная задержка между чатами: {delay_between_chats} секунд")
    
        while True:
            try:
                logger.info(f"📥 Загрузка порции диалогов #{chunks_loaded+1} (размер порции: {chunk_size})...")
                
                result = await self.api_call(
                    self.client(GetDialogsRequest(
                        offset_date=offset_date,
                        offset_id=offset_id,
                        offset_peer=offset_peer,
                        limit=chunk_size,
                        hash=0
                    )),
                    f"Получение порции диалогов #{chunks_loaded+1}"
                )
                
                if not result.dialogs:
                    break
                    
                dialogs.extend(result.dialogs)
                logger.info(f"✅ Загружено {len(result.dialogs)} диалогов")
                
                # Сохраняем прогресс после каждой порции диалогов
                checkpoint_data = {
                    'stage': 'loading_dialogs',
                    'dialogs': dialogs,
                    'offset_date': offset_date,
                    'offset_id': offset_id,
                    'chunks_loaded': chunks_loaded
                }
                self.save_checkpoint(checkpoint_data)
                
                if len(result.dialogs) < chunk_size:
                    break
                    
                offset_date = result.messages[-1].date
                offset_peer = result.dialogs[-1].peer
                offset_id = result.messages[-1].id
                
                chunks_loaded += 1
                
                # Добавляем случайную паузу между запросами (безопасная пауза)
                pause_time = random.uniform(4.0, 6.0)
                logger.info(f"⏱️ Пауза {pause_time:.1f} секунд перед загрузкой следующей порции диалогов...")
                await asyncio.sleep(pause_time)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке диалогов: {e}")
                # Больше ждем при ошибке
                await asyncio.sleep(15)
    
        logger.info(f"📊 Найдено {len(dialogs)} диалогов всего")
        
        # Сохраняем все диалоги в контрольной точке
        checkpoint_data = {
            'stage': 'dialogs_loaded',
            'dialogs': dialogs
        }
        self.save_checkpoint(checkpoint_data)
        
        # Увеличим уровень логирования для отладки
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Фильтрация только личных чатов с расширенной диагностикой
        personal_chats = []
        logger.info("🔍 Начинаю анализ типов диалогов...")
        user_count = 0
        
        for i, dialog in enumerate(dialogs):
            try:
                # Проверяем peer чата
                if hasattr(dialog, 'peer'):
                    peer = dialog.peer
                    
                    # Проверяем, является ли peer личным пользователем
                    if isinstance(peer, PeerUser):
                        user_count += 1
                        
                        # Безопасно получаем ID пользователя
                        user_id = getattr(peer, 'user_id', 'Unknown')
                        
                        # Безопасный вывод идентификатора диалога
                        dialog_id = f"ID:{user_id}"
                        try:
                            if hasattr(dialog, 'title') and dialog.title:
                                dialog_id = f"{dialog.title} (ID:{user_id})"
                        except:
                            pass
                            
                        if i < 20:  # Подробный вывод только для первых 20 диалогов
                            logger.debug(f"   ✅ Найден личный чат #{user_count}: {dialog_id}")
                        
                        # Сохраняем диалог и ID пользователя для дальнейшего использования
                        personal_chats.append({
                            'dialog': dialog,
                            'user_id': user_id,
                            'dialog_id': dialog_id
                        })
                    else:
                        chat_type = "Группа" if isinstance(peer, PeerChat) else "Канал" if isinstance(peer, PeerChannel) else "Неизвестный тип"
                        
                        if i < 20:  # Подробный вывод только для первых 20 диалогов
                            logger.debug(f"   ⏩ Пропуск {chat_type}")
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе диалога #{i}: {e}")
        
        logger.info(f"✅ Всего найдено {len(personal_chats)} личных чатов из {len(dialogs)} диалогов")
        
        # Сохраняем список личных чатов в контрольной точке
        checkpoint_data = {
            'stage': 'personal_chats_filtered',
            'personal_chats': personal_chats,
            'processed_chats': []
        }
        self.save_checkpoint(checkpoint_data)
        
        # Обработка личных чатов
        return await self._process_personal_chats(personal_chats, [], limit_per_chat, max_chats, delay_between_chats)
    
    async def _continue_loading_dialogs(self, dialogs, offset_date, offset_id, chunks_loaded, limit_per_chat, max_chats, delay_between_chats):
        """
        Продолжение загрузки диалогов после перезапуска.
        
        Args:
            dialogs (list): Список диалогов, загруженных до перезапуска
            offset_date (datetime): Дата последнего сообщения в последнем диалоге
            offset_id (int): ID последнего сообщения в последнем диалоге
            chunks_loaded (int): Количество загруженных порций диалогов
            limit_per_chat (int, optional): Максимальное количество сообщений для загрузки из каждого чата.
            max_chats (int, optional): Максимальное количество чатов для обработки.
            delay_between_chats (int): Минимальная задержка между обработкой чатов.
            
        Returns:
            list: Список собранных сообщений
        """
        logger.info("🔄 Продолжение загрузки диалогов...")
        
        # Продолжаем загрузку диалогов с сохраненных параметров
        while True:
            try:
                logger.info(f"📥 Загрузка порции диалогов #{chunks_loaded+1}...")
                
                result = await self.api_call(
                    self.client(GetDialogsRequest(
                        offset_date=offset_date,
                        offset_id=offset_id,
                        offset_peer=InputPeerEmpty(),
                        limit=100,
                        hash=0
                    )),
                    f"Получение порции диалогов #{chunks_loaded+1}"
                )
                
                if not result.dialogs:
                    break
                    
                dialogs.extend(result.dialogs)
                logger.info(f"✅ Загружено {len(result.dialogs)} диалогов")
                
                # Сохраняем прогресс после каждой порции диалогов
                checkpoint_data = {
                    'stage': 'loading_dialogs',
                    'dialogs': dialogs,
                    'offset_date': offset_date,
                    'offset_id': offset_id,
                    'chunks_loaded': chunks_loaded
                }
                self.save_checkpoint(checkpoint_data)
                
                if len(result.dialogs) < 100:
                    break
                    
                offset_date = result.messages[-1].date
                offset_id = result.messages[-1].id
                
                chunks_loaded += 1
                
                # Добавляем случайную паузу между запросами (безопасная пауза)
                pause_time = random.uniform(4.0, 6.0)
                logger.info(f"⏱️ Пауза {pause_time:.1f} секунд перед загрузкой следующей порции диалогов...")
                await asyncio.sleep(pause_time)
                
            except Exception as e:
                logger.error(f"❌ Ошибка при загрузке диалогов: {e}")
                # Больше ждем при ошибке
                await asyncio.sleep(15)
    
        logger.info(f"📊 Найдено {len(dialogs)} диалогов всего")
        
        # Сохраняем все диалоги в контрольной точке
        checkpoint_data = {
            'stage': 'dialogs_loaded',
            'dialogs': dialogs
        }
        self.save_checkpoint(checkpoint_data)
        
        # Увеличим уровень логирования для отладки
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Фильтрация только личных чатов с расширенной диагностикой
        personal_chats = []
        logger.info("🔍 Начинаю анализ типов диалогов...")
        user_count = 0
        
        for i, dialog in enumerate(dialogs):
            try:
                # Проверяем peer чата
                if hasattr(dialog, 'peer'):
                    peer = dialog.peer
                    
                    # Проверяем, является ли peer личным пользователем
                    if isinstance(peer, PeerUser):
                        user_count += 1
                        
                        # Безопасно получаем ID пользователя
                        user_id = getattr(peer, 'user_id', 'Unknown')
                        
                        # Безопасный вывод идентификатора диалога
                        dialog_id = f"ID:{user_id}"
                        try:
                            if hasattr(dialog, 'title') and dialog.title:
                                dialog_id = f"{dialog.title} (ID:{user_id})"
                        except:
                            pass
                            
                        if i < 20:  # Подробный вывод только для первых 20 диалогов
                            logger.debug(f"   ✅ Найден личный чат #{user_count}: {dialog_id}")
                        
                        # Сохраняем диалог и ID пользователя для дальнейшего использования
                        personal_chats.append({
                            'dialog': dialog,
                            'user_id': user_id,
                            'dialog_id': dialog_id
                        })
                    else:
                        chat_type = "Группа" if isinstance(peer, PeerChat) else "Канал" if isinstance(peer, PeerChannel) else "Неизвестный тип"
                        
                        if i < 20:  # Подробный вывод только для первых 20 диалогов
                            logger.debug(f"   ⏩ Пропуск {chat_type}")
            except Exception as e:
                logger.error(f"❌ Ошибка при анализе диалога #{i}: {e}")
        
        logger.info(f"✅ Всего найдено {len(personal_chats)} личных чатов из {len(dialogs)} диалогов")
        
        # Сохраняем список личных чатов в контрольной точке
        checkpoint_data = {
            'stage': 'personal_chats_filtered',
            'personal_chats': personal_chats,
            'processed_chats': []
        }
        self.save_checkpoint(checkpoint_data)
        
        # Обработка личных чатов
        return await self._process_personal_chats(personal_chats, [], limit_per_chat, max_chats, delay_between_chats)
    
    async def _get_messages_from_dialog(self, dialog, limit=None):
        """
        Получает сообщения из диалога с обработкой ограничений API.
        
        Args:
            dialog: Объект диалога
            limit (int, optional): Максимальное количество сообщений для загрузки
            
        Returns:
            list: Список сообщений
        """
        messages = []
        offset_id = 0
        loaded_count = 0
        
        try:
            # Получаем peer диалога
            entity = await self.client.get_entity(dialog.peer)
            
            # Устанавливаем безопасный лимит для личных чатов - не более 3000 сообщений
            if isinstance(entity, types.User):
                if limit is None or limit > 3000:
                    safe_limit = 3000
                    logger.info(f"⚠️ Установлено ограничение в 3000 сообщений для личного чата")
                else:
                    safe_limit = limit
                
                # Выводим информацию о пользователе для диагностики
                user_info = []
                if hasattr(entity, 'first_name') and entity.first_name:
                    user_info.append(entity.first_name)
                if hasattr(entity, 'last_name') and entity.last_name:
                    user_info.append(entity.last_name)
                    
                user_name = " ".join(user_info) if user_info else "Неизвестный пользователь"
                user_id = entity.id
                username = f"@{entity.username}" if hasattr(entity, 'username') and entity.username else "без username"
                
                logger.info(f"👤 Загрузка сообщений из личного чата с: {user_name} ({username}, ID:{user_id})")
                
                # Проверяем, не является ли пользователь ботом
                if hasattr(entity, 'bot') and entity.bot:
                    logger.warning(f"🤖 Обнаружен бот {user_name}. Пропускаем...")
                    # Увеличиваем счетчик ботов в статистике
                    if hasattr(self, 'stats'):
                        self.stats['bot_chats'] += 1
                    return []
                
                # Проверяем точно что это не группа/канал
                if hasattr(entity, 'megagroup') and entity.megagroup:
                    logger.warning(f"⚠️ Обнаружена мегагруппа вместо личного чата с {user_name}. Пропускаем...")
                    return []
                if hasattr(entity, 'broadcast') and entity.broadcast:
                    logger.warning(f"⚠️ Обнаружен канал вместо личного чата с {user_name}. Пропускаем...")
                    return []
            else:
                logger.warning(f"⚠️ Диалог не является личным чатом (тип: {type(entity).__name__}). Пропускаем...")
                return []
                
            # Проверяем подключение перед запросом сообщений
            if not self.client.is_connected():
                logger.warning("⚠️ Клиент не подключен перед запросом сообщений. Выполняем переподключение...")
                await self.reconnect()
                
            # Загружаем сообщения с ограничением
            while True:
                try:
                    history = await self.api_call(
                        self.client.get_messages(entity, limit=100, offset_id=offset_id),
                        f"Загрузка сообщений (offset_id={offset_id})"
                    )
                    
                    if not history:
                        break
                        
                    messages.extend(history)
                    loaded_count += len(history)
                    offset_id = history[-1].id
                    
                    # Обновляем прогресс каждые 500 сообщений
                    if loaded_count % 500 == 0:
                        logger.info(f"   ✅ Загружено {loaded_count} сообщений из чата с {user_name}...")
                    
                    # Проверяем достижение лимита
                    if safe_limit and loaded_count >= safe_limit:
                        logger.info(f"   ⚠️ Достигнут лимит в {safe_limit} сообщений для чата")
                        break
                        
                    # Небольшая пауза между запросами
                    await asyncio.sleep(random.uniform(0.3, 0.7))
                    
                except FloodWaitError as e:
                    logger.warning(f"⚠️ FloodWaitError: Ожидание {e.seconds} секунд...")
                    await asyncio.sleep(e.seconds + random.uniform(1, 3))
                except Exception as e:
                    logger.error(f"❌ Ошибка при загрузке сообщений: {e}")
                    # Пауза перед повторной попыткой
                    await asyncio.sleep(random.uniform(3, 5))
                    break
                    
            logger.info(f"✅ Загружено {len(messages)} сообщений из чата с {user_name}")
            return messages
        except Exception as e:
            logger.error(f"❌ Ошибка при получении сообщений: {e}")
            return []
    
    def _process_messages_to_conversation_pairs(self, messages):
        """
        Обрабатывает сообщения, создавая пары "запрос-ответ" для улучшения контекста.
        
        Args:
            messages (list): Список сообщений из диалога
            
        Returns:
            tuple: (список сообщений, список пар "запрос-ответ")
        """
        messages_json = []
        conversation_pairs = []
        owner_id = self.me.id
        
        # Сортируем сообщения по дате (от старых к новым)
        sorted_messages = sorted(messages, key=lambda x: x.date)
        
        previous_message = None
        
        for msg in sorted_messages:
            try:
                # Обработка текущего сообщения
                message_data = {
                    'id': msg.id,
                    'date': msg.date.isoformat(),
                    'message': msg.message,
                    'out': msg.out,
                    'mentioned': msg.mentioned,
                    'media_unread': msg.media_unread,
                    'silent': msg.silent if hasattr(msg, 'silent') else None,
                    'post': msg.post if hasattr(msg, 'post') else None,
                    'from_scheduled': msg.from_scheduled if hasattr(msg, 'from_scheduled') else None,
                    'legacy': msg.legacy if hasattr(msg, 'legacy') else None,
                    'edit_hide': msg.edit_hide if hasattr(msg, 'edit_hide') else None,
                    'pinned': msg.pinned if hasattr(msg, 'pinned') else None,
                    'noforwards': msg.noforwards if hasattr(msg, 'noforwards') else None,
                    'from_id': msg.from_id.user_id if hasattr(msg, 'from_id') and hasattr(msg.from_id, 'user_id') else None
                }
                
                # Добавляем в общий список сообщений
                messages_json.append(message_data)
                
                # Формируем пары "запрос-ответ"
                # Если текущее сообщение от владельца и предыдущее от другого пользователя
                if previous_message and msg.message and previous_message.message:
                    current_from_owner = msg.out
                    previous_not_from_owner = not previous_message.out
                    
                    # Если это ответ владельца на сообщение другого пользователя
                    if current_from_owner and previous_not_from_owner:
                        pair = {
                            'query': previous_message.message,
                            'response': msg.message,
                            'query_id': previous_message.id,
                            'response_id': msg.id,
                            'query_date': previous_message.date.isoformat(),
                            'response_date': msg.date.isoformat()
                        }
                        conversation_pairs.append(pair)
                
                # Сохраняем текущее сообщение как предыдущее для следующей итерации
                previous_message = msg
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка при обработке сообщения {msg.id}: {e}")
                
        return messages_json, conversation_pairs
    
    async def _process_personal_chats(self, personal_chats, processed_chats=None, limit_per_chat=None, max_chats=None, delay_between_chats=5):
        """
        Обработка личных чатов и сбор сообщений.
        
        Args:
            personal_chats (list): Список личных чатов
            processed_chats (list, optional): Список уже обработанных чатов
            limit_per_chat (int, optional): Максимальное количество сообщений для загрузки из каждого чата
            max_chats (int, optional): Максимальное количество чатов для обработки
            delay_between_chats (int): Минимальная задержка между обработкой чатов
            
        Returns:
            list: Список собранных сообщений
        """
        if processed_chats is None:
            processed_chats = []
            
        # Определяем ограничения
        safe_limit_per_chat = limit_per_chat  # Может быть None
        total_chats = len(personal_chats)
        
        # Ограничиваем количество чатов, если задано
        max_to_process = min(total_chats, max_chats) if max_chats else total_chats
        personal_chats_to_process = personal_chats[:max_to_process]
        logger.info(f"🔄 Начинаем обработку {max_to_process} из {total_chats} личных чатов")
        
        # Если есть обработанные чаты, выводим информацию
        if len(processed_chats) > 0:
            logger.info(f"⏭️ Пропускаем {len(processed_chats)} ранее обработанных чатов")
            
        # Проверяем сколько чатов осталось обработать
        chats_remaining = max_to_process - len(processed_chats)
        
        if chats_remaining <= 0:
            logger.info("✅ Все чаты уже обработаны")
            # Сохраняем контрольную точку о завершении
            checkpoint_data = {
                'stage': 'completed',
                'messages_data': self.messages_data
            }
            self.save_checkpoint(checkpoint_data)
            return self.messages_data
        
        logger.info(f"📊 Осталось обработать {chats_remaining} чатов")
            
        # Обрабатываем каждый чат
        for i, chat in enumerate(personal_chats_to_process):
            # Пропускаем уже обработанные чаты
            if i < len(processed_chats):
                continue
                
            # Проверяем подключение перед обработкой каждого чата
            if not self.client.is_connected():
                logger.warning("⚠️ Клиент не подключен перед обработкой чата. Выполняем переподключение...")
                await self.reconnect()
            
            processed_count = len(processed_chats)
            total_progress = f"{processed_count + 1}/{total_chats}"
            
            # Определяем данные чата в зависимости от формата
            if isinstance(chat, dict):
                dialog = chat['dialog']
                user_id = chat['user_id']
                dialog_id = chat['dialog_id']
                logger.info(f"📨 Обработка чата {total_progress}: ID:{dialog_id}")
            else:
                dialog = chat
                user_id = chat.id
                dialog_id = chat.id
                logger.info(f"📨 Обработка чата {total_progress}: ID:{chat.id}")
            
            try:
                # Создаем директорию для пользователя
                user_dir = os.path.join(self.data_dir, f"user_{user_id}")
                os.makedirs(user_dir, exist_ok=True)
                
                # Файлы для сохранения сообщений и пар "запрос-ответ"
                messages_file = os.path.join(user_dir, 'messages.json')
                conversations_file = os.path.join(user_dir, 'conversations.json')
                
                # Проверяем, есть ли уже сохраненные сообщения
                if os.path.exists(messages_file) and os.path.exists(conversations_file):
                    logger.info(f"📁 Найдены существующие файлы сообщений и диалогов для чата ID:{dialog_id}")
                    
                    try:
                        # Проверяем формат существующих файлов
                        with open(messages_file, 'r', encoding='utf-8') as f:
                            existing_messages = json.load(f)
                        with open(conversations_file, 'r', encoding='utf-8') as f:
                            existing_conversations = json.load(f)
                            
                        # Проверяем, что файлы не пустые и содержат данные в нужном формате
                        if existing_messages and isinstance(existing_messages, list) and len(existing_messages) > 0:
                            # Проверяем первое сообщение на наличие нужных полей
                            first_msg = existing_messages[0]
                            if 'id' in first_msg and 'date' in first_msg and 'message' in first_msg:
                                logger.info(f"✅ Найдено {len(existing_messages)} сообщений и {len(existing_conversations)} пар запрос-ответ, пропускаем чат")
                                # Добавляем в статистику
                                if hasattr(self, 'stats'):
                                    self.stats['messages_collected'] += len(existing_messages)
                                    self.stats['conversation_pairs_collected'] += len(existing_conversations)
                                # Добавляем в список обработанных чатов
                                processed_chats.append(chat)
                                # Переходим к следующему чату
                                continue
                            else:
                                logger.warning(f"⚠️ Формат существующего файла сообщений некорректен, загружаем заново")
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка при чтении существующих файлов: {e}, загружаем заново")
                
                # Загружаем сообщения с обработкой ограничений
                messages = await self._get_messages_from_dialog(dialog, safe_limit_per_chat)
                
                if messages:
                    # Преобразуем сообщения в формат для сохранения и создаем пары "запрос-ответ"
                    messages_json, conversation_pairs = self._process_messages_to_conversation_pairs(messages)
                    
                    # Сохраняем сообщения в файл
                    with open(messages_file, 'w', encoding='utf-8') as f:
                        json.dump(messages_json, f, ensure_ascii=False, indent=2)
                    
                    # Сохраняем пары "запрос-ответ" в файл
                    with open(conversations_file, 'w', encoding='utf-8') as f:
                        json.dump(conversation_pairs, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"✅ Сохранено {len(messages_json)} сообщений и {len(conversation_pairs)} пар запрос-ответ для чата {dialog_id}")
                    
                    # Обновляем статистику
                    if hasattr(self, 'stats'):
                        self.stats['messages_collected'] += len(messages_json)
                        self.stats['conversation_pairs_collected'] += len(conversation_pairs)
                else:
                    logger.info(f"⚠️ Нет сообщений для сохранения в чате с ID:{dialog_id}")
                    if hasattr(self, 'stats'):
                        if 'empty_chats' in self.stats:
                            self.stats['empty_chats'] += 1
                        else:
                            # Если поле не существует, создаем его
                            self.stats['empty_chats'] = 1
            
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке чата {dialog_id}: {e}")
                if hasattr(self, 'stats'):
                    if hasattr(self.stats, 'skipped_chats'):
                        self.stats['skipped_chats'] += 1
                    else:
                        # Если поле не существует, создаем его
                        self.stats['skipped_chats'] = 1
                
            # Добавляем в список обработанных чатов
            processed_chats.append(chat)
            
            # Обновляем контрольную точку после каждого обработанного чата
            checkpoint_data = {
                'stage': 'processing_chats',
                'personal_chats': personal_chats,
                'processed_chats': processed_chats,
                'messages_data': self.messages_data
            }
            self.save_checkpoint(checkpoint_data)
            
            # Случайная пауза между чатами для избежания ограничений API
            pause_time = random.uniform(delay_between_chats, delay_between_chats + 2)
            logger.info(f"⏱️ Пауза {pause_time:.1f} секунд перед обработкой следующего чата...")
            await asyncio.sleep(pause_time)
            
        logger.info(f"🎉 Обработка личных чатов завершена. Собрано {self.stats['messages_collected']} сообщений и {self.stats['conversation_pairs_collected']} пар запрос-ответ из {len(processed_chats)} чатов")
        
        # Финальная контрольная точка - все чаты обработаны
        checkpoint_data = {
            'stage': 'completed',
            'personal_chats': personal_chats,
            'processed_chats': processed_chats,
            'messages_data': self.messages_data,
            'stats': self.stats
        }
        self.save_checkpoint(checkpoint_data)
        
        return self.messages_data
    
    async def save_messages_to_json(self, filename=None):
        """
        Сохранение собранных сообщений в JSON файл.
        
        Args:
            filename (str, optional): Имя файла для сохранения. По умолчанию генерируется автоматически.
            
        Returns:
            str: Путь к созданному файлу
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telegram_messages_{timestamp}.json"
        
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        logger.info(f"💾 Сохранение {len(self.messages_data)} сообщений в файл...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.messages_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Сообщения сохранены в файл: {file_path}")
        return file_path
    
    async def disconnect(self):
        """
        Закрытие клиента Telegram и освобождение ресурсов.
        """
        if hasattr(self, 'client') and self.client is not None:
            try:
                await self.client.disconnect()
                logger.debug("✅ Клиент Telegram успешно отключен")
            except Exception as e:
                logger.error(f"❌ Ошибка при отключении клиента: {e}")
    
    async def close(self):
        """Завершение работы клиента."""
        await self.disconnect()
        logger.info("👋 Клиент Telegram отключен")
        
        # Фиксируем время завершения и выводим статистику
        self.stats["end_time"] = datetime.now()
        elapsed_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        logger.info("\n" + "="*50)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"⏱️ Время выполнения: {elapsed_time:.1f} секунд")
        logger.info(f"💬 Обработано чатов: {self.stats['chats_processed']}")
        logger.info(f"📝 Собрано сообщений: {self.stats['messages_collected']}")
        logger.info(f"🔄 Собрано пар запрос-ответ: {self.stats['conversation_pairs_collected']}")
        logger.info(f"⚠️ Случаев FloodWaitError: {self.stats['flood_wait_errors']}")
        logger.info(f"⏳ Общее время ожидания из-за FloodWaitError: {self.stats['total_wait_time']} секунд")
        if self.stats["messages_collected"] > 0:
            logger.info(f"⚡️ Средняя скорость: {self.stats['messages_collected'] / elapsed_time:.2f} сообщений/сек")
        logger.info("="*50)

    async def reconnect(self):
        """Переподключение к Telegram."""
        try:
            await self.client.reconnect()
            logger.info("✅ Переподключение к Telegram выполнено успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при переподключении к Telegram: {e}")

async def main():
    """Основная функция для запуска сбора сообщений."""
    start_time = time.time()
    logger.info("📁 Директория для хранения сообщений: /Users/wsgp/TGparser/data/messages")
    
    # Загружаем переменные окружения из .env
    load_dotenv()
    api_id = os.getenv('API_ID')
    api_hash = os.getenv('API_HASH')
    phone = os.getenv('PHONE')
    
    if not api_id or not api_hash or not phone:
        logger.error("❌ Отсутствуют необходимые переменные окружения в файле .env")
        logger.error("Убедитесь, что переменные API_ID, API_HASH и PHONE заданы")
        return
    
    # Создаем экземпляр сборщика
    collector = TelegramMessageCollector(api_id, api_hash, phone)
    logger.info("🚀 Запуск сборщика сообщений из Telegram")
    
    try:
        # Подключаемся к Telegram
        logger.info("🔄 Запуск клиента Telegram...")
        await collector.start()
        
        # Сбор личных сообщений
        try:
            messages = await collector.collect_personal_chats(
                limit_per_chat=None,  # Получать все доступные сообщения
                max_chats=None,  # Обрабатывать все доступные чаты
                delay_between_chats=5  # Пауза между чатами
            )
            
            # Дополнительно сохраняем все сообщения в отдельный файл
            if messages:
                logger.info(f"🎯 Сохранение {len(messages)} сообщений в общий архив...")
                await collector.save_messages_to_json()
                
        except Exception as e:
            logger.error(f"❌ Произошла ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске сборщика: {e}")
    finally:
        # Закрываем клиент в любом случае
        await collector.close()
        logger.info("👋 Клиент отключен")
        
        # Выводим статистику
        logger.info(f"⏱️ Общее время выполнения скрипта: {time.time() - start_time:.2f} секунд")
        if hasattr(collector, 'stats') and collector.stats:
            logger.info("📊 Статистика:")
            for key, value in collector.stats.items():
                logger.info(f"   - {key}: {value}")

if __name__ == "__main__":
    # Запуск асинхронной функции main
    asyncio.run(main())
