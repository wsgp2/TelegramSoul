#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Оптимизированный модуль для сбора сообщений из Telegram.
Принцип 80/20: максимум функциональности, минимум сложности.
"""

import os
import sys
import json
import time
import asyncio
import logging
import pickle
from datetime import datetime
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from telethon.errors import FloodWaitError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tg_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
PHONE = os.getenv('PHONE')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'data/messages')

if not all([API_ID, API_HASH, PHONE]):
    logger.error("❌ Отсутствуют переменные окружения (API_ID, API_HASH, PHONE)")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

class TelegramMessageCollector:
    """Оптимизированный сборщик сообщений Telegram."""
    
    def __init__(self, api_id, api_hash, phone, data_dir='data/messages'):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        
        # Создаем папку для конкретного клиента
        client_folder = phone.replace('+', '').replace('-', '').replace(' ', '')
        self.data_dir = os.path.join(data_dir, f"client_{client_folder}")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.client = TelegramClient('anton_vlasov_session', api_id, api_hash)
        self.me = None
        self.all_messages = []
        
        # Checkpoint файл
        self.checkpoint_file = os.path.join(self.data_dir, "checkpoint.pkl")
        
        # Простая статистика
        self.stats = {
            "chats_processed": 0,
            "messages_collected": 0,
            "start_time": None,
            "end_time": None
        }
        
    async def start(self):
        """Запуск клиента и аутентификация."""
        logger.info("🔄 Запуск клиента Telegram...")
        self.stats["start_time"] = datetime.now()
        
        await self.client.start(phone=self.phone)
        self.me = await self.client.get_me()
        logger.info(f"✅ Вход выполнен: {self.me.first_name} (@{self.me.username})")
        
    def save_checkpoint(self, processed_chat_ids, all_messages):
        """Сохранение checkpoint после каждого чата."""
        try:
            checkpoint_data = {
                'processed_chat_ids': processed_chat_ids,
                'messages': all_messages,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.debug(f"💾 Checkpoint сохранен: {len(processed_chat_ids)} чатов")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения checkpoint: {e}")
    
    def load_checkpoint(self):
        """Загрузка checkpoint для продолжения работы."""
        if not os.path.exists(self.checkpoint_file):
            return None, []
            
        try:
            with open(self.checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            processed_ids = data.get('processed_chat_ids', [])
            messages = data.get('messages', [])
            saved_stats = data.get('stats', {})
            
            # Восстанавливаем статистику
            self.stats.update(saved_stats)
            self.all_messages = messages
            
            logger.info(f"🔄 Восстановлено из checkpoint: {len(processed_ids)} чатов, {len(messages)} сообщений")
            return processed_ids, messages
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки checkpoint: {e}")
            return None, []
    
    def cleanup_checkpoint(self):
        """Удаление checkpoint после успешного завершения."""
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("🧹 Checkpoint файл удален")
        except Exception as e:
            logger.error(f"❌ Ошибка удаления checkpoint: {e}")
        
    async def collect_messages(self, limit_per_chat=1000, max_chats=None):
        """
        Основная функция сбора сообщений с поддержкой checkpoint.
        
        Args:
            limit_per_chat: Максимум сообщений на чат
            max_chats: Максимум чатов для обработки
        """
        logger.info("📝 Начинаем сбор сообщений...")
        
        # 🔄 Пытаемся восстановиться из checkpoint
        processed_chat_ids, _ = self.load_checkpoint()
        if processed_chat_ids:
            logger.info(f"⚡️ Продолжаем с checkpoint: пропускаем {len(processed_chat_ids)} уже обработанных чатов")
        else:
            processed_chat_ids = []
        
        try:
            # Получаем список диалогов
            dialogs_response = await self.client(GetDialogsRequest(
                offset_date=None,
                offset_id=0,
                offset_peer=InputPeerEmpty(),
                limit=200,
                hash=0
            ))
            
            dialogs = dialogs_response.dialogs
            logger.info(f"📋 Найдено диалогов: {len(dialogs)}")
            
            # Фильтруем только личные чаты
            personal_chats = []
            for dialog in dialogs:
                if hasattr(dialog.peer, 'user_id'):
                    # Пропускаем уже обработанные чаты
                    if dialog.peer.user_id not in processed_chat_ids:
                        personal_chats.append(dialog)
            
            logger.info(f"👥 Личных чатов для обработки: {len(personal_chats)}")
            
            # Ограничиваем количество чатов если нужно
            if max_chats:
                remaining_slots = max_chats - len(processed_chat_ids)
                if remaining_slots > 0:
                    personal_chats = personal_chats[:remaining_slots]
                    logger.info(f"📊 Ограничено до {len(personal_chats)} новых чатов")
                else:
                    logger.info("✅ Лимит чатов уже достигнут")
                    personal_chats = []
            
            # Обрабатываем каждый чат
            total_to_process = len(personal_chats)
            for i, dialog in enumerate(personal_chats, 1):
                try:
                    await self._process_chat(dialog, limit_per_chat, i, total_to_process)
                    
                    # 💾 Сохраняем checkpoint после каждого чата
                    processed_chat_ids.append(dialog.peer.user_id)
                    self.save_checkpoint(processed_chat_ids, self.all_messages)
                    
                    # Пауза между чатами для соблюдения лимитов API
                    if i < total_to_process:
                        await asyncio.sleep(2)
                        
                except FloodWaitError as e:
                    logger.warning(f"⏳ FloodWait: ждем {e.seconds} секунд...")
                    await asyncio.sleep(e.seconds + 1)
                except Exception as e:
                    logger.error(f"❌ Ошибка при обработке чата {i}: {e}")
                    continue
            
            await self._save_results()
            
            # 🧹 Удаляем checkpoint после успешного завершения
            self.cleanup_checkpoint()
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            logger.info("💾 Прогресс сохранен в checkpoint - можете перезапустить")
            raise
            
    async def _process_chat(self, dialog, limit_per_chat, chat_num, total_chats):
        """Обработка одного чата."""
        try:
            # Получаем информацию о пользователе
            user = await self.client.get_entity(dialog.peer.user_id)
            
            # Пропускаем ботов
            if user.bot:
                logger.info(f"🤖 Пропускаем бота: {user.first_name}")
                return
                
            logger.info(f"📱 [{chat_num}/{total_chats}] Обрабатываем: {user.first_name}")
            
            # Получаем сообщения
            messages = await self.client.get_messages(
                dialog.peer.user_id,
                limit=limit_per_chat
            )
            
            if not messages:
                logger.info(f"📭 Нет сообщений в чате с {user.first_name}")
                return
            
            # Обрабатываем сообщения
            chat_messages = []
            for msg in messages:
                if msg.message:  # Только текстовые сообщения
                    message_data = {
                        "id": msg.id,
                        "date": msg.date.isoformat(),
                        "from_id": msg.from_id.user_id if msg.from_id else None,
                        "from_name": user.first_name if msg.from_id and msg.from_id.user_id == user.id else "Я",
                        "text": msg.message,
                        "chat_with": user.first_name,
                        "chat_id": user.id,
                        "is_outgoing": msg.out,
                        "word_count": len(msg.message.split()),
                        "timestamp": int(msg.date.timestamp()),
                        "hour": msg.date.hour,
                        "day_of_week": msg.date.weekday(),
                        "month": msg.date.month
                    }
                    chat_messages.append(message_data)
            
            if chat_messages:
                self.all_messages.extend(chat_messages)
                self.stats["messages_collected"] += len(chat_messages)
                logger.info(f"✅ Собрано {len(chat_messages)} сообщений из чата с {user.first_name}")
            
            self.stats["chats_processed"] += 1
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке чата: {e}")
            
    async def _save_results(self):
        """Сохранение результатов."""
        self.stats["end_time"] = datetime.now()
        
        # Сортируем сообщения по дате
        self.all_messages.sort(key=lambda x: x['timestamp'])
        
        # Создаем уникальный timestamp для файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Основной файл с сообщениями
        messages_file = os.path.join(self.data_dir, f"all_messages_{timestamp}.json")
        with open(messages_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_messages, f, ensure_ascii=False, indent=2)
        
        # Метаданные анализа
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_messages": len(self.all_messages),
            "chats_processed": self.stats["chats_processed"],
            "collection_duration": str(self.stats["end_time"] - self.stats["start_time"]),
            "phone": self.phone,
            "unique_chats": len(set(msg['chat_id'] for msg in self.all_messages)),
            "date_range": {
                "earliest": min(msg['date'] for msg in self.all_messages) if self.all_messages else None,
                "latest": max(msg['date'] for msg in self.all_messages) if self.all_messages else None
            }
        }
        
        metadata_file = os.path.join(self.data_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Файл для анализа (только текст)
        analysis_text = "\n\n".join([
            f"[{msg['from_name']} -> {msg['chat_with']}]: {msg['text']}"
            for msg in self.all_messages
        ])
        
        analysis_file = os.path.join(self.data_dir, f"analysis_text_{timestamp}.txt")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        
        logger.info(f"💾 Результаты сохранены:")
        logger.info(f"   📄 Сообщения: {messages_file}")
        logger.info(f"   📊 Метаданные: {metadata_file}")
        logger.info(f"   📝 Текст для анализа: {analysis_file}")
        logger.info(f"✅ Собрано {len(self.all_messages)} сообщений из {self.stats['chats_processed']} чатов")
        
    async def disconnect(self):
        """Отключение от Telegram."""
        await self.client.disconnect()
        logger.info("🔐 Отключение от Telegram")

async def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Сбор сообщений из Telegram")
    parser.add_argument("--all-chats", action="store_true", help="Собрать все доступные чаты")
    parser.add_argument("--limit", type=int, default=1000, help="Лимит сообщений на чат")
    parser.add_argument("--max-chats", type=int, default=None, help="Максимум чатов для обработки")
    
    args = parser.parse_args()
    
    collector = TelegramMessageCollector(API_ID, API_HASH, PHONE, OUTPUT_DIR)
    
    try:
        await collector.start()
        
        if args.all_chats:
            # Собираем ВСЕ чаты без ограничений
            await collector.collect_messages(
                limit_per_chat=args.limit,
                max_chats=args.max_chats  # None = без ограничений
            )
        else:
            # Обычный режим с ограничениями для тестирования  
            await collector.collect_messages(
                limit_per_chat=args.limit,
                max_chats=args.max_chats or 30  # По умолчанию 30 для тестов
            )
            
    except KeyboardInterrupt:
        logger.info("⏹️ Прервано пользователем")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        await collector.disconnect()

if __name__ == "__main__":
    asyncio.run(main())