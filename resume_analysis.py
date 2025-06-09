#!/usr/bin/env python3
"""
Скрипт для восстановления анализа после обрыва
"""
import asyncio
import os
import json

def check_and_resume():
    """Проверяет наличие checkpoint'ов и предлагает восстановление"""
    
    checkpoints_dir = "data/reports"
    checkpoints = []
    
    # Ищем checkpoint файлы
    if os.path.exists(checkpoints_dir):
        for file in os.listdir(checkpoints_dir):
            if file.endswith("_checkpoint.json"):
                checkpoint_path = os.path.join(checkpoints_dir, file)
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    checkpoints.append({
                        'file': file,
                        'path': checkpoint_path,
                        'data': data
                    })
                except Exception as e:
                    print(f"❌ Ошибка чтения checkpoint {file}: {e}")
    
    if not checkpoints:
        print("🟢 Checkpoint'ы не найдены. Запускаем полный анализ...")
        os.system("python run_analysis.py")
        return
    
    print("🔍 Найдены незавершенные анализы:")
    for i, cp in enumerate(checkpoints):
        data = cp['data']
        progress = (data.get('last_processed_chunk', 0) + 1) / data.get('total_chunks', 1) * 100
        print(f"  {i+1}. {cp['file']}")
        print(f"     📊 Прогресс: {progress:.1f}% ({data.get('last_processed_chunk', 0) + 1}/{data.get('total_chunks', 0)} частей)")
        print(f"     ⏰ Время: {data.get('timestamp', 'неизвестно')}")
        print()
    
    choice = input("Выберите действие:\n1. Восстановить анализ\n2. Начать заново\n3. Только исправить коммерческую оценку\nВведите номер (1-3): ").strip()
    
    if choice == "1":
        print("🔄 Восстанавливаем анализ...")
        os.system("python run_analysis.py")
    elif choice == "2":
        print("🗑️ Удаляем checkpoint'ы и начинаем заново...")
        for cp in checkpoints:
            os.remove(cp['path'])
        os.system("python run_analysis.py")
    elif choice == "3":
        print("🔧 Исправляем только коммерческую оценку...")
        os.system("python fix_commercial.py")
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    check_and_resume() 