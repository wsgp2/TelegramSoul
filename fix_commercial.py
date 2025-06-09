import asyncio
import json
import os
from chatgpt_analyzer import ChatGPTAnalyzer

async def fix_commercial():
    print("🔧 Исправляем коммерческую оценку...")
    
    # Загружаем темы
    with open('data/reports/client_79957860591_topics_analysis_20250609_142324.json', 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    # Загружаем API ключи
    api_keys = []
    for i in range(1, 6):
        key = os.getenv(f'OPENAI_API_KEY{i}')
        if key:
            api_keys.append(key)
    
    if not api_keys:
        main_key = os.getenv('OPENAI_API_KEY')
        if main_key:
            api_keys = [main_key]
    
    print(f"📡 Используем {len(api_keys)} API ключей")
    
    # Создаем анализатор с API ключами
    analyzer = ChatGPTAnalyzer(api_keys=api_keys)
    
    # Запускаем умную оценку с GPT-4o
    assessment = await analyzer.assess_commercial_potential(topics)
    
    # Сохраняем исправленную оценку
    with open('data/reports/client_79957860591_commercial_assessment_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(assessment, f, ensure_ascii=False, indent=4)
    
    print('✅ Коммерческая оценка исправлена!')
    
    # Генерируем новый готовый отчет
    report = analyzer.generate_comprehensive_client_report(topics, assessment, "client_79957860591")
    with open('data/reports/client_79957860591_ГОТОВЫЙ_ОТЧЕТ_ИСПРАВЛЕННЫЙ.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print('✅ Готовый отчет обновлен!')

if __name__ == "__main__":
    asyncio.run(fix_commercial()) 