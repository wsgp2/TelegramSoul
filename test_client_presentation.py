#!/usr/bin/env python3
"""
Тестовые варианты представления результатов анализа для клиента
"""

# Данные из анализа (суммирование)
analysis_results = {
    "Инвестиции и бизнес": {
        "total_percentage": 268.0,
        "parts_found": 12,
        "average_percentage": 22.3
    },
    "Путешествия и туризм": {
        "total_percentage": 117.9,
        "parts_found": 5,
        "average_percentage": 23.6
    },
    "Личное развитие": {
        "total_percentage": 71.0,
        "parts_found": 4,
        "average_percentage": 17.8
    },
    "События и мероприятия": {
        "total_percentage": 56.0,
        "parts_found": 4,
        "average_percentage": 14.0
    },
    "Технологии": {
        "total_percentage": 45.0,
        "parts_found": 3,
        "average_percentage": 15.0
    },
    "Развлечения": {
        "total_percentage": 37.6,
        "parts_found": 2,
        "average_percentage": 18.8
    },
    "Партнерство": {
        "total_percentage": 5.0,
        "parts_found": 1,
        "average_percentage": 5.0
    }
}

def version_1_frequency_mentions():
    """Вариант 1: Частота упоминаний + интенсивность"""
    print("📊 ВАРИАНТ 1: ЧАСТОТА + ИНТЕНСИВНОСТЬ")
    print("=" * 50)
    
    sorted_topics = sorted(analysis_results.items(), key=lambda x: x[1]["total_percentage"], reverse=True)
    
    for topic, data in sorted_topics:
        frequency = data["parts_found"]
        intensity = data["average_percentage"]
        
        # Эмодзи для частоты
        if frequency >= 10:
            freq_emoji = "🔥🔥🔥"
        elif frequency >= 5:
            freq_emoji = "🔥🔥"
        elif frequency >= 3:
            freq_emoji = "🔥"
        else:
            freq_emoji = "💫"
            
        # Эмодзи для интенсивности  
        if intensity >= 20:
            intensity_emoji = "⚡⚡⚡"
        elif intensity >= 15:
            intensity_emoji = "⚡⚡"
        elif intensity >= 10:
            intensity_emoji = "⚡"
        else:
            intensity_emoji = "💡"
        
        print(f"🎯 {topic}")
        print(f"   {freq_emoji} Частота обсуждений: {frequency} раз")
        print(f"   {intensity_emoji} Интенсивность: {intensity:.1f}% за обсуждение")
        print()

def version_2_time_investment():
    """Вариант 2: Вложение времени"""
    print("⏰ ВАРИАНТ 2: ВРЕМЯ, УДЕЛЯЕМОЕ ТЕМАМ")
    print("=" * 50)
    
    sorted_topics = sorted(analysis_results.items(), key=lambda x: x[1]["total_percentage"], reverse=True)
    
    # Нормализуем для понятности (представим что общее время = 100 условных единиц)
    total_time = sum(data["total_percentage"] for _, data in analysis_results.items())
    
    for topic, data in sorted_topics:
        time_share = (data["total_percentage"] / total_time) * 100
        
        # Визуальная шкала
        bar_length = int(time_share / 5)  # 20 символов макс
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"📈 {topic}")
        print(f"   {bar} {time_share:.1f}% вашего времени")
        print(f"   📊 Обсуждается в {data['parts_found']} периодах")
        print()

def version_3_activity_score():
    """Вариант 3: Балльная система активности"""
    print("🏆 ВАРИАНТ 3: РЕЙТИНГ АКТИВНОСТИ ТЕМ")
    print("=" * 50)
    
    sorted_topics = sorted(analysis_results.items(), key=lambda x: x[1]["total_percentage"], reverse=True)
    
    # Создаем балльную систему (макс 100 баллов)
    max_total = max(data["total_percentage"] for _, data in analysis_results.items())
    
    for i, (topic, data) in enumerate(sorted_topics, 1):
        activity_score = (data["total_percentage"] / max_total) * 100
        
        # Медали и звезды
        if i == 1:
            medal = "🥇"
        elif i == 2:
            medal = "🥈"
        elif i == 3:
            medal = "🥉"
        else:
            medal = f"#{i}"
            
        # Звезды по баллам
        if activity_score >= 80:
            stars = "⭐⭐⭐⭐⭐"
        elif activity_score >= 60:
            stars = "⭐⭐⭐⭐"
        elif activity_score >= 40:
            stars = "⭐⭐⭐"
        elif activity_score >= 20:
            stars = "⭐⭐"
        else:
            stars = "⭐"
        
        print(f"{medal} {topic}")
        print(f"   {stars} Активность: {activity_score:.0f}/100 баллов")
        print(f"   🔄 Упоминается {data['parts_found']} раз")
        print()

def version_4_simple_ranking():
    """Вариант 4: Простой понятный рейтинг"""
    print("🎯 ВАРИАНТ 4: ПРОСТОЙ РЕЙТИНГ ВАЖНОСТИ")
    print("=" * 50)
    
    sorted_topics = sorted(analysis_results.items(), key=lambda x: x[1]["total_percentage"], reverse=True)
    
    for i, (topic, data) in enumerate(sorted_topics, 1):
        frequency = data["parts_found"]
        
        # Простое описание важности
        if frequency >= 10:
            importance = "ОЧЕНЬ ВАЖНАЯ"
            emoji = "🔥"
        elif frequency >= 5:
            importance = "ВАЖНАЯ"
            emoji = "⚡"
        elif frequency >= 3:
            importance = "ЗНАЧИМАЯ"
            emoji = "✨"
        else:
            importance = "ПЕРИОДИЧЕСКАЯ"
            emoji = "💡"
        
        print(f"{i}. {emoji} {topic}")
        print(f"   📌 Статус: {importance} тема")
        print(f"   📊 Обсуждается в {frequency} из 15 периодов ({frequency/15*100:.0f}% времени)")
        print(f"   💯 Глубина обсуждения: {data['average_percentage']:.1f}%")
        print()

def version_5_visual_dashboard():
    """Вариант 5: Визуальная панель"""
    print("📊 ВАРИАНТ 5: ВИЗУАЛЬНАЯ ПАНЕЛЬ")
    print("=" * 60)
    
    sorted_topics = sorted(analysis_results.items(), key=lambda x: x[1]["total_percentage"], reverse=True)
    
    print("🏆 ВАШ ПРОФИЛЬ ИНТЕРЕСОВ:")
    print("-" * 60)
    
    for topic, data in sorted_topics:
        frequency = data["parts_found"]
        intensity = data["average_percentage"]
        
        # Создаем визуальный индикатор
        freq_bar = "●" * frequency + "○" * (15 - frequency)
        
        # Категория по частоте
        if frequency >= 8:
            category = "ОСНОВНОЙ ИНТЕРЕС"
            color = "🔴"
        elif frequency >= 4:
            category = "РЕГУЛЯРНАЯ ТЕМА"
            color = "🟡"
        else:
            category = "ЭПИЗОДИЧЕСКИЙ"
            color = "🟢"
        
        print(f"{color} {topic}")
        print(f"   📈 {category}")
        print(f"   📊 Частота: {freq_bar} ({frequency}/15)")
        print(f"   ⚡ Интенсивность: {intensity:.1f}% при обсуждении")
        print()

if __name__ == "__main__":
    print("🎨 ТЕСТИРУЕМ ВАРИАНТЫ ПРЕДСТАВЛЕНИЯ КЛИЕНТУ")
    print("=" * 60)
    print()
    
    version_1_frequency_mentions()
    print("\n" + "="*60 + "\n")
    
    version_2_time_investment() 
    print("\n" + "="*60 + "\n")
    
    version_3_activity_score()
    print("\n" + "="*60 + "\n")
    
    version_4_simple_ranking()
    print("\n" + "="*60 + "\n")
    
    version_5_visual_dashboard()
    
    print("\n" + "="*60)
    print("🤔 КАКОЙ ВАРИАНТ ПОНЯТНЕЕ ДЛЯ КЛИЕНТА?")
    print("="*60) 