#!/usr/bin/env python3
"""
Тестовый скрипт для сравнения методов агрегации процентов
"""
import json

# Данные из последнего анализа (извлечены из логов)
raw_topics_data = [
    # Часть 1
    {
        "name": "Любовь и честность",
        "percentage": 20.0
    },
    {
        "name": "Монетизация и доход", 
        "percentage": 18.0
    },
    {
        "name": "Личное время и забота о детях",
        "percentage": 11.0
    },
    
    # Часть 2  
    {
        "name": "Инвестиции и монетизация",
        "percentage": 25.0
    },
    {
        "name": "Сетевые мероприятия и встречи",
        "percentage": 5.0
    },
    
    # Часть 3
    {
        "name": "Путешествия и мероприятия",
        "percentage": 15.0
    },
    {
        "name": "Бизнес и проекты",
        "percentage": 12.0
    },
    
    # Часть 4
    {
        "name": "Торговля криптовалютами", 
        "percentage": 35.0
    },
    {
        "name": "События и нетворкинг",
        "percentage": 20.0
    },
    
    # Часть 5
    {
        "name": "Инвестиции и криптовалюты",
        "percentage": 25.0
    },
    {
        "name": "Партнерские программы и предложения",
        "percentage": 5.0
    },
    
    # Часть 6
    {
        "name": "Туризм и путешествия",
        "percentage": 25.0
    },
    {
        "name": "Бизнес и инвестиции",
        "percentage": 20.0
    },
    
    # Часть 7
    {
        "name": "Выход на зарубежный рынок",
        "percentage": 25.0
    },
    {
        "name": "Искусственный интеллект и технологии",
        "percentage": 15.0
    },
    
    # Часть 8
    {
        "name": "Сексуальность и энергетический обмен",
        "percentage": 25.0
    },
    {
        "name": "Путешествия и отдых",
        "percentage": 20.0
    },
    
    # Часть 9
    {
        "name": "Финансовые обсуждения и инвестиции",
        "percentage": 20.0
    },
    {
        "name": "Социальные взаимодействия и шутки",
        "percentage": 9.0
    },
    
    # Часть 10
    {
        "name": "Инвестиции и финансовые стратегии",
        "percentage": 25.0
    },
    {
        "name": "Технические аспекты и инструменты",
        "percentage": 5.0
    },
    
    # Часть 11
    {
        "name": "Монетизация и бизнес-идеи",
        "percentage": 25.0
    },
    {
        "name": "Личные переживания и эмоции",
        "percentage": 15.0
    },
    
    # Часть 12
    {
        "name": "Деловые мероприятия и тренинги",
        "percentage": 20.0
    },
    {
        "name": "Социальные мероприятия и вечеринки",
        "percentage": 11.0
    },
    
    # Часть 13
    {
        "name": "Криптовалюты и инвестиции",
        "percentage": 25.0
    },
    {
        "name": "Жилищные вопросы на Бали",
        "percentage": 15.0
    },
    
    # Часть 14
    {
        "name": "Акселерация стартапов",
        "percentage": 20.0
    },
    {
        "name": "Поток стартапов и инвестиции",
        "percentage": 18.0
    },
    
    # Часть 15
    {
        "name": "Путешествия и туризм",
        "percentage": 42.9
    },
    {
        "name": "Видеочаты и онлайн-игры",
        "percentage": 28.6
    }
]

def aggregate_topics_by_similarity_current_method(topics):
    """Текущий метод - суммирование процентов"""
    print("🔢 ТЕКУЩИЙ МЕТОД (СУММИРОВАНИЕ):")
    
    # Группируем похожие темы (упрощенная логика)
    groups = {
        "Путешествия и туризм": ["Путешествия и мероприятия", "Туризм и путешествия", "Путешествия и отдых", "Путешествия и туризм", "Жилищные вопросы на Бали"],
        "Инвестиции и бизнес": ["Монетизация и доход", "Инвестиции и монетизация", "Бизнес и проекты", "Торговля криптовалютами", "Инвестиции и криптовалюты", "Бизнес и инвестиции", "Финансовые обсуждения и инвестиции", "Инвестиции и финансовые стратегии", "Монетизация и бизнес-идеи", "Криптовалюты и инвестиции", "Акселерация стартапов", "Поток стартапов и инвестиции"],
        "События и мероприятия": ["Сетевые мероприятия и встречи", "События и нетворкинг", "Деловые мероприятия и тренинги", "Социальные мероприятия и вечеринки"],
        "Личное развитие": ["Любовь и честность", "Личное время и забота о детях", "Сексуальность и энергетический обмен", "Личные переживания и эмоции"],
        "Технологии": ["Выход на зарубежный рынок", "Искусственный интеллект и технологии", "Технические аспекты и инструменты"],
        "Партнерство": ["Партнерские программы и предложения"],
        "Развлечения": ["Социальные взаимодействия и шутки", "Видеочаты и онлайн-игры"]
    }
    
    result_current = {}
    
    for group_name, topic_names in groups.items():
        total_percentage = 0
        count = 0
        
        for topic in topics:
            if topic["name"] in topic_names:
                total_percentage += topic["percentage"]
                count += 1
        
        if count > 0:
            result_current[group_name] = {
                "percentage": total_percentage,
                "parts_found": count,
                "method": "суммирование"
            }
    
    # Сортируем по убыванию
    sorted_current = sorted(result_current.items(), key=lambda x: x[1]["percentage"], reverse=True)
    
    for topic_name, data in sorted_current:
        print(f"  {topic_name}: {data['percentage']:.1f}% (найдено в {data['parts_found']} частях)")
    
    return result_current

def aggregate_topics_by_similarity_new_method(topics):
    """Новый метод - усреднение процентов"""
    print("\n📊 НОВЫЙ МЕТОД (УСРЕДНЕНИЕ):")
    
    # Те же группы
    groups = {
        "Путешествия и туризм": ["Путешествия и мероприятия", "Туризм и путешествия", "Путешествия и отдых", "Путешествия и туризм", "Жилищные вопросы на Бали"],
        "Инвестиции и бизнес": ["Монетизация и доход", "Инвестиции и монетизация", "Бизнес и проекты", "Торговля криптовалютами", "Инвестиции и криптовалюты", "Бизнес и инвестиции", "Финансовые обсуждения и инвестиции", "Инвестиции и финансовые стратегии", "Монетизация и бизнес-идеи", "Криптовалюты и инвестиции", "Акселерация стартапов", "Поток стартапов и инвестиции"],
        "События и мероприятия": ["Сетевые мероприятия и встречи", "События и нетворкинг", "Деловые мероприятия и тренинги", "Социальные мероприятия и вечеринки"],
        "Личное развитие": ["Любовь и честность", "Личное время и забота о детях", "Сексуальность и энергетический обмен", "Личные переживания и эмоции"],
        "Технологии": ["Выход на зарубежный рынок", "Искусственный интеллект и технологии", "Технические аспекты и инструменты"],
        "Партнерство": ["Партнерские программы и предложения"],
        "Развлечения": ["Социальные взаимодействия и шутки", "Видеочаты и онлайн-игры"]
    }
    
    result_new = {}
    
    for group_name, topic_names in groups.items():
        total_percentage = 0
        count = 0
        
        for topic in topics:
            if topic["name"] in topic_names:
                total_percentage += topic["percentage"]
                count += 1
        
        if count > 0:
            # УСРЕДНЕНИЕ вместо суммирования
            average_percentage = total_percentage / count
            result_new[group_name] = {
                "percentage": average_percentage,
                "parts_found": count,
                "method": "усреднение",
                "total_raw": total_percentage
            }
    
    # Сортируем по убыванию
    sorted_new = sorted(result_new.items(), key=lambda x: x[1]["percentage"], reverse=True)
    
    for topic_name, data in sorted_new:
        print(f"  {topic_name}: {data['percentage']:.1f}% (среднее из {data['parts_found']} частей, сумма была {data['total_raw']:.1f}%)")
    
    return result_new

def compare_methods():
    """Сравнение методов"""
    print("=" * 60)
    print("🧮 СРАВНЕНИЕ МЕТОДОВ АГРЕГАЦИИ ПРОЦЕНТОВ")
    print("=" * 60)
    
    current = aggregate_topics_by_similarity_current_method(raw_topics_data)
    new = aggregate_topics_by_similarity_new_method(raw_topics_data)
    
    print("\n" + "=" * 60)
    print("📈 СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
    print("=" * 60)
    print(f"{'Тема':<25} {'Суммирование':<15} {'Усреднение':<15} {'Разница':<10}")
    print("-" * 65)
    
    all_topics = set(current.keys()) | set(new.keys())
    
    for topic in all_topics:
        current_pct = current.get(topic, {}).get('percentage', 0)
        new_pct = new.get(topic, {}).get('percentage', 0)
        diff = new_pct - current_pct
        
        print(f"{topic:<25} {current_pct:<15.1f} {new_pct:<15.1f} {diff:<+10.1f}")
    
    print("\n" + "=" * 60)
    print("🤔 ВЫВОДЫ:")
    print("=" * 60)
    
    # Анализ результатов
    current_top3 = list(sorted(current.items(), key=lambda x: x[1]["percentage"], reverse=True))[:3]
    new_top3 = list(sorted(new.items(), key=lambda x: x[1]["percentage"], reverse=True))[:3]
    
    print("📊 ТОП-3 с суммированием:")
    for i, (name, data) in enumerate(current_top3, 1):
        print(f"  {i}. {name}: {data['percentage']:.1f}%")
    
    print("\n📊 ТОП-3 с усреднением:")
    for i, (name, data) in enumerate(new_top3, 1):
        print(f"  {i}. {name}: {data['percentage']:.1f}%")
    
    print(f"\n🔍 Изменится ли порядок тем? {'ДА' if [x[0] for x in current_top3] != [x[0] for x in new_top3] else 'НЕТ'}")
    
    # Проверяем максимальный процент
    max_current = max(current.values(), key=lambda x: x['percentage'])['percentage']
    max_new = max(new.values(), key=lambda x: x['percentage'])['percentage']
    
    print(f"📈 Максимальный процент:")
    print(f"  Суммирование: {max_current:.1f}%")
    print(f"  Усреднение: {max_new:.1f}%")
    print(f"  Разница: {max_new - max_current:+.1f}%")

if __name__ == "__main__":
    compare_methods() 