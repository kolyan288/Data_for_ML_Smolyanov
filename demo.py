"""
Демонстрационный скрипт для DataCollectionAgent.

Показывает примеры использования агента для сбора данных о недвижимости.
"""

import sys
sys.path.insert(0, '.')

from agents.data_collection_agent import DataCollectionAgent, load_dataset, merge_datasets
import pandas as pd


def demo_basic_usage():
    """Демонстрация базового использования агента."""
    print("=" * 60)
    print("ДЕМО 1: Базовое использование DataCollectionAgent")
    print("=" * 60)
    
    # Инициализация агента
    agent = DataCollectionAgent(config='config.yaml')
    
    # Определение источников данных
    sources = [
        {
            'type': 'csv',
            'path': 'data/raw/spb_real_estate.csv',
            'description': 'Синтетические данные по недвижимости СПб'
        },
        {
            'type': 'csv',
            'path': 'data/raw/moscow_real_estate.csv',
            'description': 'Данные по недвижимости Москвы от Sberbank'
        }
    ]
    
    # Сбор данных
    print("\nСбор данных из источников...")
    df = agent.run(sources)
    
    print(f"\n✓ Собрано {len(df)} записей")
    print(f"✓ Источников: {df['source'].nunique()}")
    print(f"✓ Городов: {df['city'].nunique()}")
    
    # Вывод статистики
    print("\n--- Статистика по городам ---")
    print(df.groupby('city').agg({
        'price': ['count', 'mean', 'median'],
        'area_sqm': 'mean'
    }).round(2))
    
    return df


def demo_skills():
    """Демонстрация использования скиллов напрямую."""
    print("\n" + "=" * 60)
    print("ДЕМО 2: Использование скиллов напрямую")
    print("=" * 60)
    
    # Скилл: load_dataset
    print("\n--- Скилл: load_dataset ---")
    print("Загрузка локального CSV файла...")
    df_local = load_dataset('data/raw/spb_real_estate.csv', source='csv')
    print(f"✓ Загружено {len(df_local)} записей")
    
    # Скилл: merge_datasets
    print("\n--- Скилл: merge_datasets ---")
    df_spb = load_dataset('data/raw/spb_real_estate.csv', source='csv')
    df_moscow = load_dataset('data/raw/moscow_real_estate.csv', source='csv')
    
    print(f"Объединение {len(df_spb)} + {len(df_moscow)} записей...")
    df_merged = merge_datasets([df_spb, df_moscow])
    print(f"✓ Объединено в {len(df_merged)} записей")
    
    return df_merged


def demo_data_analysis(df):
    """Демонстрация анализа данных."""
    print("\n" + "=" * 60)
    print("ДЕМО 3: Анализ собранных данных")
    print("=" * 60)
    
    # 1. Распределение по типу сделки
    print("\n--- Распределение по типу сделки ---")
    deal_counts = df['label'].value_counts()
    for deal_type, count in deal_counts.items():
        pct = count / len(df) * 100
        print(f"  {deal_type}: {count} ({pct:.1f}%)")
    
    # 2. Статистика цен
    print("\n--- Статистика цен ---")
    price_stats = df.groupby('city')['price'].agg(['min', 'median', 'mean', 'max'])
    for city, stats in price_stats.iterrows():
        print(f"\n{city}:")
        print(f"  Минимальная: {stats['min']:,.0f} руб.")
        print(f"  Медианная: {stats['median']:,.0f} руб.")
        print(f"  Средняя: {stats['mean']:,.0f} руб.")
        print(f"  Максимальная: {stats['max']:,.0f} руб.")
    
    # 3. Распределение по комнатам
    print("\n--- Распределение по комнатам ---")
    room_dist = df['rooms'].value_counts().sort_index().head(5)
    for rooms, count in room_dist.items():
        print(f"  {int(rooms)} комнат: {count} объявлений")
    
    # 4. Топ районов по количеству объявлений (СПб)
    print("\n--- Топ-5 районов СПб по количеству объявлений ---")
    df_spb = df[df['city'] == 'Санкт-Петербург']
    top_districts = df_spb['location'].value_counts().head(5)
    for district, count in top_districts.items():
        print(f"  {district}: {count} объявлений")
    
    # 5. Корреляция цены с площадью
    print("\n--- Корреляция признаков с ценой ---")
    corr_price = df[['price', 'area_sqm', 'rooms', 'floor']].corr()['price'].sort_values(ascending=False)
    for feature, corr in corr_price.items():
        if feature != 'price':
            print(f"  {feature}: {corr:.3f}")


def demo_save_data(df):
    """Демонстрация сохранения данных."""
    print("\n" + "=" * 60)
    print("ДЕМО 4: Сохранение данных")
    print("=" * 60)
    
    agent = DataCollectionAgent()
    
    # Сохранение в разных форматах
    formats = ['csv', 'json']
    
    for fmt in formats:
        output_path = f'data/processed/demo_output.{fmt}'
        print(f"\nСохранение в формате {fmt.upper()}...")
        agent.save(df.head(5000), output_path, format=fmt)
        print(f"✓ Сохранено: {output_path}")


def main():
    """Главная функция демонстрации."""
    print("\n" + "=" * 60)
    print("DataCollectionAgent - Демонстрация")
    print("Сбор данных о недвижимости в Санкт-Петербурге и Москве")
    print("=" * 60)
    
    try:
        # Демо 1: Базовое использование
        df = demo_basic_usage()
        
        # Демо 2: Использование скиллов
        df_merged = demo_skills()
        
        # Демо 3: Анализ данных
        demo_data_analysis(df)
        
        # Демо 4: Сохранение данных
        demo_save_data(df)
        
        print("\n" + "=" * 60)
        print("Демонстрация завершена успешно!")
        print("=" * 60)
        print("\nСгенерированные файлы:")
        print("  - data/processed/demo_output.csv")
        print("  - data/processed/demo_output.json")
        print("  - data/processed/*.png (визуализации)")
        
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
