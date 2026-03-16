# DataCollectionAgent: Недвижимость Санкт-Петербурга и Москвы

Агент для сбора и унификации данных о недвижимости из различных источников.

## Описание задачи ML

**Задача**: Предсказание цен на недвижимость в Санкт-Петербурге и Москве

**Тип задачи**: Регрессия

**Целевая переменная**: `price` — цена недвижимости в рублях

**Признаки**:
- `location` — район города
- `area_sqm` — общая площадь (кв.м)
- `rooms` — количество комнат
- `floor` — этаж
- `total_floors` — этажность дома
- `building_type` — тип здания
- `build_year` — год постройки

## Архитектура агента

```
DataCollectionAgent
├── skill: scrape(url, selector) → DataFrame
├── skill: fetch_api(endpoint, params) → DataFrame
├── skill: load_dataset(name, source='hf'|'kaggle') → DataFrame
└── skill: merge(sources: list[DataFrame]) → DataFrame
```

## Структура репозитория

```
.
├── agents/
│   └── data_collection_agent.py    # Основной файл агента
├── config.yaml                      # Конфигурация источников
├── notebooks/
│   └── eda.ipynb                    # EDA и визуализации
├── data/
│   ├── raw/                         # Собранные данные
│   │   ├── unified_real_estate.csv  # Объединенный датасет
│   │   ├── spb_real_estate.csv      # Данные по СПб
│   │   └── moscow_real_estate.csv   # Данные по Москве
│   └── processed/                   # Обработанные данные
├── README.md                        # Этот файл
└── requirements.txt                 # Зависимости
```

## Источники данных

### 1. Kaggle: Sberbank Russian Housing Market
- **URL**: https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-20182021
- **Описание**: Датасет содержит информацию о недвижимости в Москве и Московской области
- **Записей**: 30,471
- **Источник**: Kaggle

### 2. Синтетические данные: Санкт-Петербург
- **Описание**: Сгенерированные данные на основе реальных паттернов рынка недвижимости СПб
- **Записей**: 5,000
- **Источник**: synthetic:spb_real_estate

## Унифицированная схема данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| `id` | string | Уникальный идентификатор |
| `text` | string | Текстовое описание объекта |
| `price` | int | Цена в рублях |
| `price_currency` | string | Валюта цены (RUB) |
| `location` | string | Район города |
| `city` | string | Город |
| `property_type` | string | Тип недвижимости |
| `area_sqm` | float | Общая площадь (кв.м) |
| `living_area_sqm` | float | Жилая площадь (кв.м) |
| `kitchen_area_sqm` | float | Площадь кухни (кв.м) |
| `rooms` | int | Количество комнат |
| `floor` | int | Этаж |
| `total_floors` | int | Этажность дома |
| `building_type` | string | Тип здания |
| `build_year` | int | Год постройки |
| `label` | string | Тип сделки (Продажа/Аренда) |
| `source` | string | Источник данных |
| `collected_at` | string | Время сбора данных |
| `url` | string | Ссылка на источник |
| `date_posted` | string | Дата публикации |
| `metadata` | string | Дополнительные метаданные |

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd data_collection_agent
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Использование агента

```python
from agents.data_collection_agent import DataCollectionAgent

# Инициализация агента
agent = DataCollectionAgent(config='config.yaml')

# Определение источников данных
sources = [
    {
        'type': 'csv',
        'path': 'data/raw/spb_real_estate.csv'
    },
    {
        'type': 'csv', 
        'path': 'data/raw/moscow_real_estate.csv'
    }
]

# Сбор данных
df = agent.run(sources)

# Сохранение результатов
agent.save(df, 'data/processed/output.csv', format='csv')
```

### 4. Использование скиллов напрямую

```python
from agents.data_collection_agent import load_dataset, scrape, fetch_api, merge_datasets

# Загрузка датасета из HuggingFace
df_hf = load_dataset('imdb', source='hf')

# Веб-скрапинг
df_scraped = scrape('https://example.com', selector='.listing')

# API запрос
df_api = fetch_api('https://api.example.com/data', params={'limit': 100})

# Объединение датасетов
df_merged = merge_datasets([df_hf, df_scraped, df_api])
```

## EDA (Exploratory Data Analysis)

Для проведения разведочного анализа данных откройте ноутбук:

```bash
jupyter notebook notebooks/eda.ipynb
```

### Основные выводы EDA:

1. **Цены**: Москва имеет более высокие цены на недвижимость по сравнению со Санкт-Петербургом
2. **Площадь**: Распределение площади схоже в обоих городах, преобладают квартиры 35-80 кв.м
3. **Комнаты**: Наиболее популярны 1- и 2-комнатные квартиры
4. **Районы**: В СПб лидируют по количеству объявлений Центральный, Адмиралтейский и Василеостровский районы
5. **Типы зданий**: Преобладают монолитные и кирпичные дома

## Статистика датасета

| Метрика | Значение |
|---------|----------|
| Всего записей | 35,471 |
| Санкт-Петербург | 5,000 (14.1%) |
| Москва | 30,471 (85.9%) |
| Продажа | 34,728 (97.9%) |
| Аренда | 743 (2.1%) |
| Уникальных районов (СПб) | 14 |
| Уникальных районов (Москва) | 146 |

## Требования

- Python 3.8+
- pandas
- numpy
- requests
- beautifulsoup4
- pyyaml
- datasets (для HuggingFace)
- kaggle (для Kaggle API)

## Лицензия

MIT License

## Автор

DataCollectionAgent Project
