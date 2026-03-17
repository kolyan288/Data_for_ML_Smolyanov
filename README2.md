# DataQualityAgent — «Детектив данных»

Агент-детектив для автоматического обнаружения и устранения проблем качества данных.

## Архитектура агента

```
DataQualityAgent
├── skill: detect_issues(df) → QualityReport
│   ├── missing values
│   ├── duplicates
│   ├── outliers (IQR / Z-score)
│   └── class imbalance
├── skill: fix(df, strategy: dict) → DataFrame
│   ├── missing: 'drop' | 'mean' | 'median' | 'mode'
│   ├── duplicates: 'drop' | 'keep'
│   └── outliers: 'drop' | 'clip_iqr' | 'clip_zscore'
└── skill: compare(df_before, df_after) → ComparisonReport
```

## Структура репозитория

```
.
├── agents/
│   └── data_quality_agent.py       # Основной файл агента
├── data/
│   ├── raw/                        # Исходные данные
│   └── processed/                  # Очищенные данные
├── visualizations/                 # Визуализации проблем
│   ├── part1_missing_values.png
│   ├── part1_outliers_boxplot.png
│   ├── part1_class_imbalance.png
│   └── part2_strategies_comparison.png
├── notebooks/
│   └── data_quality_analysis.ipynb # Jupyter notebook с анализом
├── README.md                       # Этот файл
└── requirements.txt                # Зависимости
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Базовое использование

```python
from agents.data_quality_agent import DataQualityAgent

# Инициализация агента
agent = DataQualityAgent()

# 1. Обнаружение проблем
report = agent.detect_issues(df)
print(report)

# 2. Исправление проблем
df_clean = agent.fix(df, strategy={
    'missing': 'median',
    'duplicates': 'drop',
    'outliers': 'clip_iqr'
})

# 3. Сравнение до/после
comparison = agent.compare(df, df_clean)
print(comparison)

# 4. Визуализация
agent.visualize_issues(df, output_dir='visualizations')
```

### Использование скиллов напрямую

```python
from agents.data_quality_agent import detect_issues, fix_data, compare_data

# Обнаружение проблем
report = detect_issues(df)

# Исправление
df_clean = fix_data(df, strategy={
    'missing': 'median',
    'outliers': 'clip_iqr'
})

# Сравнение
comparison = compare_data(df, df_clean)
```

## Три части задания

### Часть 1: Детектив — Обнаружение проблем

Обнаружены следующие проблемы в датасете:

| Проблема | Количество | Процент |
|----------|------------|---------|
| Пропущенные значения | 15,955 | 2.14% |
| Дубликаты | 0 | 0% |
| Выбросы (IQR) | 9,366 | 26.4% |

**Пропущенные значения:**
- `living_area_sqm`: 6,383 (17.99%)
- `kitchen_area_sqm`: 9,572 (26.99%)

**Выбросы по колонкам:**
- `price`: 2,156 (6.08%)
- `area_sqm`: 1,304 (3.68%)
- `build_year`: 3,133 (8.83%)

**Дисбаланс классов:**
- `label`: Продажа 97.9% vs Аренда 2.1% (критический)
- `city`: Москва 85.9% vs СПб 14.1% (высокий)

### Часть 2: Хирург — Стратегии чистки

| Стратегия | Описание | Строк | Пропуски | Выбросы | Средняя цена | Std |
|-----------|----------|-------|----------|---------|--------------|-----|
| Исходные | — | 35,471 | 15,955 | 9,366 | 7.09M | 4.92M |
| Стратегия 1 | Консервативная (median) | 35,471 | 0 | 10,662 | 7.09M | 4.92M |
| **Стратегия 2** | **Умеренная (median + clip_iqr)** | **35,471** | **0** | **0** | **6.71M** | **3.39M** |
| Стратегия 3 | Агрессивная (median + drop) | 27,978 | 0 | 7,198 | 6.10M | 2.81M |

### Часть 3: Аргумент — Обоснование выбора

**Рекомендуемая стратегия: Стратегия 2 (Умеренная)**

```python
strategy = {
    'missing': 'median',
    'duplicates': 'none',
    'outliers': 'clip_iqr'
}
```

**Обоснование:**

1. **Сохранение данных**: Не теряем 21% данных (как в стратегии 3)
2. **Управление выбросами**: Метод `clip_iqr` ограничивает выбросы разумными границами
3. **Стабилизация**: Снижение Std на 31% (с 4.92M до 3.39M)
4. **Реалистичность**: Максимальная цена 14.15M вместо 111M
5. **Робастность**: Медиана устойчива к выбросам при заполнении пропусков

## Визуализации

Все визуализации сохранены в `visualizations/`:
- `part1_missing_values.png` — пропущенные значения
- `part1_outliers_boxplot.png` — выбросы (box plots)
- `part1_class_imbalance.png` — дисбаланс классов
- `part2_strategies_comparison.png` — сравнение стратегий

## Требования

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Лицензия

MIT License