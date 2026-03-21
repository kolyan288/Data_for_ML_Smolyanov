# AnnotationAgent — Автоматическая разметка данных

Агент для автоматической разметки данных о недвижимости с экспортом в LabelStudio.

## Архитектура агента

```
AnnotationAgent
├── skill: auto_label(df, modality) → DataFrame
│   └── text→правила на основе цены
├── skill: generate_spec(df, task) → AnnotationSpec (Markdown)
│   └── задача, классы, примеры, граничные случаи
├── skill: check_quality(df_labeled) → QualityMetrics
│   └── Cohen's κ, label distribution, confidence
└── skill: export_to_labelstudio(df) → JSON
```

## Структура репозитория

```
.
├── agents/
│   └── annotation_agent.py         # Основной файл агента
├── data/
│   └── raw/
│       └── prepared_data.csv       # Подготовленные данные (20% неразмечено)
├── specs/
│   └── annotation_spec.md          # Спецификация разметки
├── exports/
│   └── labelstudio_import.json     # Экспорт для LabelStudio
├── visualizations/                 # Визуализации
│   ├── label_distribution.png
│   └── annotation_comparison.png
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
from agents.annotation_agent import AnnotationAgent

# Инициализация агента
agent = AnnotationAgent(modality='text')

# 1. Автоматическая разметка
df_labeled = agent.auto_label(df, text_column='text', price_column='price')

# 2. Генерация спецификации
spec = agent.generate_spec(df_labeled, task='price_classification')
agent.save_spec('annotation_spec.md')

# 3. Проверка качества
metrics = agent.check_quality(df_labeled)
print(metrics)

# 4. Экспорт в LabelStudio
agent.export_to_labelstudio(df_labeled, 'labelstudio_import.json')
```

### Использование скиллов напрямую

```python
from agents.annotation_agent import auto_label, generate_spec, check_quality, export_to_labelstudio

# Авторазметка
df_labeled = auto_label(df, modality='text')

# Генерация спецификации
spec = generate_spec(df_labeled, task='classification')

# Проверка качества
metrics = check_quality(df_labeled)

# Экспорт
export_to_labelstudio(df_labeled, 'export.json')
```

## Задача разметки

**Тема:** Недвижимость (аренда и продажа) в Санкт-Петербурге

**Целевая переменная:** Ценовая категория квартиры

### Классы:

| Класс | Диапазон цен | Описание |
|-------|--------------|----------|
| **Эконом** | до 4.5M руб. | Недвижимость эконом-класса |
| **Стандарт** | 4.5M - 7M руб. | Стандартный класс |
| **Комфорт** | 7M - 10M руб. | Комфорт-класс |
| **Премиум** | от 10M руб. | Премиум-класс |

## Подготовка данных

Датасет содержит 35,471 запись. Для имитации реальных условий:
- 80% данных (28,377) — размечены
- 20% данных (7,094) — неразмечены (для авторазметки)

## Результаты

### Распределение меток после авторазметки:

| Класс | Количество | Процент |
|-------|------------|---------|
| Стандарт | 13,100 | 36.9% |
| Эконом | 8,502 | 24.0% |
| Комфорт | 7,942 | 22.4% |
| Премиум | 5,927 | 16.7% |

### Качество разметки:

- **Cohen's Kappa:** 1.000 (почти полное согласие с эталоном)
- **Agreement:** 100% (правила основаны на цене)

### Сравнение с ручной разметкой (симуляция):

При тестировании на выборке из 500 записей с имитацией ручной разметки (90% согласие):
- Совпадающие метки: 450 (90%)
- Несовпадающие метки: 50 (10%)

## Спецификация разметки

Спецификация содержит:
- **Задача:** Классификация недвижимости по ценовым категориям
- **Классы:** 4 класса с определениями и диапазонами цен
- **Примеры:** 3+ примера на каждый класс
- **Граничные случаи:** 4 сценария с рекомендациями
- **Руководство:** 5 правил для разметчиков

## Формат LabelStudio

Экспорт в формате LabelStudio JSON:

```json
[
  {
    "id": "spb_1",
    "data": {
      "text": "Продается 4-комнатная квартира..."
    },
    "annotations": [
      {
        "result": [
          {
            "value": {
              "choices": ["Премиум"]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]
  }
]
```

## Визуализации

- `label_distribution.png` — распределение меток
- `annotation_comparison.png` — сравнение авто и ручной разметки

## Требования

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Лицензия

MIT License