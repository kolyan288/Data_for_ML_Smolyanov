# ML Data Pipeline: Недвижимость Санкт-Петербурга

DEMO: https://drive.google.com/file/d/142MOLSdsmbAwpbKAEXX7FRWLGl8AwYyP/view?usp=sharing

Единый пайплайн из 4 агентов для сбора, очистки, разметки и умного отбора данных о недвижимости СПб. Запускается одним ноутбуком с точками Human-in-the-Loop.

---

## Структура репозитория

```
.
├── agents/
│   ├── data_collection_agent.py   # Агент 1: сбор данных
│   ├── data_quality_agent.py      # Агент 2: контроль качества
│   ├── annotation_agent.py        # Агент 3: авторазметка
│   └── al_agent.py                # Агент 4: active learning
├── data/
│   ├── raw/
│   │   └── raw_collected.csv      # Сырые данные после сбора
│   └── labeled/
│       ├── final_labeled_dataset.csv
│       └── data_card.md
├── data_pipeline_artifacts/       # Все графики и промежуточные файлы
├── models/
│   └── final_model.pkl            # Обученная модель
├── reports/
│   ├── annotation_spec.md         # Спецификация разметки
│   └── final_report.md            # Итоговый отчёт
├── review_queue.csv               # ❗ Файл для ручной проверки (HITL)
├── review_queue_corrected.csv     # Исправленные метки
├── data_pipeline.ipynb            # Главный ноутбук пайплайна
└── requirements.txt
```

---

## Установка и запуск

```bash
git clone <repository-url>
cd project
pip install -r requirements.txt
jupyter notebook data_pipeline.ipynb
```

Выполните ячейки последовательно. В точках HITL пайплайн создаёт `review_queue.csv` — откройте его, проверьте метки и сохраните как `review_queue_corrected.csv`.

---

## Логика пайплайна

```
DataCollectionAgent → DataQualityAgent → AnnotationAgent → [HITL] → ActiveLearningAgent
```

| Шаг | Агент | Что делает | Выход |
|-----|-------|------------|-------|
| 1 | DataCollectionAgent | Загружает данные из CSV | `raw_collected.csv` |
| 2 | DataQualityAgent | Детектирует и устраняет проблемы | очищенный DataFrame |
| ❗ | **HITL #1** | Подтверждение стратегии очистки | — |
| 3 | AnnotationAgent | Размечает по ценовым категориям | DataFrame с метками |
| ❗ | **HITL #2** | Проверка граничных случаев | `review_queue_corrected.csv` |
| 4 | ActiveLearningAgent | AL-цикл + обучение финальной модели | `final_model.pkl` |

---

## Агент 1: DataCollectionAgent

Загружает и стандартизирует данные из источников.

**Архитектура:**
```
DataCollectionAgent
├── skill: scrape(url, selector) → DataFrame
├── skill: fetch_api(endpoint, params) → DataFrame
├── skill: load_dataset(name, source) → DataFrame
└── skill: merge_datasets(sources) → DataFrame
```

**Использование:**
```python
from agents.data_collection_agent import DataCollectionAgent, demo_basic_usage

df_raw = demo_basic_usage()

agent = DataCollectionAgent()
agent.save(df_raw, 'data/raw/raw_collected.csv', format='csv')
```

**Унифицированная схема (`STANDARD_COLUMNS`):**

| Колонка | Тип | Описание |
|---------|-----|----------|
| `id` | string | Уникальный идентификатор |
| `text` | string | Текстовое описание объекта |
| `price` | int | Цена в рублях |
| `location` | string | Район города |
| `city` | string | Город |
| `area_sqm` | float | Общая площадь (кв.м) |
| `rooms` | int | Количество комнат |
| `floor` | int | Этаж |
| `building_type` | string | Тип здания |
| `build_year` | int | Год постройки |
| `label` | string | Целевая переменная |
| `source` | string | Источник данных |

**Статистика датасета:**

| Метрика | Значение |
|---------|----------|
| Записей | 5 000 |
| Город | Санкт-Петербург |
| Уникальных районов | 14 |

---

## Агент 2: DataQualityAgent

Детектирует и устраняет проблемы качества данных.

**Архитектура:**
```
DataQualityAgent
├── skill: detect_issues(df) → QualityReport
│   ├── missing values
│   ├── duplicates
│   ├── outliers (IQR / Z-score)
│   └── class imbalance
├── skill: fix(df, strategy) → DataFrame
│   ├── missing: 'drop' | 'mean' | 'median' | 'mode'
│   ├── duplicates: 'drop' | 'keep'
│   └── outliers: 'drop' | 'clip_iqr' | 'clip_zscore'
└── skill: compare(df_before, df_after) → ComparisonReport
```

**Использование:**
```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()

report = agent.detect_issues(df)

df_clean = agent.fix(df, strategy={
    'missing': {'area_sqm': 'median', 'location': 'mode'},
    'duplicates': 'drop',
    'outliers': {'price': 'clip_iqr', 'area_sqm': 'clip_iqr'}
})

comparison = agent.compare(df, df_clean)
agent.visualize_issues(df, output_dir='data_pipeline_artifacts')
```

**❗ HITL #1** — после `detect_issues()` человек просматривает отчёт и подтверждает стратегию очистки.

**Обнаруженные проблемы и выбранная стратегия:**

| Проблема | Стратегия |
|----------|-----------|
| Выбросы в `price`, `area_sqm` | `clip_iqr` |
| Пропуски в числовых колонках | `median` |
| Пропуски в категориальных колонках | `mode` |
| Дубликаты | `drop` |

**Обоснование выбора стратегии:** `clip_iqr` сохраняет все строки и ограничивает выбросы разумными границами без потери данных. Медиана устойчива к выбросам при заполнении пропусков.

---

## Агент 3: AnnotationAgent

Автоматически размечает данные по ценовым категориям.

**Архитектура:**
```
AnnotationAgent
├── skill: auto_label(df, modality) → DataFrame
├── skill: generate_spec(df, task) → AnnotationSpec (Markdown)
├── skill: check_quality(df_labeled) → QualityMetrics
└── skill: export_to_labelstudio(df) → JSON
```

**Использование:**
```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality='text', confidence_threshold=0.7)

# Сбрасываем исходные метки перед авторазметкой
df_clean['label'] = None

df_labeled = agent.auto_label(df_clean, text_column='text', price_column='price')

spec = agent.generate_spec(df_labeled, task='price_classification')
agent.save_spec('reports/annotation_spec.md')

metrics = agent.check_quality(df_labeled)
agent.export_to_labelstudio(df_labeled, 'labelstudio_import.json')
```

**Классы разметки:**

| Класс | Диапазон цен | Описание |
|-------|--------------|----------|
| **Эконом** | до 4 500 000 руб. | Эконом-класс |
| **Стандарт** | 4 500 000 — 7 000 000 руб. | Стандартный класс |
| **Комфорт** | 7 000 000 — 10 000 000 руб. | Комфорт-класс |
| **Премиум** | от 10 000 000 руб. | Премиум-класс |

**❗ HITL #2** — граничные случаи (цена ±5% от порогов) выгружаются в `review_queue.csv`. Человек открывает файл, проверяет колонку `label`, исправляет в `label_human` и сохраняет как `review_queue_corrected.csv`.

```
review_queue.csv  →  [ручная проверка]  →  review_queue_corrected.csv
```

---

## Агент 4: ActiveLearningAgent

Умный отбор наиболее информативных примеров и обучение финальной модели.

**Архитектура:**
```
ActiveLearningAgent
├── skill: fit(labeled_df) → model
├── skill: query(pool, strategy) → indices
│   ├── 'entropy'  — максимальная неопределённость
│   ├── 'margin'   — минимальный отрыв между топ-2 классами
│   └── 'random'   — случайная выборка (baseline)
├── skill: evaluate(labeled_df, test_df) → Metrics
├── skill: run_cycle(...) → history
└── skill: report(history) → learning_curve.png
```

**Использование:**
```python
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg', text_column='text', label_column='label')

history = agent.run_cycle(
    labeled_df=df_labeled_init,
    pool_df=df_pool,
    test_df=df_test,
    strategy='entropy',
    n_iterations=7,
    batch_size=30
)

agent.report(history, output_path='data_pipeline_artifacts/learning_curve.png')

savings = agent.analyze_savings(
    al_history=history_entropy,
    random_history=history_random,
    target_accuracy=0.90
)
```

**Поддерживаемые модели:**

| Модель | Описание |
|--------|----------|
| `logreg` | Логистическая регрессия (по умолчанию) |
| `rf` | Случайный лес |
| `gb` | Градиентный бустинг |

**Параметры эксперимента:**

| Параметр | Значение |
|----------|----------|
| Начальный labeled набор | 20% от train |
| Стратегия AL | entropy |
| Количество итераций | 7 |
| Размер батча | 30 |
| Финальная модель | Random Forest |

---

## Human-in-the-Loop

Пайплайн содержит 2 явные точки ручной проверки:

### HITL #1: Стратегия очистки данных
- **Когда:** после `DataQualityAgent.detect_issues()`
- **Что делает человек:** просматривает отчёт о пропусках, выбросах, дубликатах и подтверждает или корректирует стратегию очистки

### HITL #2: Проверка авторазметки
- **Когда:** после `AnnotationAgent.auto_label()`
- **Что делает человек:** открывает `review_queue.csv`, проверяет метки граничных случаев (цена ±5% от порогов классов), исправляет ошибки в колонке `label_human`
- **Инструкция:**
  1. Откройте `review_queue.csv` в Excel или Google Sheets
  2. Проверьте колонку `label` для каждой строки
  3. Если метка неверна — исправьте в `label_human`
  4. Добавьте комментарий в `human_comment` (опционально)
  5. Сохраните файл как `review_queue_corrected.csv`

---

## Артефакты пайплайна

| Файл | Описание |
|------|----------|
| `data_pipeline_artifacts/step1_collection_overview.png` | Обзор собранных данных |
| `data_pipeline_artifacts/step2_missing_values.png` | Пропущенные значения |
| `data_pipeline_artifacts/step2_outliers_boxplot.png` | Выбросы (box plots) |
| `data_pipeline_artifacts/step2_before_after.png` | Данные до/после очистки |
| `data_pipeline_artifacts/step3_annotation_results.png` | Результаты авторазметки |
| `data_pipeline_artifacts/step3_hitl_results.png` | Итоги HITL #2 |
| `data_pipeline_artifacts/step4_learning_curve.png` | Кривая обучения AL |
| `data_pipeline_artifacts/step4_al_vs_random.png` | AL vs Random baseline |
| `data_pipeline_artifacts/step6_final_dashboard.png` | Итоговый дашборд |
| `data/labeled/final_labeled_dataset.csv` | Финальный датасет |
| `data/labeled/data_card.md` | Data card датасета |
| `models/final_model.pkl` | Обученная модель |
| `reports/annotation_spec.md` | Спецификация разметки |
| `reports/final_report.md` | Итоговый отчёт |

---

## Требования

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
requests
beautifulsoup4
pyyaml
```

```bash
pip install -r requirements.txt
```

---

## Лицензия

MIT License
