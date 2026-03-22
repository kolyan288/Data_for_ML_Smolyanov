Агент 3

Необходимо написать агента, который автоматически размечает данные по теме @ТЕМА@, генерирует спецификацию разметки, оценивает качество и умеет экспортировать задачи для ручной доразметки в LabelStudio. Агент войдёт в пайплайн как auto_label_op. Если при анализе датасета ты увидишь, что все данные размечены, убери 20% из столбца целевого (target) признака @TARGET@, чтобы таким образом создать неразмеченные данные

АРХИТЕКТУРА АГЕНТА

- AnnotationAgent
- skill: auto_label(df, modality) → DataFrame (text→spaCy/zero-shot, audio→Whisper, image→YOLO)
- skill: generate_spec(df, task) → AnnotationSpec (Markdown-файл)
- skill: check_quality(df_labeled) → QualityMetrics (Cohen's κ, label distribution, confidence)
- skill: export_to_labelstudio(df) → JSON в формате LabelStudio import

ТЕХНИЧЕСКИЙ КОНТРАКТ

from annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality='text')
df_labeled = agent.auto_label(df)

spec = agent.generate_spec(df, task='sentiment_classification')
# → annotation_spec.md: задача, классы, примеры, граничные случаи

metrics = agent.check_quality(df_labeled)
# → {'kappa': 0.72, 'label_dist': {...}, 'confidence_mean': 0.85}

agent.export_to_labelstudio(df_labeled)  # → labelstudio_import.json

ТРЕБОВАНИЯ

- auto_label работает хотя бы для одной модальности (text / audio / image)
- Спецификация содержит: задача, классы с определениями, 3+ примера на класс, граничные случаи
- Экспорт в формат LabelStudio — JSON должен загружаться без ошибок
- Метрики качества: Cohen's κ или % agreement + распределение меток
- Спецификацию передать однокурснику → он размечает → сравнить его разметку с авторазметкой

Напоминаю, что @TEMA@ == недвижимость (аренда и продажа) в Санкт-Петербурге
@TARGET@ == Цена за квартиру (за аренду квартиры или за покупку)