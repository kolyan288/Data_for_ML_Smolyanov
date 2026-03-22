
Агент 4

Необходимо построить агента для умного отбора данных (Active Learning) по теме @ТЕМА@. Агент войдёт в финальный пайплайн как active_learning_op.

АРХИТЕКТУРА АГЕНТА

- ActiveLearningAgent
- skill: fit(labeled_df) → model (обучить базовую модель)
- skill: query(pool, strategy) → indices (стратегии: 'entropy', 'margin', 'random')
- skill: evaluate(labeled_df, test_df) → Metrics (accuracy, F1)
- skill: report(history) → LearningCurve (график quality vs. n_labeled)

ТЕХНИЧЕСКИЙ КОНТРАКТ

from al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')

# Цикл: старт с N=50, 5 итераций по 20 примеров
history = agent.run_cycle(
    labeled_df=df_labeled_50,
    pool_df=df_unlabeled,
    strategy='entropy',
    n_iterations=5,
    batch_size=20
)

# → history: список {iteration, n_labeled, accuracy, f1}
agent.report(history)  # → learning_curve.png

ЧТО СДАТЬ

- AL-цикл: старт с N=50 → 5 итераций → финальная модель
- Сравнение стратегий: entropy vs random — кривые обучения на одном графике
- Вывод: сколько примеров сэкономлено при том же качестве (accuracy/F1) vs random baseline
- Файл: agents/al_agent.py + notebooks/al_experiment.ipynb

Напоминаю, что @TEMA@ == недвижимость (аренда и продажа) в Санкт-Петербурге