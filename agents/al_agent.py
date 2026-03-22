# al_agent.py

"""
ActiveLearningAgent - агент для умного отбора данных (Active Learning).

Тема: недвижимость (аренда и продажа) в Санкт-Петербурге

Архитектура:
- fit(labeled_df) → model (обучить базовую модель)
- query(pool, strategy) → indices (стратегии: 'entropy', 'margin', 'random')
- evaluate(labeled_df, test_df) → Metrics (accuracy, F1)
- report(history) → LearningCurve (график quality vs. n_labeled)
- run_cycle() → history (AL-цикл)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Метрики качества модели."""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    n_labeled: int
    iteration: int
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            'accuracy': self.accuracy,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'n_labeled': self.n_labeled,
            'iteration': self.iteration
        }
    
    def __str__(self) -> str:
        """Строковое представление."""
        return (
            f"Iteration {self.iteration} (n={self.n_labeled}): "
            f"Accuracy={self.accuracy:.4f}, F1(macro)={self.f1_macro:.4f}, F1(weighted)={self.f1_weighted:.4f}"
        )


@dataclass
class LearningCurve:
    """Кривая обучения Active Learning."""
    history: List[Dict]
    strategy: str
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            'history': self.history,
            'strategy': self.strategy
        }
    
    def get_summary(self) -> pd.DataFrame:
        """Получить summary в виде DataFrame."""
        return pd.DataFrame(self.history)


class ActiveLearningAgent:
    """
    Агент для умного отбора данных (Active Learning).
    
    Поддерживаемые стратегии:
    - 'entropy': неопределенность на основе энтропии
    - 'margin': разница между двумя наиболее вероятными классами
    - 'random': случайная выборка (baseline)
    
    Поддерживаемые модели:
    - 'logreg': логистическая регрессия
    - 'rf': случайный лес
    - 'gb': градиентный бустинг
    """
    
    def __init__(self, 
                 model: str = 'logreg',
                 text_column: str = 'text',
                 label_column: str = 'label',
                 feature_columns: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Инициализация агента.
        
        Args:
            model: Тип модели ('logreg', 'rf', 'gb')
            text_column: Колонка с текстовым описанием
            label_column: Колонка с метками классов
            feature_columns: Список числовых признаков (опционально)
            random_state: Seed для воспроизводимости
        """
        self.model_type = model
        self.text_column = text_column
        self.label_column = label_column
        self.feature_columns = feature_columns or []
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.is_fitted = False
        
        logger.info(f"ActiveLearningAgent инициализирован с моделью: {model}")
    
    def _create_model(self) -> Any:
        """Создание модели классификации."""
        if self.model_type == 'logreg':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Неизвестная модель: {self.model_type}")
    
    def _extract_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Извлечение признаков из данных.
        
        Args:
            df: DataFrame с данными
            fit: Фитить ли векторизаторы (True для обучающих данных)
            
        Returns:
            Матрица признаков
        """
        features_list = []
        
        # Текстовые признаки (TF-IDF)
        if self.text_column in df.columns:
            texts = df[self.text_column].fillna('').astype(str)
            if fit:
                text_features = self.vectorizer.fit_transform(texts)
            else:
                text_features = self.vectorizer.transform(texts)
            features_list.append(text_features.toarray())
        
        # Числовые признаки
        numeric_features = []
        for col in self.feature_columns:
            if col in df.columns:
                numeric_features.append(df[col].fillna(df[col].median()))
        
        # Добавляем цену как важный признак
        if 'price' in df.columns:
            numeric_features.append(df['price'].fillna(df['price'].median()))
        
        # Добавляем площадь
        if 'area_sqm' in df.columns:
            numeric_features.append(df['area_sqm'].fillna(df['area_sqm'].median()))
        
        # Добавляем количество комнат
        if 'rooms' in df.columns:
            numeric_features.append(df['rooms'].fillna(df['rooms'].median()))
        
        if numeric_features:
            numeric_matrix = np.column_stack([f.values for f in numeric_features])
            if fit:
                numeric_matrix = self.scaler.fit_transform(numeric_matrix)
            else:
                numeric_matrix = self.scaler.transform(numeric_matrix)
            features_list.append(numeric_matrix)
        
        # Объединяем все признаки
        if features_list:
            return np.hstack(features_list)
        else:
            raise ValueError("Не найдено признаков для обучения")
    
    def fit(self, labeled_df: pd.DataFrame) -> Any:
        """
        Обучение базовой модели на размеченных данных.
        
        Args:
            labeled_df: DataFrame с размеченными данными
            
        Returns:
            Обученная модель
        """
        logger.info(f"Обучение модели на {len(labeled_df)} примерах...")
        
        # Подготовка данных
        df_clean = labeled_df.dropna(subset=[self.label_column])
        
        if len(df_clean) == 0:
            raise ValueError("Нет размеченных данных для обучения")
        
        # Извлечение признаков
        X = self._extract_features(df_clean, fit=True)
        
        # Кодирование меток
        y = self.label_encoder.fit_transform(df_clean[self.label_column])
        
        # Создание и обучение модели
        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"✓ Модель обучена на {len(df_clean)} примерах, {len(self.label_encoder.classes_)} классов")
        
        return self.model
    
    def query(self, pool: pd.DataFrame, 
              strategy: str = 'entropy',
              batch_size: int = 20) -> List[int]:
        """
        Выбор наиболее информативных примеров из пула.
        
        Args:
            pool: DataFrame с неразмеченными данными
            strategy: Стратегия выбора ('entropy', 'margin', 'random')
            batch_size: Количество примеров для выбора
            
        Returns:
            Индексы выбранных примеров (относительно pool)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        
        if len(pool) == 0:
            return []
        
        logger.info(f"Query с стратегией '{strategy}', batch_size={batch_size}")
        
        if strategy == 'random':
            # Случайная выборка
            indices = np.random.choice(
                len(pool), 
                size=min(batch_size, len(pool)), 
                replace=False
            )
            return indices.tolist()
        
        # Извлечение признаков
        X_pool = self._extract_features(pool, fit=False)
        
        # Получение вероятностей
        proba = self.model.predict_proba(X_pool)
        
        if strategy == 'entropy':
            # Энтропия: -sum(p * log(p))
            # Чем выше энтропия, тем больше неопределенность
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            indices = np.argsort(entropy)[-batch_size:][::-1]
            
        elif strategy == 'margin':
            # Margin: разница между двумя наиболее вероятными классами
            # Чем меньше margin, тем больше неопределенность
            sorted_proba = np.sort(proba, axis=1)
            margin = sorted_proba[:, -1] - sorted_proba[:, -2]
            indices = np.argsort(margin)[:batch_size]
            
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        
        return indices.tolist()
    
    def evaluate(self, labeled_df: pd.DataFrame, 
                 test_df: pd.DataFrame) -> Metrics:
        """
        Оценка качества модели.
        
        Args:
            labeled_df: DataFrame с размеченными данными (для обучения)
            test_df: DataFrame с тестовыми данными
            
        Returns:
            Metrics с метриками качества
        """
        # Обучаем модель на labeled_df
        self.fit(labeled_df)
        
        # Подготовка тестовых данных
        test_clean = test_df.dropna(subset=[self.label_column])
        
        if len(test_clean) == 0:
            raise ValueError("Нет тестовых данных с метками")
        
        # Извлечение признаков
        X_test = self._extract_features(test_clean, fit=False)
        y_test = self.label_encoder.transform(test_clean[self.label_column])
        
        # Предсказание
        y_pred = self.model.predict(X_test)
        
        # Расчет метрик
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = Metrics(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            n_labeled=len(labeled_df),
            iteration=0
        )
        
        return metrics
    
    def run_cycle(self,
                  labeled_df: pd.DataFrame,
                  pool_df: pd.DataFrame,
                  test_df: Optional[pd.DataFrame] = None,
                  strategy: str = 'entropy',
                  n_iterations: int = 5,
                  batch_size: int = 20,
                  verbose: bool = True) -> List[Dict]:
        """
        Запуск Active Learning цикла.
        
        Args:
            labeled_df: Начальный набор размеченных данных
            pool_df: Пул неразмеченных данных
            test_df: Тестовый набор для оценки (опционально)
            strategy: Стратегия выбора ('entropy', 'margin', 'random')
            n_iterations: Количество итераций
            batch_size: Размер батча на итерацию
            verbose: Выводить ли прогресс
            
        Returns:
            История обучения (список словарей с метриками)
        """
        history = []
        
        # Текущий набор размеченных данных
        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()
        
        logger.info(f"Запуск AL-цикла: strategy={strategy}, n_iterations={n_iterations}, batch_size={batch_size}")
        logger.info(f"Начальный размер labeled: {len(current_labeled)}, pool: {len(current_pool)}")
        
        for iteration in range(n_iterations + 1):
            # Оценка на текущем наборе
            if test_df is not None:
                metrics = self.evaluate(current_labeled, test_df)
                metrics.iteration = iteration
            else:
                # Если нет тестового набора, используем кросс-валидацию
                metrics = self._evaluate_cv(current_labeled)
                metrics.iteration = iteration
            
            history.append(metrics.to_dict())
            
            if verbose:
                logger.info(f"  {metrics}")
            
            # На последней итерации не делаем query
            if iteration == n_iterations:
                break
            
            # Query: выбор следующего батча
            if len(current_pool) == 0:
                logger.warning("Пул пуст, остановка цикла")
                break
            
            query_indices = self.query(current_pool, strategy=strategy, batch_size=batch_size)
            
            # Перемещаем выбранные примеры из pool в labeled
            selected = current_pool.iloc[query_indices]
            current_labeled = pd.concat([current_labeled, selected], ignore_index=True)
            current_pool = current_pool.drop(current_pool.index[query_indices]).reset_index(drop=True)
            
            logger.info(f"  Итерация {iteration + 1}: добавлено {len(selected)} примеров, всего labeled: {len(current_labeled)}")
        
        logger.info(f"✓ AL-цикл завершен. Финальный размер labeled: {len(current_labeled)}")
        
        return history
    
    def _evaluate_cv(self, labeled_df: pd.DataFrame, n_splits: int = 3) -> Metrics:
        """
        Оценка качества с помощью кросс-валидации.
        
        Args:
            labeled_df: DataFrame с размеченными данными
            n_splits: Количество сплитов
            
        Returns:
            Metrics с усредненными метриками
        """
        from sklearn.model_selection import StratifiedKFold
        
        df_clean = labeled_df.dropna(subset=[self.label_column])
        
        if len(df_clean) < n_splits * 2:
            # Недостаточно данных для CV
            return Metrics(
                accuracy=0.0,
                f1_macro=0.0,
                f1_weighted=0.0,
                n_labeled=len(labeled_df),
                iteration=0
            )
        
        X = self._extract_features(df_clean, fit=True)
        y = self.label_encoder.fit_transform(df_clean[self.label_column])
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        accuracies = []
        f1_macros = []
        f1_weighteds = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._create_model()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            accuracies.append(accuracy_score(y_val, y_pred))
            f1_macros.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            f1_weighteds.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
        
        return Metrics(
            accuracy=np.mean(accuracies),
            f1_macro=np.mean(f1_macros),
            f1_weighted=np.mean(f1_weighteds),
            n_labeled=len(labeled_df),
            iteration=0
        )
    
    def report(self, history: List[Dict], 
               output_path: str = 'learning_curve.png',
               compare_history: Optional[List[Dict]] = None,
               compare_label: str = 'random') -> str:
        """
        Создание отчета с кривой обучения.
        
        Args:
            history: История обучения (от run_cycle)
            output_path: Путь для сохранения графика
            compare_history: История для сравнения (опционально)
            compare_label: Название стратегии для сравнения
            
        Returns:
            Путь к сохраненному графику
        """
        df = pd.DataFrame(history)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Accuracy vs n_labeled
        ax1 = axes[0]
        ax1.plot(df['n_labeled'], df['accuracy'], 'o-', linewidth=2, markersize=8, label='Active Learning')
        
        if compare_history:
            df_compare = pd.DataFrame(compare_history)
            ax1.plot(df_compare['n_labeled'], df_compare['accuracy'], 's--', 
                    linewidth=2, markersize=8, label=f'{compare_label} (baseline)', alpha=0.7)
        
        ax1.set_xlabel('Количество размеченных примеров', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Кривая обучения: Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # График 2: F1-score vs n_labeled
        ax2 = axes[1]
        ax2.plot(df['n_labeled'], df['f1_macro'], 'o-', linewidth=2, markersize=8, label='F1-macro')
        ax2.plot(df['n_labeled'], df['f1_weighted'], '^-', linewidth=2, markersize=8, label='F1-weighted')
        
        if compare_history:
            df_compare = pd.DataFrame(compare_history)
            ax2.plot(df_compare['n_labeled'], df_compare['f1_macro'], 's--', 
                    linewidth=2, markersize=8, label=f'F1-macro ({compare_label})', alpha=0.7)
        
        ax2.set_xlabel('Количество размеченных примеров', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Кривая обучения: F1 Score', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ График сохранен: {output_path}")
        
        return output_path
    
    def analyze_savings(self, 
                        al_history: List[Dict],
                        random_history: List[Dict],
                        target_accuracy: Optional[float] = None,
                        target_f1: Optional[float] = None) -> Dict:
        """
        Анализ экономии данных при использовании AL vs random.
        
        Args:
            al_history: История AL-стратегии
            random_history: История random-стратегии
            target_accuracy: Целевая accuracy (опционально)
            target_f1: Целевой F1 (опционально)
            
        Returns:
            Словарь с результатами анализа
        """
        df_al = pd.DataFrame(al_history)
        df_random = pd.DataFrame(random_history)
        
        results = {
            'target_accuracy': target_accuracy,
            'target_f1': target_f1,
            'savings': {}
        }
        
        # Находим максимальные достигнутые метрики
        max_acc_al = df_al['accuracy'].max()
        max_acc_random = df_random['accuracy'].max()
        max_f1_al = df_al['f1_macro'].max()
        max_f1_random = df_random['f1_macro'].max()
        
        results['max_accuracy'] = {
            'al': max_acc_al,
            'random': max_acc_random,
            'improvement': max_acc_al - max_acc_random
        }
        
        results['max_f1_macro'] = {
            'al': max_f1_al,
            'random': max_f1_random,
            'improvement': max_f1_al - max_f1_random
        }
        
        # Анализ экономии для целевой accuracy
        if target_accuracy:
            # Находим минимальное количество примеров для достижения target
            al_mask = df_al['accuracy'] >= target_accuracy
            random_mask = df_random['accuracy'] >= target_accuracy
            
            if al_mask.any() and random_mask.any():
                n_al = df_al[al_mask]['n_labeled'].min()
                n_random = df_random[random_mask]['n_labeled'].min()
                savings = n_random - n_al
                savings_pct = (savings / n_random) * 100
                
                results['savings']['accuracy'] = {
                    'target': target_accuracy,
                    'n_al': int(n_al),
                    'n_random': int(n_random),
                    'savings': int(savings),
                    'savings_percent': float(savings_pct)
                }
        
        # Анализ экономии для целевого F1
        if target_f1:
            al_mask = df_al['f1_macro'] >= target_f1
            random_mask = df_random['f1_macro'] >= target_f1
            
            if al_mask.any() and random_mask.any():
                n_al = df_al[al_mask]['n_labeled'].min()
                n_random = df_random[random_mask]['n_labeled'].min()
                savings = n_random - n_al
                savings_pct = (savings / n_random) * 100
                
                results['savings']['f1_macro'] = {
                    'target': target_f1,
                    'n_al': int(n_al),
                    'n_random': int(n_random),
                    'savings': int(savings),
                    'savings_percent': float(savings_pct)
                }
        
        return results
    
    def print_summary(self, history: List[Dict]):
        """Вывод сводки по истории обучения."""
        df = pd.DataFrame(history)
        
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING - СВОДКА ПО ОБУЧЕНИЮ")
        print("=" * 70)
        
        print(f"\n{'Итерация':<12} {'N размечено':<15} {'Accuracy':<12} {'F1-macro':<12} {'F1-weighted':<12}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            print(f"{int(row['iteration']):<12} {int(row['n_labeled']):<15} "
                  f"{row['accuracy']:<12.4f} {row['f1_macro']:<12.4f} {row['f1_weighted']:<12.4f}")
        
        print("-" * 70)
        print(f"\nНачальное качество:     Accuracy={df['accuracy'].iloc[0]:.4f}, F1={df['f1_macro'].iloc[0]:.4f}")
        print(f"Финальное качество:     Accuracy={df['accuracy'].iloc[-1]:.4f}, F1={df['f1_macro'].iloc[-1]:.4f}")
        print(f"Прирост:                Accuracy={df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]:+.4f}, "
              f"F1={df['f1_macro'].iloc[-1] - df['f1_macro'].iloc[0]:+.4f}")
        print("=" * 70)


def fit(labeled_df: pd.DataFrame, model: str = 'logreg', **kwargs) -> Any:
    """
    Скилл для обучения модели.
    
    Args:
        labeled_df: DataFrame с размеченными данными
        model: Тип модели ('logreg', 'rf', 'gb')
        **kwargs: Дополнительные параметры
        
    Returns:
        Обученная модель
    """
    agent = ActiveLearningAgent(model=model, **kwargs)
    return agent.fit(labeled_df)


def query(pool: pd.DataFrame, 
          labeled_df: pd.DataFrame,
          strategy: str = 'entropy',
          batch_size: int = 20,
          model: str = 'logreg',
          **kwargs) -> List[int]:
    """
    Скилл для выбора информативных примеров.
    
    Args:
        pool: DataFrame с неразмеченными данными
        labeled_df: DataFrame с размеченными данными (для обучения)
        strategy: Стратегия выбора
        batch_size: Размер батча
        model: Тип модели
        **kwargs: Дополнительные параметры
        
    Returns:
        Индексы выбранных примеров
    """
    agent = ActiveLearningAgent(model=model, **kwargs)
    agent.fit(labeled_df)
    return agent.query(pool, strategy=strategy, batch_size=batch_size)


def evaluate(labeled_df: pd.DataFrame, 
             test_df: pd.DataFrame,
             model: str = 'logreg',
             **kwargs) -> Metrics:
    """
    Скилл для оценки качества модели.
    
    Args:
        labeled_df: DataFrame с размеченными данными
        test_df: DataFrame с тестовыми данными
        model: Тип модели
        **kwargs: Дополнительные параметры
        
    Returns:
        Metrics с метриками качества
    """
    agent = ActiveLearningAgent(model=model, **kwargs)
    return agent.evaluate(labeled_df, test_df)


def run_cycle(labeled_df: pd.DataFrame,
              pool_df: pd.DataFrame,
              test_df: Optional[pd.DataFrame] = None,
              strategy: str = 'entropy',
              n_iterations: int = 5,
              batch_size: int = 20,
              model: str = 'logreg',
              **kwargs) -> List[Dict]:
    """
    Скилл для запуска AL-цикла.
    
    Args:
        labeled_df: Начальный набор размеченных данных
        pool_df: Пул неразмеченных данных
        test_df: Тестовый набор для оценки
        strategy: Стратегия выбора
        n_iterations: Количество итераций
        batch_size: Размер батча
        model: Тип модели
        **kwargs: Дополнительные параметры
        
    Returns:
        История обучения
    """
    agent = ActiveLearningAgent(model=model, **kwargs)
    return agent.run_cycle(
        labeled_df=labeled_df,
        pool_df=pool_df,
        test_df=test_df,
        strategy=strategy,
        n_iterations=n_iterations,
        batch_size=batch_size
    )


def report(history: List[Dict], 
           output_path: str = 'learning_curve.png',
           **kwargs) -> str:
    """
    Скилл для создания отчета с кривой обучения.
    
    Args:
        history: История обучения
        output_path: Путь для сохранения графика
        **kwargs: Дополнительные параметры
        
    Returns:
        Путь к сохраненному графику
    """
    agent = ActiveLearningAgent()
    return agent.report(history, output_path=output_path, **kwargs)


if __name__ == '__main__':
    print("ActiveLearningAgent - Агент для умного отбора данных")
    print("=" * 60)
    print("\nДоступные методы:")
    print("  - fit(labeled_df) → model")
    print("  - query(pool, strategy) → indices")
    print("  - evaluate(labeled_df, test_df) → Metrics")
    print("  - run_cycle(...) → history")
    print("  - report(history) → learning_curve.png")
    print("\nСтратегии: 'entropy', 'margin', 'random'")
    print("Модели: 'logreg', 'rf', 'gb'")
