"""
AnnotationAgent - агент для автоматической разметки данных.

Архитектура:
- auto_label(df, modality) → DataFrame
- generate_spec(df, task) → AnnotationSpec (Markdown)
- check_quality(df_labeled) → QualityMetrics
- export_to_labelstudio(df) → JSON
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AnnotationSpec:
    """Спецификация разметки."""
    task_name: str
    task_description: str
    classes: Dict[str, Dict[str, Any]]
    examples: Dict[str, List[Dict]]
    edge_cases: List[Dict]
    guidelines: List[str]
    
    def to_markdown(self) -> str:
        """Конвертация в Markdown."""
        lines = [
            f"# Спецификация разметки: {self.task_name}",
            "",
            f"**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Описание задачи",
            "",
            self.task_description,
            "",
            "## Классы",
            ""
        ]
        
        for class_name, class_info in self.classes.items():
            lines.extend([
                f"### {class_name}",
                "",
                f"**Определение:** {class_info.get('definition', '')}",
                "",
                f"**Диапазон цен:** {class_info.get('price_range', 'N/A')}",
                "",
                f"**Примерная доля в датасете:** {class_info.get('proportion', 'N/A')}",
                ""
            ])
        
        lines.extend([
            "## Примеры разметки",
            ""
        ])
        
        for class_name, examples in self.examples.items():
            lines.extend([
                f"### Класс: {class_name}",
                ""
            ])
            for i, example in enumerate(examples, 1):
                lines.extend([
                    f"**Пример {i}:**",
                    f"- Текст: \"{example.get('text', '')[:100]}...\"",
                    f"- Цена: {example.get('price', 'N/A')} руб.",
                    f"- Обоснование: {example.get('reasoning', '')}",
                    ""
                ])
        
        lines.extend([
            "## Граничные случаи",
            ""
        ])
        
        for i, case in enumerate(self.edge_cases, 1):
            lines.extend([
                f"### Случай {i}",
                f"- Описание: {case.get('description', '')}",
                f"- Рекомендация: {case.get('recommendation', '')}",
                ""
            ])
        
        lines.extend([
            "## Руководство по разметке",
            ""
        ])
        
        for i, guideline in enumerate(self.guidelines, 1):
            lines.append(f"{i}. {guideline}")
        
        lines.extend([
            "",
            "---",
            "",
            "*Спецификация сгенерирована автоматически AnnotationAgent*"
        ])
        
        return "\n".join(lines)


@dataclass
class QualityMetrics:
    """Метрики качества разметки."""
    label_distribution: Dict[str, int]
    label_proportions: Dict[str, float]
    confidence_stats: Dict[str, float]
    agreement_score: Optional[float] = None
    kappa_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь."""
        return {
            'label_distribution': self.label_distribution,
            'label_proportions': self.label_proportions,
            'confidence_stats': self.confidence_stats,
            'agreement_score': self.agreement_score,
            'kappa_score': self.kappa_score
        }
    
    def __str__(self) -> str:
        """Строковое представление."""
        lines = [
            "=" * 60,
            "МЕТРИКИ КАЧЕСТВА РАЗМЕТКИ",
            "=" * 60,
            "",
            "📊 Распределение меток:",
            "-" * 40
        ]
        
        for label, count in self.label_distribution.items():
            pct = self.label_proportions.get(label, 0) * 100
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            "📊 Статистика уверенности:",
            "-" * 40
        ])
        
        for stat, value in self.confidence_stats.items():
            lines.append(f"  {stat}: {value:.3f}")
        
        if self.kappa_score is not None:
            lines.extend([
                "",
                f"📊 Cohen's Kappa: {self.kappa_score:.3f}"
            ])
        
        if self.agreement_score is not None:
            lines.extend([
                "",
                f"📊 Agreement Score: {self.agreement_score:.3f}"
            ])
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class AnnotationAgent:
    """
    Агент для автоматической разметки данных.
    
    Поддерживаемые модальности:
    - text: текстовые данные (правила на основе признаков)
    - audio: аудио данные (Whisper)
    - image: изображения (YOLO)
    """
    
    def __init__(self, modality: str = 'text', confidence_threshold: float = 0.7):
        """
        Инициализация агента.
        
        Args:
            modality: Тип данных ('text', 'audio', 'image')
            confidence_threshold: Порог уверенности для авторазметки
        """
        self.modality = modality
        self.confidence_threshold = confidence_threshold
        self.spec = None
        
    def auto_label(self, df: pd.DataFrame, 
                   text_column: str = 'text',
                   price_column: str = 'price',
                   label_column: str = 'label') -> pd.DataFrame:
        """
        Автоматическая разметка данных.
        
        Для текстовых данных использует правила на основе цены.
        
        Args:
            df: DataFrame с данными
            text_column: Колонка с текстом
            price_column: Колонка с ценой
            label_column: Колонка для меток
            
        Returns:
            DataFrame с добавленными/обновленными метками
        """
        df_labeled = df.copy()
        
        # Подсчитываем неразмеченные данные
        unlabeled_mask = df_labeled[label_column].isna()
        unlabeled_count = unlabeled_mask.sum()
        
        if unlabeled_count == 0:
            print("Все данные уже размечены!")
            return df_labeled
        
        print(f"Авторазметка {unlabeled_count} неразмеченных записей...")
        
        if self.modality == 'text':
            df_labeled = self._auto_label_text(
                df_labeled, text_column, price_column, label_column
            )
        elif self.modality == 'audio':
            df_labeled = self._auto_label_audio(
                df_labeled, text_column, price_column, label_column
            )
        elif self.modality == 'image':
            df_labeled = self._auto_label_image(
                df_labeled, text_column, price_column, label_column
            )
        else:
            raise ValueError(f"Неизвестная модальность: {self.modality}")
        
        # Подсчитываем результаты
        newly_labeled = df_labeled.loc[unlabeled_mask, label_column].notna().sum()
        print(f"✓ Размечено: {newly_labeled} / {unlabeled_count}")
        
        return df_labeled
    
    def _auto_label_text(self, df: pd.DataFrame, 
                         text_column: str,
                         price_column: str,
                         label_column: str) -> pd.DataFrame:
        """Авторазметка текстовых данных на основе цены."""
        df_result = df.copy()
        
        # Маска неразмеченных данных
        unlabeled_mask = df_result[label_column].isna()
        
        # Правила классификации на основе цены
        def classify_by_price(price):
            if pd.isna(price):
                return None
            if price < 4_500_000:
                return 'Эконом'
            elif price < 7_000_000:
                return 'Стандарт'
            elif price < 10_000_000:
                return 'Комфорт'
            else:
                return 'Премиум'
        
        # Применяем классификацию к неразмеченным данным
        df_result.loc[unlabeled_mask, label_column] = df_result.loc[
            unlabeled_mask, price_column
        ].apply(classify_by_price)
        
        # Добавляем колонку уверенности (для правил - всегда 1.0)
        if 'confidence' not in df_result.columns:
            df_result['confidence'] = np.nan
        df_result.loc[unlabeled_mask, 'confidence'] = 1.0
        
        return df_result
    
    def _auto_label_audio(self, df: pd.DataFrame,
                          text_column: str,
                          price_column: str,
                          label_column: str) -> pd.DataFrame:
        """Авторазметка аудио данных (заглушка)."""
        print("Аудио разметка требует Whisper API (не реализовано)")
        return df
    
    def _auto_label_image(self, df: pd.DataFrame,
                          text_column: str,
                          price_column: str,
                          label_column: str) -> pd.DataFrame:
        """Авторазметка изображений (заглушка)."""
        print("Image разметка требует YOLO (не реализовано)")
        return df
    
    def generate_spec(self, df: pd.DataFrame, 
                      task: str = 'price_classification',
                      label_column: str = 'label',
                      price_column: str = 'price',
                      text_column: str = 'text') -> AnnotationSpec:
        """
        Генерация спецификации разметки.
        
        Args:
            df: DataFrame с размеченными данными
            task: Название задачи
            label_column: Колонка с метками
            price_column: Колонка с ценой
            text_column: Колонка с текстом
            
        Returns:
            AnnotationSpec с полной спецификацией
        """
        # Анализируем распределение
        label_counts = df[label_column].value_counts()
        total = len(df)
        
        # Определяем классы
        classes = {
            'Эконом': {
                'definition': 'Недвижимость эконом-класса с доступной ценой',
                'price_range': 'до 4,500,000 руб.',
                'proportion': f"{label_counts.get('Эконом', 0) / total * 100:.1f}%"
            },
            'Стандарт': {
                'definition': 'Недвижимость стандартного класса с умеренной ценой',
                'price_range': '4,500,000 - 7,000,000 руб.',
                'proportion': f"{label_counts.get('Стандарт', 0) / total * 100:.1f}%"
            },
            'Комфорт': {
                'definition': 'Недвижимость комфорт-класса с повышенной ценой',
                'price_range': '7,000,000 - 10,000,000 руб.',
                'proportion': f"{label_counts.get('Комфорт', 0) / total * 100:.1f}%"
            },
            'Премиум': {
                'definition': 'Недвижимость премиум-класса с высокой ценой',
                'price_range': 'от 10,000,000 руб.',
                'proportion': f"{label_counts.get('Премиум', 0) / total * 100:.1f}%"
            }
        }
        
        # Собираем примеры для каждого класса
        examples = {}
        for class_name in classes.keys():
            class_df = df[df[label_column] == class_name].dropna(subset=[text_column, price_column])
            if len(class_df) >= 3:
                sample = class_df.sample(3, random_state=42)
                examples[class_name] = [
                    {
                        'text': row[text_column],
                        'price': f"{row[price_column]:,.0f}",
                        'reasoning': f"Цена {row[price_column]:,.0f} руб. попадает в диапазон {classes[class_name]['price_range']}"
                    }
                    for _, row in sample.iterrows()
                ]
        
        # Граничные случаи
        edge_cases = [
            {
                'description': 'Цена на границе диапазонов (например, ровно 4,500,000 руб.)',
                'recommendation': 'Относить к более высокому классу (Стандарт)'
            },
            {
                'description': 'Отсутствие информации о цене',
                'recommendation': 'Пропустить или отметить как "неопределено"'
            },
            {
                'description': 'Аномально низкая цена для района (возможно, ошибка)',
                'recommendation': 'Проверить исходные данные, возможно требуется ручная проверка'
            },
            {
                'description': 'Цена указана за квадратный метр, а не за объект',
                'recommendation': 'Пересчитать общую цену и затем классифицировать'
            }
        ]
        
        # Руководство по разметке
        guidelines = [
            "Ориентируйтесь прежде всего на цену объекта",
            "При граничных значениях (±5% от порога) проверяйте дополнительные факторы: район, площадь",
            "Для квартир в центральных районах возможно повышение класса на 1 уровень",
            "Учитывайте год постройки: старые дома (до 1960) обычно на 1 класс ниже",
            "При сомнениях выбирайте более консервативный (низкий) класс"
        ]
        
        spec = AnnotationSpec(
            task_name='Классификация недвижимости по ценовым категориям',
            task_description='Задача заключается в определении ценовой категории объекта недвижимости на основе его стоимости. Классификация используется для сегментации рынка и рекомендательных систем.',
            classes=classes,
            examples=examples,
            edge_cases=edge_cases,
            guidelines=guidelines
        )
        
        self.spec = spec
        return spec
    
    def check_quality(self, df_labeled: pd.DataFrame,
                      label_column: str = 'label',
                      confidence_column: str = 'confidence',
                      reference_column: Optional[str] = None) -> QualityMetrics:
        """
        Оценка качества разметки.
        
        Args:
            df_labeled: DataFrame с разметкой
            label_column: Колонка с метками
            confidence_column: Колонка с уверенностью
            reference_column: Колонка с эталонной разметкой (для сравнения)
            
        Returns:
            QualityMetrics с метриками качества
        """
        # Распределение меток
        label_dist = df_labeled[label_column].value_counts().to_dict()
        label_props = (df_labeled[label_column].value_counts(normalize=True)).to_dict()
        
        # Статистика уверенности
        if confidence_column in df_labeled.columns:
            conf_data = df_labeled[confidence_column].dropna()
            confidence_stats = {
                'mean': float(conf_data.mean()),
                'std': float(conf_data.std()),
                'min': float(conf_data.min()),
                'max': float(conf_data.max())
            }
        else:
            confidence_stats = {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}
        
        # Если есть эталонная разметка - считаем согласие и kappa
        agreement = None
        kappa = None
        
        if reference_column and reference_column in df_labeled.columns:
            # Фильтруем только строки с обеими метками
            comparison_df = df_labeled.dropna(subset=[label_column, reference_column])
            
            if len(comparison_df) > 0:
                # Agreement
                agreement = (comparison_df[label_column] == comparison_df[reference_column]).mean()
                
                # Cohen's Kappa
                kappa = self._cohens_kappa(
                    comparison_df[label_column],
                    comparison_df[reference_column]
                )
        
        metrics = QualityMetrics(
            label_distribution=label_dist,
            label_proportions=label_props,
            confidence_stats=confidence_stats,
            agreement_score=agreement,
            kappa_score=kappa
        )
        
        return metrics
    
    def _cohens_kappa(self, labels1: pd.Series, labels2: pd.Series) -> float:
        """Расчет Cohen's Kappa."""
        # Создаем матрицу согласия
        classes = sorted(set(labels1.unique()) | set(labels2.unique()))
        n = len(labels1)
        
        # Observed agreement
        po = (labels1 == labels2).sum() / n
        
        # Expected agreement
        pe = 0
        for c in classes:
            p1 = (labels1 == c).sum() / n
            p2 = (labels2 == c).sum() / n
            pe += p1 * p2
        
        # Kappa
        if pe == 1:
            return 1.0
        
        kappa = (po - pe) / (1 - pe)
        return float(kappa)
    
    def export_to_labelstudio(self, df: pd.DataFrame,
                              output_path: str = 'labelstudio_import.json',
                              text_column: str = 'text',
                              label_column: str = 'label',
                              id_column: str = 'id') -> str:
        """
        Экспорт данных в формат LabelStudio.
        
        Args:
            df: DataFrame с данными
            output_path: Путь для сохранения JSON
            text_column: Колонка с текстом
            label_column: Колонка с метками
            id_column: Колонка с ID
            
        Returns:
            Путь к сохраненному файлу
        """
        tasks = []
        
        for idx, row in df.iterrows():
            task = {
                "id": str(row.get(id_column, idx)),
                "data": {
                    "text": str(row.get(text_column, ''))
                },
                "annotations": []
            }
            
            # Если есть разметка - добавляем
            if pd.notna(row.get(label_column)):
                task["annotations"].append({
                    "result": [
                        {
                            "value": {
                                "choices": [str(row[label_column])]
                            },
                            "from_name": "sentiment",
                            "to_name": "text",
                            "type": "choices"
                        }
                    ]
                })
            
            tasks.append(task)
        
        # Сохраняем
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Экспортировано {len(tasks)} задач в {output_path}")
        return output_path
    
    def save_spec(self, output_path: str = 'annotation_spec.md'):
        """Сохранение спецификации в Markdown файл."""
        if self.spec is None:
            raise ValueError("Спецификация не создана. Сначала вызовите generate_spec()")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.spec.to_markdown())
        
        print(f"✓ Спецификация сохранена: {output_path}")
        return output_path


# Функции-скиллы для прямого использования
def auto_label(df: pd.DataFrame, modality: str = 'text', **kwargs) -> pd.DataFrame:
    """Скилл для автоматической разметки."""
    agent = AnnotationAgent(modality=modality)
    return agent.auto_label(df, **kwargs)


def generate_spec(df: pd.DataFrame, task: str = 'classification', **kwargs) -> AnnotationSpec:
    """Скилл для генерации спецификации."""
    agent = AnnotationAgent()
    return agent.generate_spec(df, task=task, **kwargs)


def check_quality(df_labeled: pd.DataFrame, **kwargs) -> QualityMetrics:
    """Скилл для проверки качества разметки."""
    agent = AnnotationAgent()
    return agent.check_quality(df_labeled, **kwargs)


def export_to_labelstudio(df: pd.DataFrame, output_path: str = 'labelstudio_import.json', **kwargs) -> str:
    """Скилл для экспорта в LabelStudio."""
    agent = AnnotationAgent()
    return agent.export_to_labelstudio(df, output_path=output_path, **kwargs)


if __name__ == '__main__':
    print("AnnotationAgent - Агент для автоматической разметки данных")
    print("Используйте методы: auto_label(), generate_spec(), check_quality(), export_to_labelstudio()")

    df = pd.read_csv("/home/ml-with-lm/AITH_2_Sem/Data_for_ML/Data_for_ML_Smolyanov/data/raw/prepared_data.csv")
    agent = AnnotationAgent(modality='text')

    df_labeled = agent.auto_label(df, text_column='text', price_column='price')

    # 2. Генерация спецификации
    spec = agent.generate_spec(df_labeled, task='price_classification')
    agent.save_spec('annotation_spec.md')

    # 3. Проверка качества
    metrics = agent.check_quality(df_labeled)
    print(metrics)

    # 4. Экспорт в LabelStudio
    agent.export_to_labelstudio(df_labeled, 'labelstudio_import.json')