# data_quality_agent.py

"""
DataQualityAgent - агент-детектив для обнаружения и исправления проблем качества данных.

Архитектура:
- detect_issues(df) → QualityReport
- fix(df, strategy: dict) → DataFrame
- compare(df_before, df_after) → ComparisonReport
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityReport:
    """Отчет о качестве данных."""
    missing: Dict[str, Any] = field(default_factory=dict)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    outliers: Dict[str, Any] = field(default_factory=dict)
    imbalance: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Конвертация отчета в словарь."""
        return {
            'missing': self.missing,
            'duplicates': self.duplicates,
            'outliers': self.outliers,
            'imbalance': self.imbalance,
            'summary': self.summary
        }
    
    def __str__(self) -> str:
        """Строковое представление отчета."""
        lines = ["=" * 60, "ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ", "=" * 60]
        
        # Пропущенные значения
        lines.extend(["\n📊 ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:", "-" * 40])
        if self.missing.get('total_missing', 0) > 0:
            lines.append(f"Всего пропущенных: {self.missing['total_missing']:,}")
            lines.append(f"Процент пропусков: {self.missing['total_percent']:.2f}%")
            lines.append("\nПо колонкам:")
            for col, info in self.missing.get('by_column', {}).items():
                lines.append(f"  {col}: {info['count']} ({info['percent']:.2f}%)")
        else:
            lines.append("Пропущенных значений не обнаружено ✓")
        
        # Дубликаты
        lines.extend(["\n📊 ДУБЛИКАТЫ:", "-" * 40])
        dup_count = self.duplicates.get('count', 0)
        lines.append(f"Количество дубликатов: {dup_count:,}")
        lines.append(f"Процент дубликатов: {self.duplicates.get('percent', 0):.2f}%")
        
        # Выбросы
        lines.extend(["\n📊 ВЫБРОСЫ:", "-" * 40])
        if self.outliers.get('columns'):
            total_outliers = sum(info['count'] for info in self.outliers['columns'].values())
            lines.append(f"Всего выбросов: {total_outliers:,}")
            lines.append("\nПо колонкам (метод IQR):")
            for col, info in self.outliers['columns'].items():
                lines.append(f"  {col}: {info['count']} ({info['percent']:.2f}%)")
                lines.append(f"    Диапазон нормы: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        else:
            lines.append("Выбросы не обнаружены")
        
        # Дисбаланс классов
        lines.extend(["\n📊 ДИСБАЛАНС КЛАССОВ:", "-" * 40])
        if self.imbalance.get('columns'):
            for col, info in self.imbalance['columns'].items():
                lines.append(f"\nКолонка '{col}':")
                lines.append(f"  Коэффициент дисбаланса: {info['imbalance_ratio']:.2f}")
                lines.append(f"  Степень дисбаланса: {info['severity']}")
                lines.append("  Распределение:")
                for val, pct in info['distribution'].items():
                    lines.append(f"    {val}: {pct:.1f}%")
        
        # Сводка
        lines.extend(["\n📊 СВОДКА:", "-" * 40])
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Отчет о сравнении данных до и после очистки."""
    before_stats: Dict[str, Any] = field(default_factory=dict)
    after_stats: Dict[str, Any] = field(default_factory=dict)
    changes: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Строковое представление отчета."""
        lines = ["=" * 80, "СРАВНИТЕЛЬНЫЙ ОТЧЕТ: ДО / ПОСЛЕ", "=" * 80]
        
        lines.extend(["\n📊 ОБЩИЕ МЕТРИКИ:", "-" * 60])
        lines.append(f"{'Метрика':<30} {'До':<15} {'После':<15} {'Изменение':<15}")
        lines.append("-" * 60)
        
        metrics = ['rows', 'columns', 'missing_values', 'duplicates', 'outliers']
        for metric in metrics:
            before_val = self.before_stats.get(metric, 0)
            after_val = self.after_stats.get(metric, 0)
            change = after_val - before_val
            change_pct = (change / before_val * 100) if before_val > 0 else 0
            lines.append(f"{metric:<30} {before_val:<15,} {after_val:<15,} {change:+,} ({change_pct:+.1f}%)")
        
        lines.extend(["\n📊 ИЗМЕНЕНИЯ В ЧИСЛОВЫХ КОЛОНКАХ:", "-" * 60])
        if self.changes.get('numeric'):
            lines.append(f"{'Колонка':<20} {'Метрика':<15} {'До':<15} {'После':<15}")
            lines.append("-" * 60)
            for col, metrics in self.changes['numeric'].items():
                for metric_name, values in metrics.items():
                    before_val = values['before']
                    after_val = values['after']
                    lines.append(f"{col:<20} {metric_name:<15} {before_val:<15.2f} {after_val:<15.2f}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class DataQualityAgent:
    """
    Агент-детектив для обнаружения и исправления проблем качества данных.
    
    Поддерживаемые проблемы:
    - Пропущенные значения
    - Дубликаты
    - Выбросы (IQR, Z-score)
    - Дисбаланс классов
    """
    
    def __init__(self, verbose: bool = True):
        """
        Инициализация агента.
        
        Args:
            verbose: Выводить ли подробную информацию
        """
        self.verbose = verbose
        self.report = None
        
    def detect_issues(self, df: pd.DataFrame, 
                     outlier_method: str = 'iqr',
                     outlier_threshold: float = 1.5,
                     imbalance_threshold: float = 0.5) -> QualityReport:
        """
        Обнаружение проблем качества данных.
        
        Args:
            df: DataFrame для анализа
            outlier_method: Метод обнаружения выбросов ('iqr' или 'zscore')
            outlier_threshold: Порог для выбросов
            imbalance_threshold: Порог для детекции дисбаланса
            
        Returns:
            QualityReport с результатами анализа
        """
        report = QualityReport()
        
        # 1. Пропущенные значения
        report.missing = self._detect_missing(df)
        
        # 2. Дубликаты
        report.duplicates = self._detect_duplicates(df)
        
        # 3. Выбросы
        report.outliers = self._detect_outliers(df, method=outlier_method, threshold=outlier_threshold)
        
        # 4. Дисбаланс классов
        report.imbalance = self._detect_imbalance(df, threshold=imbalance_threshold)
        
        # 5. Сводка
        report.summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'has_issues': (report.missing['total_missing'] > 0 or 
                          report.duplicates['count'] > 0 or 
                          len(report.outliers.get('columns', {})) > 0)
        }
        
        self.report = report
        
        if self.verbose:
            print(report)
        
        return report
    
    def _detect_missing(self, df: pd.DataFrame) -> Dict:
        """Обнаружение пропущенных значений."""
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        by_column = {}
        for col in df.columns:
            if missing_count[col] > 0:
                by_column[col] = {
                    'count': int(missing_count[col]),
                    'percent': float(missing_percent[col])
                }
        
        return {
            'total_missing': int(missing_count.sum()),
            'total_percent': float((missing_count.sum() / (len(df) * len(df.columns))) * 100),
            'by_column': by_column
        }
    
    def _detect_duplicates(self, df: pd.DataFrame) -> Dict:
        """Обнаружение дубликатов."""
        dup_count = df.duplicated().sum()
        return {
            'count': int(dup_count),
            'percent': float((dup_count / len(df)) * 100) if len(df) > 0 else 0
        }
    
    def _detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                        threshold: float = 1.5) -> Dict:
        """Обнаружение выбросов в числовых колонках."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_by_col = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = df[col].isin(data[z_scores > threshold])
                lower_bound = data.mean() - threshold * data.std()
                upper_bound = data.mean() + threshold * data.std()
            else:
                raise ValueError(f"Неизвестный метод: {method}")
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outliers_by_col[col] = {
                    'count': int(outlier_count),
                    'percent': float((outlier_count / len(df)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': method
                }
        
        return {
            'method': method,
            'columns': outliers_by_col
        }
    
    def _detect_imbalance(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Обнаружение дисбаланса классов в категориальных колонках."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        imbalance_by_col = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) <= 1:
                continue
                
            # Коэффициент дисбаланса
            max_count = value_counts.iloc[0]
            min_count = value_counts.iloc[-1]
            imbalance_ratio = min_count / max_count
            
            if imbalance_ratio < threshold:
                # Определяем степень дисбаланса
                if imbalance_ratio < 0.1:
                    severity = 'Критический'
                elif imbalance_ratio < 0.3:
                    severity = 'Высокий'
                else:
                    severity = 'Умеренный'
                
                distribution = {
                    str(k): float(v / len(df) * 100) 
                    for k, v in value_counts.head(5).items()
                }
                
                imbalance_by_col[col] = {
                    'imbalance_ratio': float(imbalance_ratio),
                    'severity': severity,
                    'distribution': distribution
                }
        
        return {'columns': imbalance_by_col}
    
    def fix(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Исправление проблем качества данных.
        
        Args:
            df: Исходный DataFrame
            strategy: Словарь стратегий для каждого типа проблем
                Пример: {
                    'missing': 'median',  # 'drop', 'mean', 'median', 'mode', 'fill_value'
                    'duplicates': 'drop',  # 'drop' или 'keep'
                    'outliers': 'clip_iqr'  # 'drop', 'clip_iqr', 'clip_zscore', 'none'
                }
                
        Returns:
            Очищенный DataFrame
        """
        df_clean = df.copy()
        
        # 1. Обработка пропущенных значений
        missing_strategy = strategy.get('missing', 'none')
        if missing_strategy != 'none':
            df_clean = self._fix_missing(df_clean, missing_strategy)
        
        # 2. Обработка дубликатов
        duplicates_strategy = strategy.get('duplicates', 'none')
        if duplicates_strategy == 'drop':
            df_clean = df_clean.drop_duplicates()
        
        # 3. Обработка выбросов
        outliers_strategy = strategy.get('outliers', 'none')
        if outliers_strategy != 'none':
            df_clean = self._fix_outliers(df_clean, outliers_strategy)
        
        if self.verbose:
            print(f"\n✓ Очистка завершена: {len(df)} → {len(df_clean)} строк")
        
        return df_clean
    
    def _fix_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Исправление пропущенных значений."""
        df_clean = df.copy()
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
        elif strategy.startswith('fill_'):
            fill_value = strategy.split('_', 1)[1]
            df_clean = df_clean.fillna(fill_value)
        
        return df_clean
    
    def _fix_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Исправление выбросов."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df_clean[col].dropna()
            if len(data) == 0:
                continue
            
            if strategy == 'drop':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                
            elif strategy == 'clip_iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif strategy == 'clip_zscore':
                mean = data.mean()
                std = data.std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
        """
        Сравнение данных до и после очистки.
        
        Args:
            df_before: DataFrame до очистки
            df_after: DataFrame после очистки
            
        Returns:
            ComparisonReport с результатами сравнения
        """
        # Статистика до
        before_stats = {
            'rows': len(df_before),
            'columns': len(df_before.columns),
            'missing_values': df_before.isnull().sum().sum(),
            'duplicates': df_before.duplicated().sum(),
            'outliers': sum(
                self._detect_outliers(df_before)['columns'].get(col, {}).get('count', 0)
                for col in df_before.select_dtypes(include=[np.number]).columns
            )
        }
        
        # Статистика после
        after_stats = {
            'rows': len(df_after),
            'columns': len(df_after.columns),
            'missing_values': df_after.isnull().sum().sum(),
            'duplicates': df_after.duplicated().sum(),
            'outliers': sum(
                self._detect_outliers(df_after)['columns'].get(col, {}).get('count', 0)
                for col in df_after.select_dtypes(include=[np.number]).columns
            )
        }
        
        # Изменения в числовых колонках
        numeric_changes = {}
        numeric_cols = df_before.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df_after.columns:
                numeric_changes[col] = {
                    'mean': {
                        'before': float(df_before[col].mean()),
                        'after': float(df_after[col].mean())
                    },
                    'std': {
                        'before': float(df_before[col].std()),
                        'after': float(df_after[col].std())
                    },
                    'min': {
                        'before': float(df_before[col].min()),
                        'after': float(df_after[col].min())
                    },
                    'max': {
                        'before': float(df_before[col].max()),
                        'after': float(df_after[col].max())
                    }
                }
        
        report = ComparisonReport(
            before_stats=before_stats,
            after_stats=after_stats,
            changes={'numeric': numeric_changes}
        )
        
        if self.verbose:
            print(report)
        
        return report
    
    def visualize_issues(self, df: pd.DataFrame, output_dir: str = 'visualizations'):
        """
        Визуализация проблем качества данных.
        
        Args:
            df: DataFrame для визуализации
            output_dir: Директория для сохранения графиков
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Визуализация пропущенных значений
        self._visualize_missing(df, output_dir)
        
        # 2. Визуализация выбросов
        self._visualize_outliers(df, output_dir)
        
        # 3. Визуализация дисбаланса
        self._visualize_imbalance(df, output_dir)
        
        print(f"\n✓ Визуализации сохранены в: {output_dir}/")
    
    def _visualize_missing(self, df: pd.DataFrame, output_dir: str):
        """Визуализация пропущенных значений."""
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            return
        
        plt.figure(figsize=(12, 6))
        missing.plot(kind='bar', color='coral')
        plt.title('Пропущенные значения по колонкам')
        plt.xlabel('Колонки')
        plt.ylabel('Количество пропусков')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/missing_values.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_outliers(self, df: pd.DataFrame, output_dir: str):
        """Визуализация выбросов."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return
        
        # Выбираем до 6 колонок для визуализации
        cols_to_plot = numeric_cols[:6]
        n_cols = len(cols_to_plot)
        
        fig, axes = plt.subplots((n_cols + 2) // 3, 3, figsize=(15, 4 * ((n_cols + 2) // 3)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot):
            data = df[col].dropna()
            
            # Box plot
            axes[idx].boxplot(data, vert=True)
            axes[idx].set_title(f'Выбросы: {col}')
            axes[idx].set_ylabel('Значение')
            axes[idx].grid(True, alpha=0.3)
        
        # Скрываем лишние оси
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/outliers_boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_imbalance(self, df: pd.DataFrame, output_dir: str):
        """Визуализация дисбаланса классов."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Выбираем колонки с небольшим количеством уникальных значений
        cols_to_plot = [col for col in categorical_cols if df[col].nunique() <= 10]
        
        if len(cols_to_plot) == 0:
            return
        
        n_cols = min(len(cols_to_plot), 4)
        fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(14, 4 * ((n_cols + 1) // 2)))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols_to_plot[:n_cols]):
            value_counts = df[col].value_counts()
            
            axes[idx].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            axes[idx].set_title(f'Распределение: {col}')
        
        # Скрываем лишние оси
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/class_imbalance.png', dpi=150, bbox_inches='tight')
        plt.close()


# Функции-скиллы для прямого использования
def detect_issues(df: pd.DataFrame, **kwargs) -> QualityReport:
    """Скилл для обнаружения проблем качества данных."""
    agent = DataQualityAgent()
    return agent.detect_issues(df, **kwargs)


def fix_data(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """Скилл для исправления проблем качества данных."""
    agent = DataQualityAgent()
    return agent.fix(df, strategy)


def compare_data(df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
    """Скилл для сравнения данных до и после."""
    agent = DataQualityAgent()
    return agent.compare(df_before, df_after)


if __name__ == '__main__':
    # Пример использования
    print("DataQualityAgent - Агент для контроля качества данных")
    print("Используйте методы detect_issues(), fix(), compare()")
