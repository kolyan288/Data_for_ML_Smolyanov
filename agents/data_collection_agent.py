# data_collection_agent.py

"""
DataCollectionAgent - агент для сбора данных из различных источников.
Поддерживает HuggingFace datasets, Kaggle, веб-скрапинг и API.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Any
import yaml
import logging
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCollectionAgent:
    """
    Агент для сбора и унификации данных из различных источников.
    
    Поддерживаемые источники:
    - HuggingFace datasets
    - Kaggle datasets
    - Веб-скрапинг
    - API endpoints
    """
    
    # Стандартная схема выходного датасета
    STANDARD_COLUMNS = [
        'id',             # Уникальный идентификатор
        'text',           # Текстовое описание (описание недвижимости)
        'price',          # Цена
        'price_currency', # Валюта цены
        'location',       # Местоположение (район)
        'city',           # Город
        'property_type',  # Тип недвижимости
        'area_sqm',       # Площадь в кв.м
        'living_area_sqm',   # Жилая площадь
        'kitchen_area_sqm',  # Площадь кухни
        'rooms',          # Количество комнат
        'floor',          # Этаж
        'total_floors',   # Всего этажей в доме
        'building_type',  # Тип здания
        'build_year',     # Год постройки
        'label',          # Категория (аренда/продажа)
        'source',         # Источник данных
        'collected_at',   # Время сбора
        'url',            # Ссылка на источник
        'date_posted',    # Дата публикации
        'metadata'        # Дополнительные метаданные
    ]
    
    def __init__(self, config: Optional[Union[str, Dict]] = None):
        """
        Инициализация агента.
        
        Args:
            config: Путь к YAML-файлу конфигурации или словарь с конфигурацией
        """
        self.config = self._load_config(config) if config else {}
        self.collected_data: List[pd.DataFrame] = []
        
    def _load_config(self, config: Union[str, Dict]) -> Dict:
        """Загрузка конфигурации из файла или словаря."""
        if isinstance(config, str):
            with open(config, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return config
    
    def run(self, sources: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Основной метод для сбора данных из указанных источников.
        
        Args:
            sources: Список источников данных с параметрами
            
        Returns:
            pd.DataFrame с унифицированной схемой
        """
        self.collected_data = []
        
        for source in sources:
            source_type = source.get('type')
            logger.info(f"Сбор данных из источника: {source_type}")
            
            try:
                if source_type == 'hf_dataset':
                    df = self._load_hf_dataset(source)
                elif source_type == 'kaggle_dataset':
                    df = self._load_kaggle_dataset(source)
                elif source_type == 'scrape':
                    df = self._scrape_data(source)
                elif source_type == 'api':
                    df = self._fetch_api(source)
                elif source_type == 'csv':
                    df = self._load_csv(source)
                else:
                    logger.warning(f"Неизвестный тип источника: {source_type}")
                    continue
                    
                if df is not None and not df.empty:
                    df = self._standardize_schema(df, source_type)
                    self.collected_data.append(df)
                    logger.info(f"Собрано {len(df)} записей из {source_type}")
                    
            except Exception as e:
                logger.error(f"Ошибка при сборе данных из {source_type}: {e}")
                continue
        
        if not self.collected_data:
            logger.warning("Не удалось собрать данные ни из одного источника")
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)
        
        # Объединение всех источников
        unified_df = self.merge(self.collected_data)
        logger.info(f"Итого собрано {len(unified_df)} записей")
        
        return unified_df
    
    def _load_hf_dataset(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка датасета из HuggingFace."""
        try:
            from datasets import load_dataset
            
            dataset_name = source.get('name')
            split = source.get('split', 'train')
            
            logger.info(f"Загрузка датасета {dataset_name} из HuggingFace")
            
            dataset = load_dataset(dataset_name, split=split)
            df = dataset.to_pandas()
            
            # Добавление метаданных
            df['source'] = f"hf:{dataset_name}"
            df['collected_at'] = datetime.now().isoformat()
            
            return df
            
        except ImportError:
            logger.error("Библиотека 'datasets' не установлена. Установите: pip install datasets")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки HF датасета: {e}")
            return None
    
    def _load_kaggle_dataset(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка датасета из Kaggle."""
        try:
            import kaggle
            
            dataset_name = source.get('name')
            file_name = source.get('file')
            
            logger.info(f"Загрузка датасета {dataset_name} из Kaggle")
            
            # Скачивание датасета
            kaggle.api.dataset_download_files(dataset_name, path='./temp_kaggle', unzip=True)
            
            # Поиск CSV файла
            temp_path = Path('./temp_kaggle')
            if file_name:
                csv_file = temp_path / file_name
            else:
                csv_files = list(temp_path.glob('*.csv'))
                if not csv_files:
                    logger.error("CSV файлы не найдены в датасете")
                    return None
                csv_file = csv_files[0]
            
            df = pd.read_csv(csv_file)
            
            # Добавление метаданных
            df['source'] = f"kaggle:{dataset_name}"
            df['collected_at'] = datetime.now().isoformat()
            
            # Очистка временных файлов
            import shutil
            shutil.rmtree('./temp_kaggle', ignore_errors=True)
            
            return df
            
        except ImportError:
            logger.error("Библиотека 'kaggle' не установлена. Установите: pip install kaggle")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки Kaggle датасета: {e}")
            return None
    
    def _scrape_data(self, source: Dict) -> Optional[pd.DataFrame]:
        """Веб-скрапинг данных."""
        url = source.get('url')
        selector = source.get('selector')
        headers = source.get('headers', {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        try:
            logger.info(f"Скрапинг данных с {url}")
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Если указан селектор, извлекаем элементы
            if selector:
                elements = soup.select(selector)
                data = []
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text:
                        data.append({'text': text, 'url': url})
                df = pd.DataFrame(data)
            else:
                # Извлечение всех текстовых данных
                texts = [t.strip() for t in soup.stripped_strings if len(t.strip()) > 50]
                df = pd.DataFrame({'text': texts, 'url': url})
            
            df['source'] = f"scrape:{url}"
            df['collected_at'] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка скрапинга: {e}")
            return None
    
    def _fetch_api(self, source: Dict) -> Optional[pd.DataFrame]:
        """Получение данных через API."""
        endpoint = source.get('endpoint')
        params = source.get('params', {})
        headers = source.get('headers', {})
        
        try:
            logger.info(f"Запрос к API: {endpoint}")
            
            response = requests.get(endpoint, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Преобразование JSON в DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Попытка найти массив данных в ответе
                for key in data:
                    if isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                logger.error("Неизвестный формат ответа API")
                return None
            
            df['source'] = f"api:{endpoint}"
            df['collected_at'] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка API запроса: {e}")
            return None
    
    def _load_csv(self, source: Dict) -> Optional[pd.DataFrame]:
        """Загрузка локального CSV файла."""
        file_path = source.get('path')
        
        try:
            logger.info(f"Загрузка CSV файла: {file_path}")
            
            df = pd.read_csv(file_path)
            df['source'] = f"csv:{file_path}"
            df['collected_at'] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV: {e}")
            return None
    
    def _standardize_schema(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """
        Приведение датасета к стандартной схеме.
        
        Args:
            df: Исходный DataFrame
            source_type: Тип источника данных
            
        Returns:
            DataFrame со стандартными колонками
        """
        # Создание нового DataFrame со стандартными колонками
        standardized = pd.DataFrame(columns=self.STANDARD_COLUMNS)
        
        # Копирование существующих колонок
        for col in df.columns:
            if col in self.STANDARD_COLUMNS:
                standardized[col] = df[col]
        
        # Заполнение отсутствующих колонок значениями по умолчанию
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                if col == 'collected_at':
                    standardized[col] = datetime.now().isoformat()
                elif col == 'source':
                    standardized[col] = source_type
                elif col == 'metadata':
                    standardized[col] = '{}'
                else:
                    standardized[col] = None
        
        return standardized
    
    def merge(self, sources: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Объединение нескольких DataFrame в один.
        
        Args:
            sources: Список DataFrame для объединения
            
        Returns:
            Объединенный DataFrame
        """
        if not sources:
            return pd.DataFrame(columns=self.STANDARD_COLUMNS)
        
        # Объединение всех источников
        merged = pd.concat(sources, ignore_index=True, sort=False)
        
        # Удаление дубликатов по URL или тексту
        if 'url' in merged.columns and merged['url'].notna().any():
            merged = merged.drop_duplicates(subset=['url'], keep='first')
        elif 'text' in merged.columns and merged['text'].notna().any():
            merged = merged.drop_duplicates(subset=['text'], keep='first')
        
        # Переупорядочивание колонок
        merged = merged[self.STANDARD_COLUMNS]
        
        return merged
    
    def save(self, df: pd.DataFrame, output_path: str, format: str = 'csv'):
        """
        Сохранение датасета в файл.
        
        Args:
            df: DataFrame для сохранения
            output_path: Путь для сохранения
            format: Формат файла (csv, parquet, json)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', force_ascii=False)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
        
        logger.info(f"Датасет сохранен: {output_path}")


# Функции-скиллы для прямого использования

def scrape(url: str, selector: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Скилл для веб-скрапинга.
    
    Args:
        url: URL для скрапинга
        selector: CSS селектор для извлечения элементов
        **kwargs: Дополнительные параметры
        
    Returns:
        DataFrame с извлеченными данными
    """
    agent = DataCollectionAgent()
    return agent._scrape_data({'url': url, 'selector': selector, **kwargs})


def fetch_api(endpoint: str, params: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
    """
    Скилл для получения данных через API.
    
    Args:
        endpoint: URL API endpoint
        params: Параметры запроса
        **kwargs: Дополнительные параметры
        
    Returns:
        DataFrame с данными из API
    """
    agent = DataCollectionAgent()
    return agent._fetch_api({'endpoint': endpoint, 'params': params or {}, **kwargs})


def load_dataset(name: str, source: str = 'hf', **kwargs) -> pd.DataFrame:
    """
    Скилл для загрузки датасета.
    
    Args:
        name: Название датасета или путь к файлу
        source: Источник ('hf', 'kaggle', 'csv')
        **kwargs: Дополнительные параметры
        
    Returns:
        DataFrame с загруженными данными
    """
    agent = DataCollectionAgent()
    
    if source == 'hf':
        return agent._load_hf_dataset({'name': name, **kwargs})
    elif source == 'kaggle':
        return agent._load_kaggle_dataset({'name': name, **kwargs})
    elif source == 'csv':
        return agent._load_csv({'path': name, **kwargs})
    else:
        raise ValueError(f"Неподдерживаемый источник: {source}")


def merge_datasets(sources: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Скилл для объединения датасетов.
    
    Args:
        sources: Список DataFrame для объединения
        
    Returns:
        Объединенный DataFrame
    """
    agent = DataCollectionAgent()
    return agent.merge(sources)


if __name__ == '__main__':
    # Пример использования
    agent = DataCollectionAgent()
    
    # Определение источников данных
    sources = [
        {
            'type': 'hf_dataset',
            'name': 'imdb',
            'split': 'train[:1000]'
        }
    ]
    
    # Запуск сбора данных
    df = agent.run(sources)
    print(f"Собрано {len(df)} записей")
    print(df.head())
