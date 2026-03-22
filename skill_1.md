Агент 1

Необхоимо аписать агента, который умеет собирать данные из нескольких источников и возвращает унифицированный датасет. 

АРХИТЕКТУРА АГЕНТА

- DataCollectionAgent
- skill: scrape(url, selector) → DataFrame
- skill: fetch_api(endpoint, params) → DataFrame
- skill: load_dataset(name, source='hf'|'kaggle') → DataFrame
- skill: merge(sources: list[DataFrame]) → DataFrame 

ТЕХНИЧЕСКИЙ КОНТРАКТ 

from data_collection_agent import DataCollectionAgent 
agent = DataCollectionAgent(config='config.yaml') 
df = agent.run(sources=[ 
    {'type': 'hf_dataset', 'name': 'imdb'}, 
    {'type': 'scrape', 'url': '...', 'selector': '...'}, 
]) 
# → pd.DataFrame со стандартными колонками: 
# text/audio/image, label, source, collected_at 

ТРЕБОВАНИЯ

- Минимум 2 источника данных (один — open dataset HuggingFace/Kaggle, один — scraping или API)
- Унифицированная схема выходного датасета: фиксированные колонки для всех источников
- EDA: распределение классов, длины текстов / длительность аудио, топ-20 слов / спектрограмма
- README.md с описанием задачи ML, схемы данных и инструкцией по запуску
- requirements.txt или pyproject.toml 

СТРУКТУРА РЕПОЗИТОРИЯ

- agents/data_collection_agent.py — основной файл агента
- config.yaml — конфигурация источников
- notebooks/eda.ipynb — EDA и визуализации
- data/raw/ — собранные данные
- README.md

Нужно написать и запустить тебе код на Python, который собирает из всего интернета данные по  @ТЕМА@. Пройдись по открытым источникам - сайтам https://huggingface.co/ и kaggle.com,  а также специализированным сайтам, которые соответствуют @ТЕМЕ@ - @САЙТЫ@. Если нет доступа к сайтам, поищи готовые проекты на github.com и возьми код оттуда. На выходе - csv таблицы с собранным датасетом по недвижимости Санкт-Петербурга. Учти, что в датасете должно быть минимум 5000 записей

@ТЕМА@ == недвижимость (аренда и продажа) в Санкт-Петербурге

@САЙТЫ@ == avito.ru, spb.cian.ru