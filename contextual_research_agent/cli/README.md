## Примеры использования cli commands для Датасетов:

#### Создать датасет со скачиванием статей

```bash
python main.py create-dataset \
    --name="mvp-v1" \
    --total=1000 \
    --categories="cs.CL,cs.LG" \
    --min-date="2023-01-01" \
    --keywords="language model,transformer,retrieval,embedding"
```

#### Создать без скачивания

```bash
python main.py create-dataset --name="mvp-v2" --total=500 --no-download
```

#### Список датасетов

```bash
python main.py list-datasets
```

#### Детали датасета

```bash
python main.py show-dataset --name="mvp-v1"
```

#### Скачать локально

```bash
python main.py download-dataset --name="mvp-v1"
```

#### Докачать недостающие статьи

```bash
python main.py resume-download --name="mvp-v1" --verify-s3
```

`--verify-s3` - проверяет есть ли статьи на s3, если нет, то докачивает их


## Команды для агента

#### Суммаризация скачанной статьи
```bash
python main.py summarize 1603.03788_9878b2e9db4cb82d --top-k 5
```

#### QA
```bash
python main.py query "What is a transformer in LLM?"
```

#### Чат
```bash
python main.py chat
```

#### Статистика
```bash
python stats
```


## Общий пайплайн
```bash
python main.py create-dataset --name="baseline-v1" --total=500 --no-download --min_date="2025-01-01" # создание датасета
python main.py resume-download --name="baseline-v1" --verify-s3 # скачивание датасета


```

## Ingestion
```bash
python main.py ingest-file s3://paper.pdf   # Заиндексировать статью из s3
python main.py ingest-dataset baseline-v1   # Заиндексировать весь датасет
python main.py ingest-status baseline-v1    # Проверить статус
python main.py reingest-failed baseline-v1  # Доиндексировать упавшие
```