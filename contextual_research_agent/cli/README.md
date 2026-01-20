## Примеры использования cli commands для Датасетов:

### Скачивание статей

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

#### Скачать LaTeX статей

```bash
python main.py download-sources --name="mvp-v1"
```

#### Докачать недостающие статьи

```bash
python main.py resume-download --name="mvp-v1" --verify-s3
```

`--verify-s3` - проверяет есть ли статьи на s3, если нет, то докачивает их

### Извлечение контента статей

#### Извлечь текст из всех PDF в датасете

```bash
python main.py extract-texts --dataset="mvp-v1"
```

#### Извлечь только первые 100

```bash
python main.py extract-texts --dataset="mvp-v1" --limit=100
```

#### Посмотреть статистику

```bash
python main.py extraction-stats --dataset="mvp-v1"
```

#### Посмотреть извлечённый текст конкретной статьи

```bash
python main.py show-text --arxiv-id="2401.12345"
```
