# Ingestion Pipeline

## Назначение

Модуль ingestion реализует полный цикл преобразования научных PDF-статей в индексированные, обогащённые векторные представления для системы Retrieval-Augmented Generation (RAG). Pipeline принимает PDF-документ и выполняет цепочку: парсинг → чанкинг → обогащение (section classification + citation extraction) → эмбеддинг → индексация → сохранение графовых связей.

Модуль спроектирован как **data pipeline**, отделённый от agent-уровня. Это позволяет:

- запускать ingestion независимо от LLM/agent инфраструктуры;
- проводить ablation studies по параметрам ingestion без изменения agent логики;
- масштабировать ingestion горизонтально (batch, concurrent).
- строить knowledge graph (citation + entity edges) параллельно с индексацией.

### Архитектурное решение: почему не LangGraph

Ingestion pipeline — линейная цепочка (parse → chunk → embed → index) без ветвлений, циклов и tool calls. LangGraph добавил бы overhead сериализации state, потерю типизации (`state.get("document")` → `Any`) и сложность compile/invoke без практической пользы. Pipeline реализован как typed async chain с explicit dataclass результатами на каждой стадии. LangGraph зарезервирован для agent orchestration, где нелинейный граф (multi-mode routing, evaluation loops) оправдывает графовую абстракцию.

## Архитектура

```
                         PDF (S3)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: DoclingParser.parse()                              │
│  PDF → DoclingDocument (sections, tables, figures)           │
│  → Document + ParseResult                                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: DoclingParser.extract_chunks()                     │
│  HybridChunker (hierarchical + merge_peers + context)        │
│  Serializers: MarkdownTable + AnnotationPicture              │
│  Formula placeholder replacement                             │
│  → list[Chunk] + ChunkingMetrics                             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Enrichment                                         │
│                                                              │
│  3a. SectionClassifier (rule-based, 15 types)                │
│      direct match → parent inheritance → propagation         │
│                                                              │
│  3b. CitationExtractor                                       │
│      bib_entries / markdown fallback (4 formats)             │
│      arxiv_id + DOI regex resolution                         │
│      inline citation context extraction                      │
│      → CitationEdge[] + ExtractionMetrics                    │
│                                                              │
│  3c. EntityExtractor (LLM-based)                             │
│      section-aware prompting (llama-server / Ollama)         │
│      representative chunk sampling                           │
│      → ExtractedEntity[] + EntityEdge[]                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: HuggingFaceEmbedder                                │
│  4a. Chunk embeddings (all chunks → Qdrant "documents")      │
│  4b. Paper-level embedding (title+abstract → Qdrant "papers")│
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 5: QdrantStore (chunk index + paper index)            │
│  Deterministic point IDs, payload indexes, retry + backoff   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 6: KnowledgeGraphRepository (PostgreSQL)              │
│  6a. citation_edges (Paper → Paper)                          │
│  6b. entities + paper_entity_edges (Paper → Entity)          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                 IngestionResult + IngestionMetrics
                           │
                     ┌─────┴─────┐
                     ▼           ▼
               MLflow        JSON report
```

## Структура модуля

```
ingestion/
├── parsers/
│   ├── base.py              # DocumentParser ABC
│   ├── config.py            # DoclingParserConfig, ChunkType (без циклических импортов)
│   ├── metrics.py           # ChunkingMetrics, compute/aggregate
│   ├── serializers.py       # ScientificSerializerProvider, AnnotationPictureSerializer
│   └── docling.py           # DoclingParser, ParseResult, create_docling_parser
├── extraction/
│   ├── section_classifier.py  # SectionType, SectionClassifier
│   ├── citation_extractor.py  # CitationExtractor, CitationEdge, BibEntry
│   └── entity_extractor.py    # EntityExtractor, LLMClient, ExtractedEntity
├── embeddings/
│   ├── base.py              # Embedder ABC
│   └── hf_embedder.py       # HuggingFaceEmbedder, EmbeddingMetrics
├── vectorstores/
│   └── qdrant_store.py      # QdrantStore, StoreOperationMetrics
├── domain/
│   ├── entities.py          # Document, Chunk, RetrievedChunk, Citation
│   └── types.py             # DocumentStatus enum
├── pipeline.py              # IngestionPipeline, IngestionResult, BatchResult
├── result.py                # ExtractionMetrics, IngestionMetrics, StageLatency
├── tracking.py              # IngestionTracker (MLflow)
└── analytics.py             # IngestionAnalytics, CorpusAnalyticsReport

db/
├── repositories/
│   ├── datasets.py          # DatasetsRepository
│   └── knowledge_graph.py   # KnowledgeGraphRepository
└── migrations/
```

## Компоненты

### DoclingParser

Парсер на основе IBM Docling. Выполняет две фазы:

1. **Parse**: PDF → DoclingDocument (структурированное представление с labeled items: section headers, paragraphs, tables, figures). Метод возвращает `ParseResult`, содержащий сериализуемый `Document` и in-memory `DoclingDocument`. DoclingDocument **не** хранится в `Document.metadata` во избежание утечек памяти и проблем сериализации.

2. **Extract chunks**: HybridChunker с иерархическим разбиением по секциям и контролем длины в токенах embedding-модели. Таблицы сериализуются в Markdown (`MarkdownTableSerializer`), рисунки — через аннотации Docling (`AnnotationPictureSerializer`).

**Formula handling**: Latex-формулы пока не поддерживаются, так как требуют отдельную OCR (Docling) или VLM (Nougat) модель, которая требует дополнительных ресурсов, поэтому пока решено оставить только Unicode-формулы. Inline-формулы (Unicode: `F(q) ≜ E(q) + H(q)`) сохраняются как текст.

### SectionClassifier

Rule-based классификатор типа секции по заголовку. Определяет 15 семантических типов: `TITLE`, `ABSTRACT`, `INTRODUCTION`, `RELATED_WORK`, `BACKGROUND`, `METHOD`, `EXPERIMENTS`, `RESULTS`, `DISCUSSION`, `CONCLUSION`, `LIMITATIONS`, `ETHICS`, `APPENDIX`, `REFERENCES`, `UNKNOWN`.

Три стратегии классификации, применяемые последовательно:

1. **Direct match**: нормализованный заголовок → regex паттерны . Пример: `"3.1 Experimental Setup"` → normalize → `"experimental setup"` → EXPERIMENTS.

2. **Parent section inheritance**: если `"2.1. Problem Setup"` → UNKNOWN, извлекается номер секции `"2.1"` → parent `"2"` → heading `"2. Background"` → BACKGROUND. Поддерживает multi-level: `"3.2.1"` → `"3.2"` → `"3"`.

3. **Propagation**: если всё ещё UNKNOWN, наследуется тип от предыдущего чанка. Корректно для последовательного порядка чанков в документе.

Секция `section_type` хранится в Qdrant payload → позволяет **task-aware retrieval**: для critique → фильтр по `METHOD + EXPERIMENTS`, для survey → `RELATED_WORK + INTRODUCTION`, для comparison → `RESULTS + EXPERIMENTS`.

### CitationExtractor

Извлечение и резолвинг цитат из научных статей.

**Extraction pipeline**:

1. **Parse bibliography**: из `DoclingDocument.bib_entries` (structured) → если недоступно → fallback: парсинг References секции из markdown. Универсальный splitter поддерживает 4 формата: bullet lists (`- Author...`), numbered brackets (`[1] Author...`), numbered dot (`1. Author...`), paragraph-based.

2. **Title extraction**: из raw reference string. Детектирует конец author block по паттерну "инициала + точка + пробел + слово ≥3 букв" (`"...Petzold, L. Selecting the Metric..."`). Handles year markers `(2021).` и `, 2021.`. Strips trailing page references `(pages 1, 2, 3)`.

3. **Identifier resolution**: regex для arxiv ID (из URL, bare text, `arXiv:XXXX.XXXXX`), DOI.

4. **Context extraction**: находит inline citation anchors (`[12]`, `[1-3]`, `[1, 3, 5]`) в chunk text.

**Output**: `CitationEdge[]` — направленные рёбра (citing_paper → cited_paper) с контекстом, секцией, метаданными.

### EntityExtractor

LLM-based извлечение научных сущностей из текста статьи. Используется llama-cpp или ollama.

**Entity types**: method, dataset, task, metric, model → relation mapping: `uses_method`, `uses_dataset`, `targets_task`, `reports_metric`, `uses_model`.

**Pipeline**:
1. Группировка чанков по section_type
2. Representative chunk sampling (top-3 по длине текста, без equation chunks) → сокращает LLM вызовы в 2-3x
3. Section-aware prompting (разные hint'ы для METHOD vs EXPERIMENTS vs RESULTS)
4. Structured JSON output → parse → deduplication по `(normalized_name, entity_type)`

**LLM backends**: `LlamaCppProviderAdapter` (llama-server), `OllamaProviderAdapter` (Ollama). Модель: Qwen3-4B (Q4_K_M).

### Paper-level Embeddings

Отдельная Qdrant collection `{collection}_papers` с embedding от `title + abstract`. Two-stage retrieval: сначала найти релевантные статьи (paper-level), затем chunk-level.

### Knowledge Graph (PostgreSQL)

Три таблицы для графовых связей:

**`entities`** — научные концепты (methods, datasets, tasks, metrics, models) с дедупликацией по `(normalized_name, entity_type)`.

**`citation_edges`** — Paper → Paper с контекстом цитирования: предложение, секция, тип секции. Позволяет отвечать "кто цитирует эту работу и в каком контексте".

**`paper_entity_edges`** — Paper → Entity: `uses_method`, `uses_dataset`, `targets_task`. Confidence score для оценки качества extraction.

Граф хранится в PostgreSQL — recursive CTE.

### HuggingFaceEmbedder

Эмбеддинг-провайдер на SentenceTransformers. Поддерживает два механизма инструктирования:

- **prompt_name**: для моделей с встроенными prompt-шаблонами (Qwen3-Embedding) — модель сама форматирует prompt;
- **prefix**: конкатенация строки-инструкции (BGE, E5).

Выбор механизма в `_MODEL_CONFIGS`. Пустые тексты заменяются на `"[empty]"` для сохранения index alignment `chunks[i] ↔ embeddings[i]`.

### QdrantStore

Payload indexes по `document_id`, `chunk_type`, `section`, `is_fallback`, `is_degenerate` позволяют фильтрованный поиск. 

### IngestionTracker

MLflow интеграция. Логирует конфигурацию как params, метрики стадий как metrics, отчёты как artifacts. Поддерживает single и batch режимы.

## Параметры, влияющие на качество

### Параметры чанкинга

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `max_tokens` | `--max-tokens` | 512 | Размер чанка в токенах. Больше → больше контекста, ниже precision при retrieval. Меньше → точнее retrieval, но возможна потеря контекста. |
| `merge_peers` | `--no-merge-peers` | True | Объединение соседних чанков одного уровня иерархии. True → когерентные чанки (целые параграфы). False → гранулярные чанки (отдельные элементы). |
| `include_context` | `--no-context` | True | Добавление иерархических заголовков секций в начало чанка. Улучшает retrieval для query вида "method in paper X", увеличивает overhead на 15–30% токенов. |
| `filter_empty_chunks` | `--filter-empty` | False | Удаление вырожденных чанков (< min_chunk_tokens). False — сохраняет для анализа в метриках. True — убирает шум в production. |
| `min_chunk_tokens` | `--min-chunk-tokens` | 20 | Порог вырожденности. Чанки ниже порога помечаются `is_degenerate=True`. |
| `do_table_structure` | — | True | Распознавание структуры таблиц. Без этого таблицы сериализуются как плоский текст. Всегда True для научных статей. |
| `do_ocr` | — | False | OCR для отсканированных страниц. Для arXiv PDF (цифровые) не нужен, замедляет в ~5x. |

### Параметры сериализации

| Параметр | Влияние |
|---|---|
| Table serializer (Markdown vs triplet) | Markdown сохраняет структуру строк/столбцов, triplet компактнее, но теряет layout. Для LLM Markdown предпочтительнее. |
| Picture serializer (annotation vs placeholder) | Annotation включает описание image (тип, caption) в текст чанка. Placeholder — только метку `[Figure]`. Annotation улучшает retrieval по визуальному контенту (архитектуры, графики). |

### Параметры эмбеддинга

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `embedding_model` | `--embedding-model` | Qwen/Qwen3-Embedding-0.6B | Определяет качество семантического представления. Токенизатор модели должен совпадать с токенизатором чанкинга. |
| `normalize` | — | True | L2-нормализация. Необходима для cosine similarity. |
| `batch_size` | `--batch-size` | 32 | Batch size для GPU inference. Влияет на throughput, не на качество. |

### Параметры индексации

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `distance` | `--distance` | cosine | Метрика расстояния. Cosine — стандарт для нормализованных эмбеддингов. Dot product — если модель не нормализует. |
| `on_disk` | `--on-disk` | False | Хранение векторов на диске. Экономит RAM, замедляет поиск. |
| `collection` | `--collection` | documents | Имя Qdrant collection. Для ablation — уникальное имя на каждую конфигурацию. |

## CLI-интерфейс

### Одиночный файл

```bash
python main.py ingest-file <s3_path> [options]
```

Полный список опций:

```bash
python main.py ingest-file s3://rag-storage/arxiv/papers/2401.12345.pdf \
    --embedding-model=Qwen/Qwen3-Embedding-0.6B \
    --max-tokens=512 \
    --collection=documents \
    --distance=cosine \
    --batch-size=32 \
    --no-merge-peers \     # отключить merge
    --no-context \         # отключить section headings в чанках
    --filter-empty \       # удалять вырожденные чанки
    --min-chunk-tokens=20 \
    --device=cuda \
    --no-tracking          # отключить MLflow
```

### Batch по датасету

```bash
python main.py ingest-dataset <name> [options]
```

Дополнительные опции для batch:

```bash
python main.py ingest-dataset baseline-v1 \
    --split=train \        # только train split
    --limit=10 \           # первые 10 файлов (для тестирования)
    --max-concurrent=1 \   # параллелизм (1=sequential, safe для GPU)
    --continue-on-error    # не останавливаться при ошибках
```

### Статус и retry

```bash
# Проверить прогресс ingestion
python main.py ingest-status baseline-v1 --collection=documents

# Повторить упавшие из предыдущего запуска
python main.py reingest-failed baseline-v1

# Retry с другими параметрами
python main.py reingest-failed baseline-v1 --max-tokens=256 --collection=exp_256
```

## Сбор метрик

Метрики собираются на каждой стадии и агрегируются в `IngestionMetrics`:

```
IngestionResult
└── metrics: IngestionMetrics
    ├── latency: StageLatency
    │   ├── parse_ms, chunk_ms, embed_ms, index_ms, total_ms
    │   └── (pipeline собирает сам)
    ├── chunking: ChunkingMetrics
    │   ├── Distributional: mean/median/std/p5/p95 token lengths
    │   ├── Quality: empty_chunk_count, oversized_chunk_count
    │   ├── Type distribution: text/table/figure/code/equation
    │   ├── Structural: section_coverage_ratio, context_overhead
    │   └── (из parser.extract_chunks)
    ├── embedding: EmbeddingMetrics
    │   ├── duration_ms, throughput_texts_per_sec
    │   ├── empty_text_count, mean_text_length
    │   └── (из embedder._encode)
    └── indexing: StoreOperationMetrics
        ├── duration_ms, num_items, success
        └── (из store.add_chunks)
```

Corpus-level агрегация через `aggregate_corpus_metrics()`:

```
BatchResult → per-document ChunkingMetrics[] → aggregate:
  corpus_mean_tokens, corpus_std_tokens, corpus_p5, corpus_p95,
  total_empty_chunks, total_oversized_chunks, empty_rate, oversized_rate,
  fallback_rate, mean_section_coverage, mean_context_overhead
```

## Отчёты

### JSON reports

Каждый ingestion run сохраняет JSON report:

```
reports/
├── ingestion/
│   └── <document_id>.json       # single file reports
├── ingestion_baseline-v1.json   # batch report
├── ingestion_baseline-v1_train.json  # batch по split
└── ingestion_baseline-v1_retry.json  # retry report
```

Batch report содержит полный `IngestionResult.to_dict()` для каждого документа, включая все метрики. Используется командой `reingest-failed` для определения упавших файлов.

### MLflow artifacts

| Artifact | Содержание |
|---|---|
| `batch_report.json` | Полный отчёт (дублирует JSON report) |
| `per_document_metrics.csv` | Таблица: document_id, chunk_count, latency per stage, chunking quality |
| `result.json` | Для single file runs |

## Известные ограничения и направления улучшения

### Formula parsing

Display-формулы не декодируются базовым PDF-парсером. Docling извлекает их как `<!-- formula-not-decoded -->`, pipeline заменяет на свой маркер. Inline-формулы (Unicode) сохраняются как текст.

**Варианты улучшения**: Docling `do_formula_enrichment=True` (OCR), Nougat (VLM для PDF → LaTeX), Mathpix API.

### Citation resolution rate

Regex-based решение дает низкий rate, так как парсинг идет по arxiv ID и DOI в тексте. 

**Варианты улучшения**: Semantic Scholar API (title-based search), local corpus matching (title → arxiv_id из БД).

### Entity extraction quality

LLM-based extraction зависит от модели и prompt. Qwen3-4B (Q4_K_M) — baseline качество. Возможные проблемы: hallucination entities, пропуск implicit entities, нестабильный JSON output.

**Варианты улучшения**: более крупная модель (7B, 14B), fine-tuning на gold annotations, ансамбль из нескольких LLM calls.

### Что не реализовано

- **Sparse index (BM25)**: для hybrid retrieval (dense + sparse fusion).
- **Figure understanding**: рисунки извлекаются как placeholder, содержание не анализируется. Для диаграмм архитектур и графиков результатов это потеря информации.
- **Cross-document entity resolution**: "BERT" в одной статье и "bert-base-uncased" в другой — разные entity. Нужна нормализация.
- **Metadata enrichment из Semantic Scholar**: citation_count, influential_citation_count.

## Воспроизводимость

Полная воспроизводимость эксперимента обеспечивается фиксацией:

1. **Dataset config** (YAML): categories, date range, random seed, split ratios → определяет набор документов.
2. **Ingestion params** (MLflow params): embedding_model, max_tokens, merge_peers, include_context, distance → определяет содержимое Qdrant collection.
3. **Pipeline metrics** (MLflow metrics + JSON report): полная диагностика каждого документа.

Комбинация dataset config + ingestion params однозначно определяет содержимое Qdrant collection и может быть воспроизведена повторным запуском с теми же флагами.