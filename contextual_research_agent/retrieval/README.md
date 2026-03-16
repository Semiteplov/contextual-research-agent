# Retrieval Module

## Назначение

Модуль реализует multi-channel retrieval pipeline для системы RAG. Pipeline принимает текстовый запрос и возвращает ранжированный набор чанков с метриками каждой стадии.

## Архитектура

```
                    Query
                      │
                      ▼
┌──────────────────────────────────────────┐
│  Stage 0: QueryAnalyzer                  │
│  intent detection → section routing      │
│  → channel weight overrides              │
│  → QueryPlan                             │
└─────────────────────┬────────────────────┘
                      │
                      ▼
            query embedding (shared)
                      │
                      ▼
┌──────────────────────────────────────────┐
│  Stage 1: Multi-Channel (parallel)       │
│                                          │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐   │
│  │  Dense   │ │ Sparse  │ │  Graph   │   │
│  │ (Qdrant) │ │ (BM25)  │ │(Citation)│   │
│  └────┬─────┘ └────┬────┘ └────┬─────┘   │
│       │            │           │         │
│  ┌────┴─────┐ ┌────┴────┐      │         │
│  │  Paper   │ │  Graph  │      │         │
│  │  Level   │ │(Entity) │      │         │
│  └────┬─────┘ └─────┬───┘      │         │
└───────┼─────────────┼──────────┼─────────┘
        │  list[ScoredCandidate] per channel
        ▼
┌──────────────────────────────────────────┐
│  Stage 2: Fusion (RRF / Weighted)        │
│  channel contribution tracking           │
│  overlap analysis                        │
└──────────────┬───────────────────────────┘
               │  top-N candidates
               ▼
┌──────────────────────────────────────────┐
│  Stage 3: Cross-Encoder Reranker         │
│  (query, chunk.text) → relevance score   │
│  rank change tracking                    │
└──────────────┬───────────────────────────┘
               │  top-K candidates
               ▼
┌──────────────────────────────────────────┐
│  Stage 4: ContextAssembler               │
│  deduplication → ordering                │
│  → formatted context string              │
└────────────────────┬─────────────────────┘
                     │
                     ▼
         RetrievalResult
           ├── candidates: list[ScoredCandidate]
           ├── context: str
           ├── channel_results, fusion_result, rerank_result
           └── metrics
```

## Структура модуля

```
retrieval/
├── __init__.py              # Public API, re-exports
├── config.py                # RetrievalConfig
├── types.py                 # ScoredCandidate, ChannelResult, RetrievalResult
├── metrics.py               # Operational + IR quality metrics
├── pipeline.py              # RetrievalPipeline (orchestrator)
├── channels/
│   ├── __init__.py          # RetrievalChannel ABC
│   ├── dense.py             # DenseChannel (cosine)
│   ├── sparse.py            # SparseChannel (BM25/SPLADE)
│   ├── graph.py             # CitationGraphChannel, EntityGraphChannel
│   └── paper_level.py       # PaperLevelChannel (paper → chunk expansion)
├── fusion/
│   └── __init__.py          # RRF + WeightedScoreFusion
├── reranking/
│   └── __init__.py          # CrossEncoderReranker
├── query/
│   └── __init__.py          # QueryAnalyzer (intent → routing)
└── context/
    └── __init__.py          # ContextAssembler (deduplication, ordering, budget)
```

## Компоненты

### QueryAnalyzer

Определяет intent запроса и маппит его на retrieval поведение:

| Intent | Section filter | Channel weight override | Пример запроса |
|---|---|---|---|
| `factual_qa` | — | — | "What learning rate was used?" |
| `method_explanation` | method, background | — | "How does LoRA work?" |
| `comparison` | results, experiments | entity | "Compare LoRA vs full fine-tuning" |
| `critique` | method, experiments, limitations | — | "What are the weaknesses?" |
| `survey` | related_work, introduction | citation, paper_level | "Overview of PEFT methods" |
| `citation_trace` | — | citation | "Who cites this paper?" |

Baseline: rule-based (regex). Upgrade path: LLM-based classification.

### DenseChannel

Bi-encoder (Qwen3-Embedding-0.6B) → Qdrant cosine similarity. Поддерживает section_type pre-filtering через Qdrant payload filters.

### SparseChannel

BM25 для exact match (названия методов, аббревиатуры, числа). Использует Qdrant native sparse vectors через `fastembed` BM25 encoder.

### CitationGraphChannel

1. Dense search → top-K seed papers.
2. PostgreSQL `citation_edges` → bidirectional walk (citing + cited) с recursive CTE.
3. Qdrant filtered search по найденным paper_ids.
4. Score: chunk_score × decay (0.8).

### EntityGraphChannel

1. Keyword matching query → `entities` table (case-insensitive substring).
2. `paper_entity_edges` → papers connected to matched entities, ranked by connection count.
3. Qdrant filtered search по найденным paper_ids.

### PaperLevelChannel

Two-stage: `papers` collection (title+abstract embeddings) → top papers → chunk expansion через `documents` collection с `document_id` filter. Combined score: `0.3 * paper_score + 0.7 * chunk_score`.

### Fusion

**RRF** (default): `RRF_score(d) = Σ_c w_c / (k + rank_c(d))`, k=60.
**WeightedScoreFusion** : min-max нормализация per channel → weighted sum.

Собирают:
- `channel_contributions`: сколько кандидатов из каждого канала попало в top-N.
- `channel_overlaps`: пересечение кандидатов между парами каналов.

### CrossEncoderReranker

Cross-encoder reranking top-N candidates после fusionю

### ContextAssembler

Post-processing: deduplication → ordering → formatted string.

Два режима ordering:
- `document_then_chunk`: группировка по документу (by best score), внутри — по chunk_index. Сохраняет структуру документа.
- `score`: строго по score. Максимизирует релевантность вверху контекста.

## Hybrid Vector Store
 
`QdrantStore` поддерживает два режима:
 
**Dense-only**
 
**Hybrid**

Методы поиска:
- `search()` — dense vector search.
- `search_sparse()` — sparse vector search.
- `search_hybrid()` — Qdrant-native RRF fusion через `prefetch`.

## Метрики

### Operational (каждый запрос, пока без ground truth)

```
RetrievalOperationalMetrics
├── timing/
│   ├── total_ms, query_analysis_ms
│   ├── channels_ms (max across parallel channels)
│   ├── fusion_ms, rerank_ms, context_assembly_ms
├── counts/
│   ├── pre_fusion, post_fusion, post_rerank, final
│   └── unique_documents
├── per_channel/
│   ├── latency_ms, num_candidates
│   └── min/max/mean_score
├── fusion/
│   ├── channel_contributions (channel → count in top-N)
│   └── channel_overlaps (pair → overlap count)
└── rerank/
    └── mean_rank_change (vs pre-rerank ordering)
```

### IR Quality (требуют ground-truth)

Per-query: `Recall@K`, `Precision@K`, `NDCG@K`, `Hit Rate@K`, `MAP@K`, `MRR`.
K ∈ {1, 3, 5, 10} по умолчанию.

Corpus-level: `mean ± std` для каждой метрики + `p95_latency_ms`.

Все метрики сериализуются в MLflow через `.to_mlflow_metrics()`.

### MLflow Tracking
 
`RetrievalTracker` логирует evaluation runs:
- `config.to_mlflow_params()` → MLflow params.
- `agg_metrics.to_mlflow_metrics()` → MLflow metrics.
- Per-query и operational metrics → JSON artifacts.
- Tags: channels, eval_set path.

## Использование

## CLI
 
### Retrieve (single query)
 
```bash
# Dense only, no reranker
python main.py retrieval-retrieve "How does LoRA reduce training cost?" \
    --collection=documents_test \
    --top-k=5 \
    --no-rerank \
    --channels=dense \
    --verbose
 
# Dense + reranker
python main.py retrieval-retrieve "How does LoRA reduce training cost?" \
    --collection=documents_test \
    --top-k=5 \
    --rerank \
    --rerank-model=BAAI/bge-reranker-v2-m3 \
    --channels=dense
 
# Multi-channel (dense + graph)
python main.py retrieval-retrieve "What datasets were used for evaluation?" \
    --collection=documents_test \
    --channels=dense,graph_citation,graph_entity \
    --rerank \
    --verbose
 
# Document-scoped
python main.py retrieval-retrieve "What is the proposed method?" \
    --collection=documents_test \
    --document=2106.09685 \
    --top-k=5
```
 
### Evaluation
 
```bash
# Generate eval set from corpus
python main.py generate-eval-set \
    --collection=documents_test \
    --num-queries=20 \
    --output=eval/test_v1.json
 
# Run evaluation → IR metrics + MLflow
python main.py retrieval-evaluate eval/test_v1.json \
    --collection=documents_test \
    --channels=dense \
    --no-rerank \
    --experiment-name=retrieval_test \
    --run-name=dense_only
 
# Compare: with reranker
python main.py retrieval-evaluate eval/test_v1.json \
    --collection=documents_test \
    --channels=dense \
    --rerank \
    --experiment-name=retrieval_test \
    --run-name=dense_rerank
 
# Full pipeline
python main.py retrieval-evaluate eval/test_v1.json \
    --collection=documents_test \
    --channels=dense,graph_citation,graph_entity \
    --rerank \
    --experiment-name=retrieval_test \
    --run-name=full_pipeline
```

## Что не реализовано
 
### Query Expansion / Rewriting
 
Multi-query retrieval: LLM переформулирует исходный query в 3-5 вариантов, каждый проходит через channels, результаты объединяются. Увеличивает recall за счёт разных формулировок. Архитектурно предусмотрено (`QueryPlan.expanded_queries` field существует), но логика в pipeline не реализована.
 
### Contextual Compression
 
Post-retrieval extraction: LLM выделяет из чанка только релевантные предложения, отбрасывая шум. Уменьшает context window usage, повышает precision подаваемого контекста. Требует дополнительный LLM call per chunk.
 
### Adaptive Retrieval
 
Routing на основе query complexity: простые factual queries → только dense (fast path), сложные analytical queries → full multi-channel pipeline. Экономит latency на простых запросах. Реализуемо через extension QueryAnalyzer.
 
### Late Interaction Models (ColBERT)
 
Token-level similarity вместо single-vector similarity. Значительно выше качество retrieval при сопоставимой latency, но требует специальный индекс (не Qdrant native). Альтернатива: ColBERT-based reranker вместо cross-encoder.
 
### Learned Sparse Representations (SPLADE)
 
Альтернатива BM25 для sparse channel. SPLADE обучен на retrieval tasks и даёт лучшее качество чем lexical BM25.
 
### Cross-Document Chunk Merging
 
При retrieval из нескольких документов — объединение смежных чанков из одного документа для восстановления контекста. Сейчас каждый чанк независим, даже если retriever вернул chunk[5] и chunk[6] из одного документа.
 
### Feedback Loop
 
User feedback (thumbs up/down) на retrieved chunks → fine-tuning fusion weights или reranker. Требует UI integration и storage для feedback data.
 
### Caching Layer
 
Query embedding cache (LRU) и result cache с TTL. Redis или in-memory dict.
 
## Направления улучшения
 
1. **LLM-based intent detection**: заменить regex на LLM call (Qwen3-4B) в QueryAnalyzer.
2. **LLM-based eval set generation**: заменить template-based question generation на LLM.
3. **Qwen3-Reranker adapter**: custom `Reranker` subclass с `AutoModelForCausalLM` + yes/no logit extraction. Ablation: bge-reranker vs Qwen3-Reranker.
4. **Extended intent patterns**: добавить regex для "what ... achieve", "what guarantees", "what bounds" → `factual_qa`.
5. **Multi-query retrieval**: QueryAnalyzer генерирует expanded queries → parallel retrieval → fusion.
6. **Hybrid search через Qdrant prefetch**: вместо отдельных dense/sparse channels → один `search_hybrid()` call с server-side RRF.
7. **Reranker fine-tuning на domain data**: fine-tune bge-reranker или train adapter для научного домена. Training data: (query, positive_chunk, negative_chunks) из evaluation runs.
8. **LoRA adapters для mode-specific retrieval**: отдельные LoRA adapters для embedder, оптимизированные под разные режимы.
9. **End-to-end retrieval evaluation**: оценка retrieval не по IR metrics (Recall, NDCG), а по downstream task quality (ответ агента оценивается LLM-as-a-judge).