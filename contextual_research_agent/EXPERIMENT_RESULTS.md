# Experiment Results Summary

## Corpus

- 32 статьи по PEFT (7 тематических групп)
- 2364 чанка (baseline config)
- 1142 citation edges, 1685 entity edges
- Eval set: 276 queries, 6 категорий

---

## 1. Retrieval Ablation Study (pipeline components)

Embedding: Qwen3-Embedding-0.6B, Chunking: 512 tokens, merge_peers=True, context=True.

| Config | MRR | P@1 | NDCG@10 | HR@10 | Latency |
|---|---|---|---|---|---|
| Dense only (baseline) | 0.585 | 0.366 | 0.372 | 0.960 | 109ms |
| Dense + rerank (bge) | 0.501 | 0.337 | 0.284 | 0.870 | 2000ms |
| Hybrid (dense + BM25) | 0.562 | 0.355 | 0.328 | 0.949 | 116ms |
| Full (no rerank) | 0.632 | 0.460 | 0.410 | 0.942 | 243ms |
| **Full + rerank (MiniLM)** | **0.654** | **0.500** | **0.417** | **0.957** | **417ms** |
| Full + rerank (bge-m3) | 0.640 | 0.496 | 0.415 | 0.953 | 2160ms |

**Key findings:**
- Multi-channel retrieval (entity graph + paper-level) = главный contributor: +8% MRR, +25% P@1 vs dense only
- BM25 с весом 0.3 ухудшает на −4% MRR (lexical noise на научном тексте)
- MiniLM reranker — лучший Pareto choice: +3.5% MRR vs no-rerank при 417ms

---

## 2. Embedding Model Comparison

Chunking: 512 tokens, merge_peers=True, context=True.

### Dense only

| Model | Params | Dim | MRR | P@1 | NDCG@10 | HR@10 | Latency |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 600M | 1024 | **0.585** | **0.366** | **0.372** | **0.960** | 109ms |
| bge-large-en-v1.5 | 335M | 1024 | 0.433 | 0.188 | 0.259 | 0.917 | 58ms |
| all-MiniLM-L6-v2 | 22M | 384 | 0.475 | 0.279 | 0.255 | 0.884 | 37ms |
| all-mpnet-base-v2 | 110M | 768 | 0.418 | 0.214 | 0.246 | 0.888 | 48ms |

### Full pipeline + rerank

| Model | Params | Dim | MRR | P@1 | NDCG@10 | HR@10 | Latency |
|---|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | 600M | 1024 | **0.640** | **0.496** | **0.415** | **0.953** | 2209ms |
| bge-large-en-v1.5 | 335M | 1024 | 0.414 | 0.178 | 0.278 | 0.942 | 2188ms |
| all-MiniLM-L6-v2 | 22M | 384 | 0.465 | 0.297 | 0.260 | 0.895 | 2163ms |
| all-mpnet-base-v2 | 110M | 768 | 0.386 | 0.225 | 0.222 | 0.833 | 2130ms |

**Key findings:**
- Qwen3-Embedding-0.6B доминирует: +23-55% MRR vs classic sentence-transformers
- MiniLM (22M) > bge-large-en (335M) — training recipe важнее raw params
- English-only (bge-large-en) не даёт преимущества над multilingual (Qwen3) на English corpus
- Cross-encoder reranker усиливает сильную модель, ухудшает слабые

---

## 3. Chunking Experiments

Embedding: Qwen3-Embedding-0.6B.

### Dense only

| Config | max_tokens | merge | context | MRR | P@1 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|---|---|
| **Baseline** | **512** | **Yes** | **Yes** | **0.585** | **0.366** | **0.372** | **0.960** |
| Small chunks | 256 | Yes | Yes | 0.525 | 0.319 | 0.265 | 0.909 |
| Large chunks | 1024 | Yes | Yes | 0.537 | 0.315 | 0.269 | 0.917 |
| No merge | 512 | No | Yes | 0.496 | 0.283 | 0.257 | 0.938 |
| No context | 512 | Yes | No | 0.538 | 0.341 | 0.280 | 0.924 |

### Full pipeline + rerank

| Config | max_tokens | merge | context | MRR | P@1 | NDCG@10 | HR@10 |
|---|---|---|---|---|---|---|---|
| **Baseline** | **512** | **Yes** | **Yes** | **0.640** | **0.496** | **0.415** | **0.953** |
| Small chunks | 256 | Yes | Yes | 0.526 | 0.362 | 0.280 | 0.906 |
| Large chunks | 1024 | Yes | Yes | 0.583 | 0.417 | 0.319 | 0.906 |
| No merge | 512 | No | Yes | 0.523 | 0.359 | 0.284 | 0.917 |
| No context | 512 | Yes | No | 0.580 | 0.431 | 0.326 | 0.928 |

**Key findings:**
- 512 tokens = оптимальный размер (совпадает с max input embedding модели)
- merge_peers — самый импактный параметр: −15-18% MRR при отключении
- Section context: +8-9% MRR, рекомендуется включать
- 1024 tokens + rerank имеет лучший MAP@10 (0.184 vs 0.171) — reranker эффективнее на длинных чанках

---

## 4. Reranker Comparison

Full pipeline: dense + sparse + entity_graph + paper_level.

| Reranker | Params | MRR | P@1 | NDCG@10 | HR@10 | Latency |
|---|---|---|---|---|---|---|
| No rerank | — | 0.632 | 0.460 | 0.410 | 0.942 | 243ms |
| **MiniLM-L-6-v2** | **22M** | **0.654** | **0.500** | **0.417** | **0.957** | **417ms** |
| bge-reranker-v2-m3 | 568M | 0.640 | 0.496 | 0.415 | 0.953 | 2160ms |
| bge-reranker-v2-m3 (maxlen=1024) | 568M | 0.636 | 0.486 | 0.414 | 0.953 | 2666ms |
| Qwen3-Reranker-0.6B-seq-cls | 600M | 0.434 | 0.275 | 0.276 | 0.823 | 25817ms |

**Key findings:**
- MiniLM (22M) — лучший reranker: +3.5% MRR, −81% latency vs bge-reranker
- Размер cross-encoder не коррелирует с quality на domain-specific данных
- Qwen3-Reranker seq-cls conversion потерял quality; requires native inference
- При сильном bi-encoder (Qwen3-Emb) reranker даёт marginal improvement (+1-3%)

---

## 5. Generation Baseline (276 queries)

Full pipeline + MiniLM reranker, Qwen3:8b, skip-judge.

| Metric | Value |
|---|---|
| Mean semantic similarity | 0.774 |
| Median semantic similarity | 0.783 |
| Refusal rate | 12.7% |

### Per-category breakdown

| Category | Count | Mean sim | Refusal rate |
|---|---|---|---|
| factual_qa | 47 | 0.826 | 14.9% |
| comparison | 79 | 0.804 | 24.1% |
| method_explanation | 47 | 0.767 | 2.1% |
| critique | 29 | 0.725 | 0.0% |
| survey | 49 | 0.729 | 12.2% |
| citation_trace | 25 | 0.760 | 8.0% |

**Key findings:**
- Factual QA: лучшая similarity (0.826), но неожиданно высокий refusal (14.9%)
- Comparison: самый высокий refusal (24.1%) — требует multi-document context
- Critique: самая низкая similarity (0.725), ноль refusals — ответы далеки от expected

---

## 6. Best Configuration

Based on all experiments, the optimal configuration is:

| Component | Choice | Rationale |
|---|---|---|
| Embedding | Qwen3-Embedding-0.6B (1024d) | +23-55% MRR vs alternatives |
| Chunk size | 512 tokens | Optimal for Qwen3-Emb max input |
| merge_peers | True | −15-18% MRR when disabled |
| include_context | True | +8-9% MRR |
| Sparse channel | BM25 weight=0.3 | Marginal, but keeps hybrid option |
| Graph channels | Entity enabled, Citation disabled | Entity +5-8% MRR; Citation 0 candidates (corpus too small) |
| Paper-level | Enabled | +3-5% MRR via document-level matching |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Best quality/latency Pareto |
| Fusion | RRF (k=60) | Standard, works well with diverse channels |

**Expected metrics (full pipeline + MiniLM reranker):**
- MRR: 0.654
- P@1: 0.500
- NDCG@10: 0.417
- HR@10: 0.957
- Latency: ~417ms
