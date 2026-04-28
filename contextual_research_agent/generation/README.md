# Generation Pipeline

## Назначение

Модуль generation реализует генеративную часть RAG-пайплайна: принимает результат retrieval (набор релевантных чанков с провенансом), формирует mode-specific промпт, вызывает LLM и возвращает ответ с полной трассировкой. Модуль также включает evaluation framework для оценки качества генерации.

Модуль спроектирован как **отдельный pipeline-слой**, отделённый от retrieval:

- принимает `RetrievalResult` как входные данные, не зависит от деталей retrieval каналов;
- поддерживает 6 когнитивных режимов с mode-specific промптами;
- предоставляет `generate_from_context()` для robustness tests (random/empty context);
- evaluation framework поддерживает semantic similarity и LLM-as-judge.

### Архитектурное решение: когнитивные режимы

Каждый когнитивный режим соответствует типу исследовательской задачи и реализован через отдельный prompt template. Маршрутизация между режимами осуществляется автоматически по intent из QueryAnalyzer, с возможностью ручного override. Режимы определяют только промпт-стратегию; LLM и retrieval pipeline одинаковы для всех режимов.

## Архитектура

```
                    RetrievalResult
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 1: Mode Resolution                                │
│  intent → CognitiveMode.from_intent()                    │
│  + _refine_comparison_intent() heuristic                 │
│  → CognitiveMode (factual_qa / summarization / ...)      │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 2: Context Preparation                            │
│  candidates[:max_context_chunks] → context string        │
│  (truncation by config, format with chunk_id + metadata) │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 3: Prompt Assembly                                │
│  get_prompt_template(mode) → system + user templates     │
│  user_template.format(context=..., query=...)            │
│  _adjust_max_tokens(mode) → adaptive token budget        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Stage 4: LLM Inference                                  │
│  LLMProvider.generate(prompt, system_prompt, temp, max)  │
│  Backends: OllamaProvider / LlamaCppProvider             │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
                    RAGResponse
                  (answer + provenance)
                         │
                    ┌────┴────┐
                    ▼         ▼
              CLI output   Evaluation
                          (sim + judge)
```

## Структура модуля

```
generation/
├── config.py           # CognitiveMode, GenerationConfig, LLMConfig
├── prompts.py          # PromptTemplate, TEMPLATES dict, get_prompt_template()
├── pipeline.py         # GenerationPipeline, RAGResponse
├── evaluation.py       # GenerationEvaluator, GenerationMetrics, AggregatedGenerationMetrics
├── cli.py              # CLI commands: generate, evaluate-generation
└── __init__.py         # Public API exports
```

## Компоненты

### CognitiveMode

Enum из 6 режимов, каждый соответствует типу исследовательской задачи:

| Режим | Intent mapping | Описание |
|---|---|---|
| `factual_qa` | factual_qa | Точный ответ на конкретный вопрос, цитирование source |
| `summarization` | method_explanation, survey | Лаконичное резюме в 3–5 предложений |
| `critical_review` | critique | Анализ strengths/weaknesses в 4–6 предложений |
| `comparison` | comparison | Сравнение методов по ключевым осям в 4–8 предложений |
| `methodological_audit` | — (manual only) | Аудит datasets, metrics, baselines, reproducibility |
| `idea_generation` | — (manual only) | Генерация направлений исследований на основе gaps |

Маршрутизация: `CognitiveMode.from_intent(retrieval_intent)` с дополнительной эвристикой `_refine_comparison_intent()` — factual queries с "compared to" корректно направляются в `factual_qa`, а не в `comparison`.

### PromptTemplate

Каждый режим определяет пару (system_prompt, user_template). Общие правила для всех режимов:

1. Отвечать **только** на основе предоставленного контекста
2. При недостатке информации — явный отказ (не галлюцинировать)
3. Цитировать чанки через `[chunk_id]` нотацию
4. Использовать точную ML/NLP терминологию

Verbose-режимы (summarization, critical_review, comparison) содержат инструкцию лаконичности ("3–5 sentences", "no bullet points or numbered lists") для контроля длины ответа. Без этого ограничения ответы в 8–10 раз длиннее эталонных, что искажает semantic similarity метрику.

### GenerationPipeline

Orchestrator: `RetrievalResult` → prompt → LLM → `RAGResponse`.

Ключевые механизмы:

**Context truncation**: если `len(candidates) > max_context_chunks`, обрезает до лимита и переформатирует контекст. Провенанс (`chunk_ids_used`, `document_ids_used`) отражает реально использованные чанки.

**Adaptive max_tokens**: `_adjust_max_tokens(mode, base_max)` гарантирует минимальный token budget для structured-режимов. factual_qa = 256 min, comparison/critical_review = 1024 min.

**Intent refinement**: `_refine_comparison_intent()` — regex-эвристика, которая корректирует ложные comparison intents. Запросы вида "How much does X reduce compared to Y" → factual_qa.

Метод `generate_from_context(query, context)` — для robustness tests: позволяет подать произвольный контекст (random chunks, empty, partial) без retrieval pipeline.

### GenerationEvaluator

Оценка качества генерации по трём осям:

**1. Semantic similarity** (cosine, 0–1): embedding сгенерированного ответа vs embedding эталонного ответа. Использует тот же embedder что и retrieval pipeline. Ограничение: penalizes length mismatch (длинный ответ vs короткий эталон).

**2. Faithfulness** (LLM-as-judge, 1–5): "Содержит ли ответ утверждения, не подтверждённые контекстом?" Отдельный LLM (judge) оценивает каждый ответ. Формат: SCORE + REASONING.

**3. Relevance** (LLM-as-judge, 1–5): "Отвечает ли ответ на заданный вопрос?" Независимо от faithfulness.

**Refusal detection**: regex-based, 6 паттернов ("The provided sources do not contain...", "insufficient information", etc.). Пустые ответы (empty string от LLM) также классифицируются как refusal.

**Aggregation**: per-category breakdown (по 6 категориям eval set), pass rates (% с score ≥ 4), refusal accuracy.

### LLMProvider

Абстракция над LLM backend. Два провайдера:

| Провайдер | API endpoint | Модели |
|---|---|---|
| `OllamaProvider` | `/api/chat` | qwen3:8b, qwen3:4b, и др. |
| `LlamaCppProvider` | `/v1/chat/completions` | Любая GGUF модель через llama-server |

Factory: `create_llm_provider(provider, model, host)`.

Важно: для Qwen3 моделей через Ollama необходим `"think": False` в payload — иначе модель тратит token budget на reasoning chain, а content поле пустое.

## Параметры, влияющие на качество

### Параметры генерации

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `temperature` | `--temperature` | 0.1 | Низкая → детерминированные ответы. Для научного ассистента 0.1 оптимально. |
| `max_tokens` | `--max-tokens` | 1024 | Token budget для ответа. Adaptive per mode: factual_qa ≥ 256, comparison ≥ 1024. |
| `max_context_chunks` | — | 10 | Максимум чанков в промпте. **Experiment-validated**: 10 чанков → refusal rate 5.1% (vs 14.5% при 5 чанках). |
| `require_citation` | — | True | Инструкция цитировать `[chunk_id]`. Улучшает traceability, минимально влияет на quality. |
| `auto_detect_mode` | — | True | Автоматическое определение когнитивного режима из retrieval intent. |

### Параметры LLM

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `llm_provider` | `--llm-provider` | ollama | Backend: ollama / llama_cpp. |
| `llm_model` | `--llm-model` | qwen3:8b | Модель. 8B — baseline quality, 4B — faster, lower quality. |
| `llm_host` | `--llm-host` | http://localhost:11434 | URL сервера. |

### Параметры evaluation

| Параметр | CLI флаг | Default | Влияние |
|---|---|---|---|
| `skip_judge` | `--skip-judge` | False | Пропустить LLM-as-judge (только semantic similarity). Экономит ~50% времени. |
| `judge_model` | `--judge-model` | (same as gen) | Отдельная модель для judge. Рекомендуется ≥ gen model по качеству. |
| `max_queries` | `--max-queries` | None | Ограничение для quick tests. |

## CLI-интерфейс

### Одиночный запрос

```bash
python main.py generate "How does LoRA work?" \
    --collection=peft_hybrid \
    --llm-model=qwen3:8b \
    --channels=dense,sparse,graph_entity,paper_level \
    --rerank=True

# С explicit mode
python main.py generate "Compare LoRA and QLoRA" \
    --mode=comparison \
    --verbose  # показывает system + user prompts
```

### Evaluation на eval set

```bash
# Quick test (20 queries, без judge)
python main.py evaluate-generation eval/peft_gold_v3_mapped.json \
    --skip-judge \
    --max-queries=20 \
    --output=eval/results/gen_quick_test.json

# Полный прогон (276 queries)
python main.py evaluate-generation eval/peft_gold_v3_mapped.json \
    --collection=peft_hybrid \
    --llm-model=qwen3:8b \
    --channels=dense,sparse,graph_entity,paper_level \
    --rerank=True \
    --rerank-model="cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --max-tokens=1024 \
    --skip-judge \
    --experiment-name=generation \
    --run-name=qwen3_8b_baseline \
    --output=eval/results/gen_full.json

# С LLM-as-judge
python main.py evaluate-generation eval/peft_gold_v3_mapped.json \
    --llm-model=qwen3:8b \
    --judge-model=qwen3:8b \
    --output=eval/results/gen_with_judge.json
```

### Checkpointing

Evaluation автоматически сохраняет checkpoint каждые 25 queries в `{output}.checkpoint.json`. При crash — перезапуск той же команды автоматически продолжает с последнего checkpoint. После успешного завершения checkpoint удаляется.

## Experimental Results

### Best configuration (276 queries, after prompt fixes)

| Metric | Value |
|---|---|
| Mean semantic similarity | 0.787 |
| Median semantic similarity | 0.795 |
| Refusal rate | 5.1% |
| MRR (retrieval) | 0.654 |
| Chunks in context | 10 |

### Per-category breakdown

| Категория | Count | Mean sim | Refusal rate |
|---|---|---|---|
| factual_qa | 47 | 0.815 | 6.4% |
| comparison | 79 | 0.818 | 7.6% |
| method_explanation | 47 | 0.799 | 0.0% |
| critique | 29 | 0.742 | 0.0% |
| survey | 49 | 0.742 | 2.0% |
| citation_trace | 25 | 0.757 | 16.0% |

### Similarity distribution

| Диапазон | Кол-во | % |
|---|---|---|
| > 0.9 | 14 | 5.3% |
| 0.8 – 0.9 | 110 | 42.0% |
| 0.7 – 0.8 | 102 | 38.9% |
| 0.6 – 0.7 | 30 | 11.5% |
| 0.5 – 0.6 | 6 | 2.3% |
| < 0.5 | 0 | 0.0% |

### Impact of prompt conciseness fix

| Metric | До фикса | После фикса | Δ |
|---|---|---|---|
| Semantic similarity | 0.772 | 0.787 | +1.9% |
| Refusal rate | 14.5% | 5.1% | −65% relative |
| Answer length ratio (gen/exp) | 8x | 2.8x | −63% |
| Chunks in context | 5 | 10 | +100% |

Основные улучшения: conciseness instructions в verbose-режимах (summarization, critical_review, comparison) + увеличение context/max_tokens с 4096 до 6144 токенов.

### LLM-as-Judge Results (external judges)
 
Оценка качества ответов с помощью внешних моделей (OpenRouter API), устраняющая self-evaluation bias:
 
| Judge Model | Mean Faithfulness | Faithfulness Pass Rate (≥4) | Mean Relevance | Relevance Pass Rate (≥4) |
|---|---|---|---|---|
| GPT-5.4 mini | 3.74 | 68.7% | 3.89 | 73.2% |
| Claude Sonnet 4 | 3.99 | 76.0% | 4.12 | 79.5% |
 
Judge-модели оценивают faithfulness (соответствие ответа контексту) и relevance (соответствие ответа вопросу) по шкале 1–5. Refusal-ответы исключаются из оценки.
 
### Robustness Evaluation
 
Оценка устойчивости генерации при деградированном контексте. Тестирует способность модели отказывать (refusal) при отсутствии или несоответствии контекста, предотвращая галлюцинации.
 
**Три сценария деградации контекста:**
 
| Сценарий | Описание | Ожидание |
|---|---|---|
| Empty context | Пустая строка вместо контекста | ~100% refusal |
| Random context | 10 случайных чанков, не связанных с запросом | Высокий refusal, низкий faithfulness |
| Partial context | Чанки из правильной статьи, но из нерелевантных секций | Умеренный refusal |
 
**Агрегированные результаты (276 queries per scenario, seed=42):**
 
| Scenario | Refusal Rate | Non-refusal | Mean Latency |
|---|---|---|---|
| Normal RAG | 5.1% | 262 | — |
| Empty context | **100.0%** | 0 | 0 ms (guard) |
| Random context | **75.4%** | 68 | 3470 ms |
| Partial context | **57.6%** | 117 | 4237 ms |
 
**Per-category refusal rates:**
 
| Категория | Normal | Empty | Random | Partial |
|---|---|---|---|---|
| factual_qa | 6.4% | 100% | 89.4% | 72.3% |
| citation_trace | 16.0% | 100% | 80.0% | 88.0% |
| comparison | 7.6% | 100% | 91.1% | 82.3% |
| critique | 0.0% | 100% | 58.6% | 31.0% |
| method_explanation | 0.0% | 100% | 61.7% | 25.5% |
| survey | 2.0% | 100% | 57.1% | 34.7% |
 
**Ключевые выводы:**
 
1. **Монотонная деградация**: refusal rate демонстрирует ожидаемый порядок empty (100%) > random (75.4%) > partial (57.6%) > normal (5.1%). Модель калиброванно реагирует на качество контекста.
2. **Mode-dependent robustness**: factual-ориентированные режимы (factual_qa, citation_trace, comparison) устойчивы к деградации контекста (80–91% refusal при random). Рассуждательные режимы (critique, method_explanation, survey) менее устойчивы (57–62% при random), так как модель способна генерировать правдоподобные ответы из parametric knowledge.
3. **Корпусный эффект в random scenario**: refusal rate (75.4%) — нижняя граница ожидаемого диапазона. Это объясняется тематической однородностью корпуса: все 32 статьи посвящены PEFT-методам, поэтому "случайные" чанки всё равно тематически близки запросам.
4. **Empty-context guard**: программная проверка пустого контекста до вызова LLM гарантирует 100% refusal с нулевой latency, без расходования LLM-ресурсов.
**Prompt-level hardening:**
 
Для повышения robustness рассуждательных режимов добавлена инструкция в промпты `summarization`, `critical_review`, `comparison`:
 
```
If the context passages section above is empty or contains no text,
you MUST refuse to answer regardless of whether you know the answer from training.
```
 
Эта инструкция не влияет на нормальный RAG (контекст всегда непустой), но повышает refusal rate при деградированном контексте.
 
**Реализация:**
 
Модуль `generation/robustness_eval.py` реализует:
 
- Три сценария контекстной деградации (empty, random, partial)
- Сэмплирование случайных чанков из Qdrant с исключением relevant_ids
- Partial context: чанки из правильной статьи с исключением релевантных секций
- Checkpointing каждые 25 queries
- Per-category metrics aggregation
- MLflow logging (experiment: `robustness`)
- Опциональный LLM-as-judge для non-refusal ответов
### Robustness CLI
 
```bash
# Полный прогон — все сценарии без judge
python main.py robustness-eval eval/peft_gold_v3_mapped.json \
    --scenario=all --skip-judge
 
# С judge на random scenario
python main.py robustness-eval eval/peft_gold_v3_mapped.json \
    --scenario=random \
    --judge-model=openai/gpt-5.4-mini
```

## Error Analysis (сводка)

Детальный анализ ошибок — см. ERROR_ANALYSIS_RU.md.

### Классификация 14 оставшихся refusals (5.1%)

| Тип ошибки | Кол-во | Причина |
|---|---|---|
| Cross-document synthesis | 6 | Query требует info из 2+ papers |
| Citation trace multi-doc | 4 | Meta-reasoning о статьях |
| Specific fact not in top-10 | 3 | Нужное число в чанке за пределами контекста |
| Survey synthesis | 1 | Обобщение нескольких обзоров |

Все оставшиеся refusals — корректные отказы при недостатке информации в контексте. Для их устранения необходим agentic RAG: query expansion, document diversity в context assembly, paper-level reasoning.

## Известные ограничения

### Semantic similarity как метрика

Cosine similarity между embeddings ответов penalizes length mismatch. Для verbose-режимов (critique, survey) эталонный ответ в 2–3 раза короче сгенерированного, что занижает sim даже при корректном содержании. Рекомендуется дополнить LLM-as-judge оценкой.

### Single-model generation

Текущий baseline — Qwen3:8b через Ollama. Для пункта 9 требований ВКР необходимо сравнение с ≥1 дополнительной моделью (Qwen3:4b или внешний API).

### Refusal calibration

Промпт инструктирует модель отказывать при недостатке информации. Robustness tests показали mode-dependent behavior: factual-режимы (factual_qa, citation_trace) устойчивы (80–91% refusal при random context), но рассуждательные режимы (critique, survey, method_explanation) имеют refusal rate 57–62% при random context — модель способна генерировать правдоподобные но потенциально недостоверные ответы из parametric knowledge. Программный guard для empty context и prompt-level hardening частично решают проблему, но полное решение требует output verification layer или constrained decoding.

### Judge self-evaluation bias

При использовании той же модели (Qwen3:8b) как generator и judge, возникает self-evaluation bias — модель склонна оценивать свои ответы выше. Для корректной оценки рекомендуется внешний judge (Claude API, GPT-4).

## Воспроизводимость

Полная воспроизводимость обеспечивается фиксацией:

1. **Retrieval config** (MLflow params): embedding_model, channels, reranker, fusion weights → определяет контекст для генерации.
2. **Generation config** (MLflow params): temperature, max_tokens, cognitive_mode, max_context_chunks → определяет поведение LLM.
3. **LLM config**: provider, model, host → определяет модель генерации.
4. **Eval set**: путь к JSON с queries + expected_answers + relevant_ids.
5. **Checkpoints**: промежуточные результаты для resume при crash.

Все параметры логируются в MLflow. Output JSON содержит per-query результаты с полной провенансом (chunk_ids_used, mode, latencies, prompts).
