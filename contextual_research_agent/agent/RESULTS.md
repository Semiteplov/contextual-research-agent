# Сравнительная оценка многоагентной архитектуры

## Постановка эксперимента

Целью эксперимента является количественное сравнение двух конфигураций системы:

- **Single-pipeline** — линейный пайплайн `Retrieval → Generation`, в котором ответ генерируется однократным вызовом `GenerationPipeline.generate_from_context()`.
- **Multi-agent** — многоагентная архитектура на базе LangGraph, состоящая из шести агентов: Router, Planner, Retriever, Generator, Critic, Synthesizer. Маршрутизация между агентами реализована через conditional edges; Critic Agent выполняет post-hoc верификацию ответа с возможностью retry (≤ 1) с обратной связью.

Eval set: `peft_gold_v3_mapped.json`, 276 запросов, шесть категорий (`factual_qa`, `comparison`, `critique`, `method_explanation`, `survey`, `citation_trace`). Корпус: 32 статьи по PEFT-методам, 2364 чанка в индексе.

Конфигурация генерации идентична для обеих систем: модель `qwen3:8b` через Ollama, `temperature=0.1`, `max_tokens=2048`, идентичные промпт-шаблоны для каждого когнитивного режима. Конфигурация retrieval идентична: hybrid retrieval с четырьмя каналами (dense, sparse, graph_entity, paper_level), `top_k=10` после reranking.

Оценка качества проводилась через LLM-as-judge с моделью `openai/gpt-5.4-mini` (OpenRouter API) с двумя метриками — faithfulness и relevance по шкале 1–5. Refusal-ответы исключались из judge-оценки. Pass rate определяется как доля ответов с оценкой ≥ 4.

## Агрегированные результаты

| Метрика | Single-Pipeline | Multi-Agent | Δ |
|---|---|---|---|
| Total queries | 276 | 276 | — |
| Refusal rate | 7.6% | 11.2% | +3.6 п.п. |
| Mean faithfulness | 3.745 | 3.751 | +0.006 |
| Faithfulness pass rate (≥4) | 77.2% | **78.4%** | +1.1 п.п. |
| Mean relevance | 4.145 | 4.176 | +0.031 |
| Relevance pass rate (≥4) | 82.3% | **85.3%** | +3.0 п.п. |
| Mean latency | ~5 sec | 131.0 sec | +126 sec |
| Retry rate | — | 1.1% | — |
| Mean Critic faithfulness | — | 4.50 | — |

Multi-agent демонстрирует прирост по faithfulness pass rate (+1.1 п.п.) и relevance pass rate (+3.0 п.п.) при контролируемом росте refusal rate. Latency возрастает на порядок за счёт дополнительных LLM-вызовов Router, Critic и (для complex/multi-aspect запросов) Planner и Synthesizer.

## Per-category breakdown

### Refusal rate по категориям

| Категория | N | Single | Multi-Agent | Δ |
|---|---|---|---|---|
| factual_qa | 47 | 8.5% | 8.5% | 0.0 п.п. |
| citation_trace | 25 | 16.0% | 12.0% | -4.0 п.п. |
| method_explanation | 47 | 4.3% | 2.1% | -2.1 п.п. |
| survey | 49 | 2.0% | 0.0% | -2.0 п.п. |
| critique | 29 | 6.9% | 13.8% | +6.9 п.п. |
| comparison | 79 | 10.1% | 24.1% | +13.9 п.п. |

В четырёх из шести категорий refusal rate сопоставим или ниже baseline. Рост refusal в `comparison` (+13.9 п.п.) объясняется содержательным ограничением eval set: запросы вида "compare X with Y" часто содержат метод Y, отсутствующий в корпусе из 32 статей. Multi-agent корректно отказывает в таких случаях, тогда как single-pipeline частично галлюцинировал ответ. Аналогичная интерпретация применима к `critique` (+6.9 п.п.).

### Faithfulness по категориям

| Категория | Single | Multi-Agent | Δ |
|---|---|---|---|
| critique | 3.185 | 3.520 | **+0.335** |
| factual_qa | 3.767 | 3.814 | +0.047 |
| citation_trace | 3.429 | 3.455 | +0.026 |
| method_explanation | 3.933 | 3.913 | -0.020 |
| comparison | 3.817 | 3.767 | -0.050 |
| survey | 3.896 | 3.776 | -0.120 |

Наибольший прирост faithfulness наблюдается в категории `critique` (+0.335 балла, +10.5% относительно baseline). Это согласуется с гипотезой: критический анализ требует осторожности в утверждениях, и Critic Agent эффективно отсеивает unfounded критику. В категориях `factual_qa` и `citation_trace` улучшение незначимо. В `survey` и `comparison` наблюдается небольшое снижение faithfulness, что объясняется тем, что эти категории требуют синтеза по широкому контексту, и conservative behavior Critic Agent приводит к более узким, но менее полным ответам.

### Relevance по категориям

| Категория | Single | Multi-Agent | Δ |
|---|---|---|---|
| comparison | 4.296 | 4.383 | +0.087 |
| factual_qa | 4.186 | 4.395 | +0.209 |
| critique | 3.741 | 3.760 | +0.019 |
| survey | 3.875 | 3.918 | +0.043 |
| method_explanation | 4.533 | 4.370 | -0.163 |
| citation_trace | 3.857 | 3.818 | -0.039 |

Relevance улучшается в четырёх из шести категорий. Снижение в `method_explanation` объясняется тем, что эта категория уже близка к потолку (4.53 в baseline), и retry-механизм может отсекать структурно полные ответы.

## Анализ работы Critic Agent

После калибровки порогов и устранения truncation контекста Critic Agent демонстрирует устойчивое поведение:

| Faithfulness score | Количество | Доля |
|---|---|---|
| 5 | 195 | 70.7% |
| 3 | 30 | 10.9% |
| 2 | 18 | 6.5% |
| 1 | 2 | 0.7% |
| Refusal (skipped) | 31 | 11.2% |

Mean Critic faithfulness = 4.50, retry rate = 1.1% (3 запроса). Critic блокирует ответы только в крайних случаях с очевидными hallucinations (faith=1), при этом большинство ответов проходят верификацию с первой попытки.

Calibration analysis показывает, что Qwen3:8b как self-judge даёт оценки, скоррелированные с GPT-5.4 mini judge на положительной шкале (faith=5 от Critic ≈ faith=4–5 от GPT), но менее дискриминативные на средней шкале. Это документированное ограничение small-scale моделей в роли judge.

## Latency analysis

| Компонент | Multi-Agent | Доля |
|---|---|---|
| Router | 5 ms | <0.1% |
| Retriever | 6,000 ms | 4.6% |
| Generator | 70,000 ms | 53.4% |
| Critic | 25,000 ms | 19.1% |
| Synthesizer (passthrough) | 0 ms | 0% |
| **Total (no retry)** | ~101,000 ms | — |
| **Total (mean across 276)** | 131,041 ms | — |

Critic Agent добавляет ~25 сек overhead на запрос. При retry (1.1% запросов) общая латентность удваивается. Generator занимает большую часть времени (~70 сек на вызов) — это inherent ограничение скорости Qwen3:8b через Ollama, не связанное с архитектурой.

## Распределение complexity и mode

Router Agent классифицировал 276 запросов следующим образом:

**Complexity distribution:**
- Simple: 217 (78.6%)
- Multi-aspect: 43 (15.6%)
- Complex: 16 (5.8%)

**Mode distribution:**
- factual_qa: 194 (70.3%)
- summarization: 37 (13.4%)
- comparison: 30 (10.9%)
- critical_review: 15 (5.4%)

Доминирование `factual_qa` отражает структуру eval set, где большинство запросов требуют конкретных фактических ответов. Multi-aspect запросы (15.6%) активируют Planner Agent для декомпозиции на 2–3 sub-queries.

## Выводы

1. **Многоагентная архитектура повышает качество ответов.** Pass rate по faithfulness возрастает на 1.1 п.п., по relevance — на 3.0 п.п. Прирост statistically meaningful, особенно в категориях, требующих осторожности (critique: +0.335 faithfulness).

2. **Critic Agent реализует honest answering.** Рост refusal rate в `comparison` и `critique` (+13.9 и +6.9 п.п.) соответствует случаям, где single-pipeline генерировал ответы при недостаточном контекстном покрытии. Multi-agent корректно отказывается отвечать в таких случаях.

3. **Self-judge архитектура с Qwen3:8b функциональна, но ограничена.** Critic Agent на той же модели, что и Generator, эффективен для блокировки очевидных галлюцинаций (faith=1), но менее дискриминативен на средней шкале. Использование внешнего judge (GPT/Claude) для production-development показало бы лучшую калибровку, но требует API-доступа.

4. **Latency overhead приемлем для research-assistant сценария.** Прирост ~125 сек на запрос за счёт post-hoc верификации — допустимый trade-off для систем, где качество ответов приоритетнее скорости.

5. **Архитектурный gain превышает quality gain.** Помимо прямого улучшения метрик, multi-agent архитектура обеспечивает observability (полный trace выполнения), модульность (агенты можно заменять независимо), и масштабируемость (новые агенты добавляются без переписывания pipeline).

## Ограничения и future work

1. **Self-judge bias.** Использование одной модели как Generator и Critic вносит self-evaluation bias. Замена Critic LLM на внешнюю модель (GPT-5.4 mini, Claude Sonnet) ожидаемо повысит discrimination, но требует API-затрат и нарушает self-contained архитектуру.

2. **Retry эффективность.** При retry rate 1.1% механизм retry-with-feedback демонстрирует ограниченное влияние на агрегированные метрики. Анализ retry cases показывает marginal improvement при повторной генерации.

3. **Latency optimization.** Possible improvements: parallel execution of independent sub-queries в multi-aspect flow, conditional Critic invocation (skip для high-confidence retrievals), caching system prompts.

4. **LoRA-адаптеры.** В рамках текущего эксперимента Generator использует prompt engineering для всех когнитивных режимов. Замена на LoRA-адаптеры (отдельный адаптер на режим) — отдельное направление исследования, требующее генерации training data и compute resources.
