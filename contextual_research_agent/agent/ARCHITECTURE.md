# Multi-Agent Research Assistant — Architecture

## File Structure

```
agent/
├── __init__.py                # Public API: create_agent, MultiAgentRAG
├── config.py                  # AgentConfig (KEEP existing, extend)
├── llm.py                     # LLMProvider (KEEP existing, no changes)
├── state.py                   # AgentState — LangGraph shared state
├── graph.py                   # LangGraph graph definition + compilation
├── context.py                 # ApplicationContext — singleton, holds pipelines
│
├── nodes/                     # LangGraph node functions (one per agent role)
│   ├── __init__.py
│   ├── router.py              # RouterAgent: intent + complexity analysis
│   ├── planner.py             # PlannerAgent: query decomposition for complex queries
│   ├── retriever.py           # RetrieverAgent: wraps RetrievalPipeline
│   ├── generator.py           # GeneratorAgent: wraps GenerationPipeline
│   ├── critic.py              # CriticAgent: verification + hallucination check
│   └── synthesizer.py         # SynthesizerAgent: merges multi-step results
│
├── tracing/                   # Debug/observability layer
│   ├── __init__.py
│   ├── trace.py               # AgentTrace — per-query execution trace
│   └── mlflow_logger.py       # MLflow integration for agent runs
│
└── service.py                 # ResearchAssistantService — thin API for UI/CLI
```

## Agent Roles

1. **Router** — Analyzes query intent + complexity → decides routing
2. **Planner** — Decomposes complex/multi-aspect queries into sub-queries
3. **Retriever** — Executes retrieval pipeline, returns candidates
4. **Generator** — Executes generation pipeline with mode-specific prompts
5. **Critic** — Verifies answer: hallucination check, completeness, citation validity
6. **Synthesizer** — Merges results from multiple sub-queries or retry attempts

## Flow Types

- **Simple**: Router → Retriever → Generator → Critic → [pass] → END
- **Complex**: Router → Planner → [Retriever → Generator]×N → Synthesizer → Critic → END
- **Retry**: Critic [fail] → Generator (with feedback) → Critic → END (max 2 retries)
