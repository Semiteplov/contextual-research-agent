import fire

from contextual_research_agent.cli.agent import agent_chat, agent_eval, agent_query
from contextual_research_agent.cli.arxiv import download_arxiv_papers, ingest_arxiv_metadata
from contextual_research_agent.cli.compare import compare_agent_vs_pipeline
from contextual_research_agent.cli.datasets import (
    create_dataset,
    create_dataset_from_json,
    download_dataset,
    export_dataset_config,
    list_datasets,
    resume_download,
    show_dataset,
)
from contextual_research_agent.cli.generation import (
    evaluate_generation,
)
from contextual_research_agent.cli.generation import (
    generate as rag_generate,
)
from contextual_research_agent.cli.ingestion import (
    ingest_dataset,
    ingest_file,
    ingest_status,
    print_ingestion_analytics,
    reingest_failed,
)
from contextual_research_agent.cli.retrieval import (
    evaluate as retrieval_evaluate,
)
from contextual_research_agent.cli.retrieval import (
    generate_eval_set,
    map_eval_queries_to_chunks,
)
from contextual_research_agent.cli.retrieval import (
    retrieve as retrieval_retrieve,
)
from contextual_research_agent.cli.robustness import robustness_eval
from contextual_research_agent.common.logging import setup_logging
from contextual_research_agent.data.kaggle.download_dataset import download_and_extract_dataset
from contextual_research_agent.generation.judge_eval import run_judge as evaluate_with_judge
from contextual_research_agent.generation.no_rag_baseline import no_rag_baseline


def main() -> None:
    setup_logging()
    fire.Fire(
        {
            # Metadata ingestion
            "download-arxiv": download_and_extract_dataset,
            "ingest-arxiv": ingest_arxiv_metadata,
            # Raw pdf ingestion
            "download-papers": download_arxiv_papers,
            # Datasets management
            "create-dataset": create_dataset,
            "create-dataset-from-json": create_dataset_from_json,
            "list-datasets": list_datasets,
            "show-dataset": show_dataset,
            "download-dataset": download_dataset,
            "export-dataset": export_dataset_config,
            "resume-download": resume_download,
            # Ingestion
            "ingest-file": ingest_file,  # single S3 path
            "ingest-dataset": ingest_dataset,  # batch by dataset name
            "ingest-status": ingest_status,  # check progress
            "reingest-failed": reingest_failed,  # retry failed from report
            "ingestion-analytics": print_ingestion_analytics,
            # Retrieval commands
            "retrieval-retrieve": retrieval_retrieve,  # query → multi-channel retrieval
            "retrieval-evaluate": retrieval_evaluate,  # run eval set → IR metrics → MLflow
            "generate-eval-set": generate_eval_set,  # corpus → synthetic eval queries
            "map-eval-queries": map_eval_queries_to_chunks,
            # Generation commands
            "generate": rag_generate,
            "evaluate-generation": evaluate_generation,
            "judge-evaluation": evaluate_with_judge,
            "no-rag-baseline": no_rag_baseline,
            "robustness-eval": robustness_eval,
            # Agent
            "agent-query": agent_query,
            "agent-eval": agent_eval,
            "agent-chat": agent_chat,
            "compare-agent-vs-pipeline": compare_agent_vs_pipeline,
        }
    )
