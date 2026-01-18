import fire

from contextual_research_agent.cli.arxiv import download_arxiv_papers, ingest_arxiv_metadata
from contextual_research_agent.cli.datasets import (
    create_dataset,
    download_dataset,
    download_sources,
    export_dataset_config,
    list_datasets,
    resume_download,
    show_dataset,
)
from contextual_research_agent.common.logging import setup_logging
from contextual_research_agent.data.kaggle.download_dataset import download_and_extract_dataset


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
            "list-datasets": list_datasets,
            "show-dataset": show_dataset,
            "download-dataset": download_dataset,
            "export-dataset": export_dataset_config,
            "resume-download": resume_download,
            "download-sources": download_sources,
        }
    )
