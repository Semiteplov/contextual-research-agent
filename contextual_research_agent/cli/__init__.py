import fire

from contextual_research_agent.cli.arxiv import download_arxiv_papers, ingest_arxiv_metadata
from contextual_research_agent.common.logging import setup_logging
from contextual_research_agent.data.kaggle.download_dataset import download_and_extract_dataset


def main() -> None:
    setup_logging()
    fire.Fire(
        {
            "download-arxiv": download_and_extract_dataset,
            "ingest-arxiv": ingest_arxiv_metadata,
            "download-papers": download_arxiv_papers,
        }
    )
