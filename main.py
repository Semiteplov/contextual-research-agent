import fire

from common.logging import setup_logging
from data_pipeline.arxiv import ingest_arxiv_metadata
from kaggle.download_dataset import download_and_extract_dataset


def main() -> None:
    setup_logging()
    fire.Fire(
        {
            "download-kaggle-arxiv": download_and_extract_dataset,
            "ingest-arxiv-metadata": ingest_arxiv_metadata,
        }
    )


if __name__ == "__main__":
    main()
