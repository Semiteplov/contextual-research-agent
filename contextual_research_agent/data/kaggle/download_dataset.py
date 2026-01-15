import zipfile
from pathlib import Path
from urllib import error, request

from contextual_research_agent.common.logging import get_logger

KAGGLE_DOWNLOAD_PATH = ".cache/kaggle"
KAGGLE_ARXIV_URL = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"

logger = get_logger(__name__)


class DatasetDownloadError(RuntimeError):
    """Raised when dataset download fails."""


class DatasetExtractionError(RuntimeError):
    """Raised when dataset extraction fails."""


def download_and_extract_dataset(
    download_path: str = KAGGLE_DOWNLOAD_PATH,
    force: bool = False,
) -> Path:
    """
    Downloads the arXiv Kaggle dataset

    Returns:
        Path to extracted directory
    """
    out_dir = Path(download_path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "arxiv.zip"
    extract_dir = out_dir / "extracted"

    if zip_path.exists() and extract_dir.exists() and not force:
        logger.info(
            "Dataset archive and extracted directory already exist. Skipping download (force=%s).",
            force,
        )
        return extract_dir

    logger.info("Starting dataset download from Kaggle")
    logger.info("Download URL: %s", KAGGLE_ARXIV_URL)
    logger.info("Download path: %s", zip_path)

    req = request.Request(
        KAGGLE_ARXIV_URL,
        method="GET",
    )

    try:
        with request.urlopen(req) as resp, Path.open(zip_path, "wb") as out:
            chunk_size = 1024 * 1024
            total_bytes = 0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                total_bytes += len(chunk)

            logger.info(
                "Dataset downloaded successfully (%d MB)",
                total_bytes // (1024 * 1024),
            )
    except error.HTTPError as e:
        logger.error(
            "HTTP error while downloading dataset: %s %s",
            e.code,
            e.reason,
        )
        raise DatasetDownloadError(
            f"HTTP error while downloading dataset: {e.code} {e.reason}"
        ) from e

    except error.URLError as e:
        logger.error("Network error while downloading dataset: %s", e.reason)
        raise DatasetDownloadError(f"Network error while downloading dataset: {e.reason}") from e

    except Exception as e:
        logger.exception("Unexpected error during dataset download")
        raise DatasetDownloadError("Unexpected error during dataset download") from e

    logger.info("Extracting dataset archive")

    try:
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=extract_dir)

    except zipfile.BadZipFile as e:
        logger.error("Downloaded file is not a valid ZIP archive")
        raise DatasetExtractionError("Invalid ZIP archive") from e

    except Exception as e:
        logger.exception("Unexpected error during dataset extraction")
        raise DatasetExtractionError("Failed to extract dataset archive") from e

    logger.info("Dataset extracted successfully to %s", extract_dir)

    return extract_dir
