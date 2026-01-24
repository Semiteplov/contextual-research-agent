from abc import ABC, abstractmethod

from contextual_research_agent.ingestion.domain.entities import Chunk, Document


class DocumentParser(ABC):
    @abstractmethod
    async def parse(self, storage_path: str) -> Document:
        """
        Parse a document file into a Document entity.

        Args:
            storage_path: Path to the document file (PDF, etc.)

        Returns:
            Parsed Document with extracted text and metadata.

        Raises:
            ParseError: If parsing fails.
        """
        ...

    @abstractmethod
    async def extract_chunks(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Extract chunks from a parsed document.

        Args:
            document: Parsed document.

        Returns:
            List of Chunk entities.
        """
        ...
