CREATE TABLE IF NOT EXISTS arxiv_papers (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(20) NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    file_type VARCHAR(10) NOT NULL,
    file_size_bytes BIGINT,
    checksum_sha256 VARCHAR(64),
    downloaded_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT fk_arxiv_metadata
        FOREIGN KEY (arxiv_id)
        REFERENCES arxiv_papers_metadata(arxiv_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_arxiv_papers_arxiv_id ON arxiv_papers(arxiv_id);
CREATE INDEX idx_arxiv_papers_file_type ON arxiv_papers(file_type);
