CREATE TABLE IF NOT EXISTS extracted_texts (
    id SERIAL PRIMARY KEY,
    arxiv_id VARCHAR(20) NOT NULL,
    extraction_method VARCHAR(20) NOT NULL,
    storage_path TEXT NOT NULL,
    num_pages INT,
    num_characters INT,
    num_words INT,
    language VARCHAR(10),
    status VARCHAR(20) NOT NULL DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_extracted_texts_arxiv_method UNIQUE (arxiv_id, extraction_method),

    CONSTRAINT fk_extracted_arxiv
        FOREIGN KEY (arxiv_id)
        REFERENCES arxiv_papers_metadata(arxiv_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_extracted_texts_arxiv ON extracted_texts(arxiv_id);
CREATE INDEX idx_extracted_texts_method ON extracted_texts(extraction_method);
CREATE INDEX idx_extracted_texts_status ON extracted_texts(status);
