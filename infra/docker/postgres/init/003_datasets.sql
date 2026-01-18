CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    selection_criteria JSONB NOT NULL,
    train_ratio REAL NOT NULL DEFAULT 0.8,
    val_ratio REAL NOT NULL DEFAULT 0.1,
    test_ratio REAL NOT NULL DEFAULT 0.1,
    random_seed INT NOT NULL DEFAULT 42,
    total_papers INT NOT NULL DEFAULT 0,
    purpose VARCHAR(50) NOT NULL DEFAULT 'training',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    version INT NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS dataset_papers (
    dataset_id INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    arxiv_id VARCHAR(20) NOT NULL REFERENCES arxiv_papers_metadata(arxiv_id),
    split VARCHAR(20) NOT NULL DEFAULT 'train',
    added_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (dataset_id, arxiv_id)
);

CREATE INDEX idx_dataset_papers_arxiv ON dataset_papers(arxiv_id);
CREATE INDEX idx_dataset_papers_split ON dataset_papers(dataset_id, split);

ALTER TABLE arxiv_papers
ADD CONSTRAINT uq_arxiv_papers_id_type UNIQUE (arxiv_id, file_type);

ALTER TABLE arxiv_papers
ADD COLUMN IF NOT EXISTS source_format VARCHAR(10);
