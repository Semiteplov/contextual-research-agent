\connect arxiv

CREATE TABLE IF NOT EXISTS arxiv_papers_metadata (
  arxiv_id TEXT PRIMARY KEY,
  title TEXT,
  abstract TEXT,
  authors TEXT,
  categories TEXT[] NOT NULL,
  primary_category TEXT,
  doi TEXT,
  journal_ref TEXT,
  update_date DATE,
  latest_version INT,
  latest_version_created TIMESTAMPTZ,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS arxiv_category_sync_state (
  category TEXT PRIMARY KEY,
  last_synced_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_arxiv_papers_metadata_primary_category ON arxiv_papers_metadata(primary_category);
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_metadata_categories_gin ON arxiv_papers_metadata USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_arxiv_papers_metadata_update_date ON arxiv_papers_metadata(update_date);
