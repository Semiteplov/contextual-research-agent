\connect arxiv

CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,               -- Original form: "LoRA", "SQuAD 2.0"
    entity_type VARCHAR(50) NOT NULL,         -- method, dataset, task, metric, model
    normalized_name VARCHAR(255) NOT NULL,    -- Lowercase, deduplicated: "lora", "squad 2.0"
    description TEXT,                         -- Optional description
    source VARCHAR(50) DEFAULT 'extracted',   -- extracted, manual, paperwithcode
    aliases TEXT[],                           -- Alternative names: {"Low-Rank Adaptation", "LoRA"}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (normalized_name, entity_type)
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_normalized ON entities(normalized_name);
CREATE INDEX idx_entities_name ON entities(name);

COMMENT ON TABLE entities IS 'Scientific concepts: methods, datasets, tasks, metrics, models';
COMMENT ON COLUMN entities.normalized_name IS 'Lowercase deduplicated name for matching';
COMMENT ON COLUMN entities.source IS 'How this entity was added: extracted (NER/LLM), manual, paperwithcode';

CREATE TABLE IF NOT EXISTS citation_edges (
    id SERIAL PRIMARY KEY,
    citing_paper_id VARCHAR(20) NOT NULL,        -- arxiv_id of citing paper
    cited_paper_id VARCHAR(255) NOT NULL,        -- arxiv_id or DOI of cited paper
    cited_id_type VARCHAR(20) DEFAULT 'arxiv',   -- arxiv, doi, unknown

    -- Citation context
    context TEXT,                                -- Sentence where citation appears
    section VARCHAR(500),                        -- Section heading
    section_type VARCHAR(50),                    -- Classified section type

    -- Reference metadata (from bibliography)
    ref_key VARCHAR(100),                        -- Internal ref key: "ref_12"
    cited_title TEXT,                            -- Title from bibliography
    cited_authors TEXT,                          -- Authors string
    cited_year VARCHAR(10),                      -- Publication year

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (citing_paper_id, cited_paper_id)
);

CREATE INDEX idx_citations_citing ON citation_edges(citing_paper_id);
CREATE INDEX idx_citations_cited ON citation_edges(cited_paper_id);
CREATE INDEX idx_citations_section_type ON citation_edges(section_type);
CREATE INDEX idx_citations_cited_type ON citation_edges(cited_id_type);

COMMENT ON TABLE citation_edges IS 'Directed citation links between papers with context';
COMMENT ON COLUMN citation_edges.context IS 'Sentence where the citation anchor appears';
COMMENT ON COLUMN citation_edges.section_type IS 'Section type where citation occurs (introduction, method, etc.)';


CREATE TABLE IF NOT EXISTS paper_entity_edges (
    id SERIAL PRIMARY KEY,
    paper_id VARCHAR(20) NOT NULL,
    entity_id INT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation VARCHAR(50) NOT NULL,

    confidence REAL DEFAULT 1.0,
    evidence TEXT,
    section_type VARCHAR(50),
    extraction_method VARCHAR(50) DEFAULT 'rule',

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (paper_id, entity_id, relation)
);

CREATE INDEX idx_paper_entities_paper ON paper_entity_edges(paper_id);
CREATE INDEX idx_paper_entities_entity ON paper_entity_edges(entity_id);
CREATE INDEX idx_paper_entities_relation ON paper_entity_edges(relation);
CREATE INDEX idx_paper_entities_section ON paper_entity_edges(section_type);

COMMENT ON TABLE paper_entity_edges IS 'Relationships between papers and scientific entities';
COMMENT ON COLUMN paper_entity_edges.relation IS 'Relation type: uses_method, uses_dataset, targets_task, reports_metric, uses_model, compared_with';
COMMENT ON COLUMN paper_entity_edges.confidence IS 'Extraction confidence from NER/LLM (1.0 = certain)';

CREATE OR REPLACE VIEW v_citing_papers AS
SELECT
    ce.cited_paper_id,
    ce.citing_paper_id,
    am.title AS citing_title,
    EXTRACT(YEAR FROM am.update_date)::INT AS citing_year,
    ce.context,
    ce.section_type
FROM citation_edges ce
LEFT JOIN arxiv_papers_metadata am ON ce.citing_paper_id = am.arxiv_id;

CREATE OR REPLACE VIEW v_cited_papers AS
SELECT
    ce.citing_paper_id,
    ce.cited_paper_id,
    ce.cited_title,
    ce.cited_year,
    ce.context,
    ce.section_type
FROM citation_edges ce;

CREATE OR REPLACE VIEW v_entity_cooccurrence AS
SELECT
    pe1.paper_id AS paper_a,
    pe2.paper_id AS paper_b,
    e.name AS entity_name,
    e.entity_type,
    pe1.relation AS relation_a,
    pe2.relation AS relation_b
FROM paper_entity_edges pe1
JOIN paper_entity_edges pe2
    ON pe1.entity_id = pe2.entity_id
    AND pe1.paper_id < pe2.paper_id
JOIN entities e ON pe1.entity_id = e.id;

CREATE OR REPLACE VIEW v_paper_entities AS
SELECT
    pe.paper_id,
    e.name AS entity_name,
    e.entity_type,
    pe.relation,
    pe.confidence,
    pe.evidence
FROM paper_entity_edges pe
JOIN entities e ON pe.entity_id = e.id
ORDER BY pe.paper_id, e.entity_type, e.name;