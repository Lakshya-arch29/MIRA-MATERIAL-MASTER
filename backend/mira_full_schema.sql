-- ============================================================
-- MIRA DEFINITIVE SCHEMA  (replaces ALL previous versions)
-- SIH26099 — Cross-CPSE Material Harmonization
--
-- This is the single source of truth.
-- Run on a fresh `mira` database.
-- All column names verified against actual Python route writes.
-- ============================================================




-- ============================================================
-- CPSE ORGANISATIONS
-- ============================================================

CREATE TABLE cpses (
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(255) NOT NULL,
    short_code       VARCHAR(50)  UNIQUE NOT NULL,
    sap_url          VARCHAR(500),
    last_sync_at     TIMESTAMP,
    last_sync_status VARCHAR(50),
    last_sync_error  TEXT,
    is_active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- USERS
-- ============================================================

CREATE TABLE users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name     VARCHAR(255),
    role          VARCHAR(50)  NOT NULL,
    cpse_id       INTEGER REFERENCES cpses(id),
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role  ON users(role);
CREATE INDEX idx_users_cpse  ON users(cpse_id);


-- ============================================================
-- UPLOAD BATCHES
-- ============================================================

CREATE TABLE upload_batches (
    id                 SERIAL PRIMARY KEY,
    cpse_id            INTEGER REFERENCES cpses(id),
    uploaded_by        INTEGER REFERENCES users(id),
    filename           VARCHAR(255),
    total_records      INTEGER,
    data_quality_score INTEGER,
    uploaded_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- MATERIALS
--
-- `cpse`    plain string used by ingestion & matching routes.
-- `cpse_id` FK for future normalised CPSE design (nullable).
-- Provenance columns written by resolve_provenance() on upload.
-- ============================================================

CREATE TABLE materials (
    id             BIGINT PRIMARY KEY,

    cpse           VARCHAR(100) NOT NULL,
    cpse_id        INTEGER REFERENCES cpses(id),

    material_code  VARCHAR(200) NOT NULL,
    description    TEXT NOT NULL,
    normalized_description TEXT,

    category       VARCHAR(200),
    unit           VARCHAR(50),
    manufacturer   VARCHAR(200),
    manufacturer_part_number VARCHAR(200),
    material_grade VARCHAR(200),

    dimensions            JSONB,
    specifications        JSONB,
    parsed_specifications JSONB,
    other_attributes      JSONB,

    upload_batch_id     INTEGER REFERENCES upload_batches(id),
    last_purchase_price DECIMAL(15,2),
    avg_annual_quantity DECIMAL(15,3),
    data_quality_score  INTEGER,

    status     VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Provenance (written by resolve_provenance() in ingestion)
    provenance_level      VARCHAR(50),
    provenance_confidence NUMERIC(6,4),
    provenance_source     VARCHAR(255),
    provenance_conflict   BOOLEAN NOT NULL DEFAULT FALSE,
    requires_review       BOOLEAN NOT NULL DEFAULT FALSE,
    provenance_details    JSONB   NOT NULL DEFAULT '{}'::jsonb,

    UNIQUE (cpse, material_code)
);

CREATE INDEX idx_materials_cpse     ON materials(cpse);
CREATE INDEX idx_materials_cpse_id  ON materials(cpse_id);
CREATE INDEX idx_materials_category ON materials(category);
CREATE INDEX idx_materials_code     ON materials(material_code);
CREATE INDEX idx_materials_desc_gin ON materials
    USING gin(to_tsvector('english', description));


-- Durable vectors created during ingestion. Matching reuses these vectors for
-- scoring and copies the same values into Milvus when it is available.
CREATE TABLE material_embedding_cache (
    material_id BIGINT PRIMARY KEY
        REFERENCES materials(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    text_hash VARCHAR(64) NOT NULL,
    embedding JSONB NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_material_embedding_model
    ON material_embedding_cache(model_name);


-- ============================================================
-- MATCH SUGGESTIONS
--
-- `scores`          JSONB dict written by scoring.py
--                   keys: text_similarity, semantic_similarity,
--                         specification_similarity,
--                         material_grade_similarity,
--                         other_attributes_similarity, final_score
-- `critical_checks` JSONB list written by critical_gates.py
-- `engine_decision`       HIGH_CONFIDENCE | REVIEW | DIFFERENT
-- `review_status`         PENDING | APPROVED | REJECTED |
--                         AUTO_APPROVED | DIFFERENT
-- ============================================================

CREATE TABLE match_suggestions (
    id BIGINT PRIMARY KEY,

    source_material_id BIGINT NOT NULL
        REFERENCES materials(id) ON DELETE CASCADE,
    target_material_id BIGINT NOT NULL
        REFERENCES materials(id) ON DELETE CASCADE,

    source_cpse        VARCHAR(100),
    target_cpse        VARCHAR(100),
    source_code        VARCHAR(200),
    target_code        VARCHAR(200),
    source_description TEXT,
    target_description TEXT,

    scores          JSONB NOT NULL DEFAULT '{}'::jsonb,
    critical_checks JSONB NOT NULL DEFAULT '[]'::jsonb,

    engine_decision VARCHAR(50) NOT NULL DEFAULT 'REVIEW',
    review_status   VARCHAR(50) NOT NULL DEFAULT 'PENDING',

    reviewer_id       VARCHAR(200),
    reviewer_comments TEXT,
    reviewed_at       TIMESTAMP,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_material_pair
        UNIQUE (source_material_id, target_material_id)
);

CREATE INDEX idx_match_source        ON match_suggestions(source_material_id);
CREATE INDEX idx_match_target        ON match_suggestions(target_material_id);
CREATE INDEX idx_match_engine_dec    ON match_suggestions(engine_decision);
CREATE INDEX idx_match_review_status ON match_suggestions(review_status);
CREATE INDEX idx_match_final_score   ON match_suggestions((scores->>'final_score') DESC);


-- ============================================================
-- CNMC — Common National Material Codes
--
-- `global_id`     monotonic ID from cnmc_global_id_seq
-- `material_type` written by CNMC generation service
-- ============================================================

CREATE SEQUENCE cnmc_global_id_seq;

CREATE TABLE cnmc (
    id            SERIAL PRIMARY KEY,
    cnmc_code     VARCHAR(100) UNIQUE NOT NULL,
    identity_hash VARCHAR(64)  UNIQUE,
    global_id     INTEGER      UNIQUE DEFAULT nextval('cnmc_global_id_seq'),
    material_type VARCHAR(50),

    standardized_description TEXT NOT NULL,
    category    VARCHAR(100),
    unspsc_code VARCHAR(20),

    canonical_material_record JSONB NOT NULL DEFAULT '{}'::jsonb,

    status      VARCHAR(50) NOT NULL DEFAULT 'PROVISIONAL',
    approved_by INTEGER REFERENCES users(id),
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cnmc_code          ON cnmc(cnmc_code);
CREATE INDEX idx_cnmc_identity_hash ON cnmc(identity_hash);
CREATE INDEX idx_cnmc_global_id     ON cnmc(global_id);
CREATE INDEX idx_cnmc_category      ON cnmc(category);
CREATE INDEX idx_cnmc_status        ON cnmc(status);


-- ============================================================
-- MAPPINGS — CPSE codes grouped into NMC clusters
--
-- `cpse_mappings`          JSONB array of
--                          {cpse, material_code, description, ...}
-- `common_material_record` JSONB synthesised CMR
-- ============================================================

CREATE TABLE mappings (
    id            BIGINT PRIMARY KEY,
    nmc           VARCHAR(100) NOT NULL,
    cpse_mappings JSONB NOT NULL DEFAULT '[]'::jsonb,
    cluster_size  INTEGER NOT NULL DEFAULT 0,
    status        VARCHAR(50) NOT NULL DEFAULT 'PROVISIONAL',
    common_material_record JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_mappings_nmc    ON mappings(nmc);
CREATE INDEX idx_mappings_status ON mappings(status);


-- ============================================================
-- FEEDBACK
-- ============================================================

CREATE TABLE feedback (
    id                  BIGSERIAL PRIMARY KEY,
    match_suggestion_id BIGINT NOT NULL
        REFERENCES match_suggestions(id) ON DELETE CASCADE,
    reviewer_id         VARCHAR(200) NOT NULL,
    action              VARCHAR(50)  NOT NULL,
    reason              TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_match ON feedback(match_suggestion_id);


-- ============================================================
-- AUDIT LOG
--
-- Column named `timestamp` (not `created_at`) because the
-- Python routes insert the key "timestamp" explicitly:
--   AUDIT_EVENTS.append({ ... "timestamp": now ... })
-- ============================================================

CREATE TABLE audit_logs (
    id           BIGSERIAL PRIMARY KEY,
    event_type   VARCHAR(100) NOT NULL,
    candidate_id BIGINT,

    source_code  VARCHAR(200),
    target_code  VARCHAR(200),
    source_cpse  VARCHAR(100),
    target_cpse  VARCHAR(100),

    actor       VARCHAR(200),
    comments    TEXT,
    final_score DECIMAL(6,4),

    "timestamp" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_candidate  ON audit_logs(candidate_id);
CREATE INDEX idx_audit_timestamp  ON audit_logs("timestamp");
