-- Add durable material vectors without altering existing material rows.
CREATE TABLE IF NOT EXISTS material_embedding_cache (
    material_id BIGINT PRIMARY KEY
        REFERENCES materials(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    text_hash VARCHAR(64) NOT NULL,
    embedding JSONB NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_material_embedding_model
    ON material_embedding_cache(model_name);
