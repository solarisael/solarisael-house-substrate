-- Migrate the semantic space from Qwen halfvec(2560) to Nemotron vector(2048).
-- Existing vectors and clusters belong to the old embedding space and cannot be reused.

BEGIN;

DROP INDEX IF EXISTS memory_chunks_emb_hnsw;

DELETE FROM memory_cluster_members;
DELETE FROM memory_clusters;

UPDATE memory_chunks
SET body_embedding = NULL,
    embedded_at = NULL;

ALTER TABLE memory_chunks
    ALTER COLUMN body_embedding TYPE vector(2048)
    USING NULL::vector(2048);

UPDATE anamnesis
SET body_embedding = NULL,
    embedded_at = NULL;

ALTER TABLE anamnesis
    ALTER COLUMN body_embedding TYPE vector(2048)
    USING NULL::vector(2048);

ALTER TABLE memory_clusters
    ALTER COLUMN centroid TYPE vector(2048)
    USING NULL::vector(2048);

INSERT INTO schema_migrations (version, applied_at)
VALUES (2, NOW())
ON CONFLICT (version) DO NOTHING;

COMMIT;
