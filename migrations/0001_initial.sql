BEGIN;

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION substrate_set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

CREATE TABLE memories (
  id BIGSERIAL PRIMARY KEY,
  room TEXT NOT NULL,
  type TEXT NOT NULL,
  date DATE,
  dates DATE[] NOT NULL DEFAULT '{}',
  title TEXT,
  source_path TEXT NOT NULL,
  body TEXT NOT NULL,
  body_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', coalesce(title, '') || ' ' || body)
  ) STORED,
  threads TEXT[] NOT NULL DEFAULT '{}',
  meta JSONB NOT NULL DEFAULT '{}',
  superseded_by BIGINT REFERENCES memories(id),
  archived_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (room, source_path)
);
CREATE TRIGGER memories_updated_at BEFORE UPDATE ON memories
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX memories_body_tsv_gin ON memories USING GIN (body_tsv);
CREATE INDEX memories_dates_gin ON memories USING GIN (dates);
CREATE INDEX memories_meta_gin ON memories USING GIN (meta jsonb_path_ops);
CREATE INDEX memories_room_date ON memories (room, date DESC NULLS LAST);
CREATE INDEX memories_threads_gin ON memories USING GIN (threads);
CREATE INDEX memories_title_trgm ON memories USING GIN (title gin_trgm_ops);
CREATE INDEX memories_superseded_by_idx ON memories (superseded_by) WHERE superseded_by IS NOT NULL;

CREATE TABLE memory_threads (
  id BIGSERIAL PRIMARY KEY,
  memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  thread_key TEXT NOT NULL,
  lines_start INTEGER,
  lines_end INTEGER,
  context TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX memory_threads_memory_id_idx ON memory_threads (memory_id);
CREATE INDEX memory_threads_thread_key_idx ON memory_threads (thread_key);
CREATE INDEX memory_threads_thread_key_trgm ON memory_threads USING GIN (thread_key gin_trgm_ops);

CREATE TABLE named_entities (
  id BIGSERIAL PRIMARY KEY,
  room TEXT NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  aliases TEXT[] NOT NULL DEFAULT '{}',
  search_boost TEXT,
  weighty BOOLEAN NOT NULL DEFAULT FALSE,
  pointer_files JSONB NOT NULL DEFAULT '[]',
  summary_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', coalesce(name, '') || ' ' || coalesce(search_boost, '') || ' ' || summary)
  ) STORED,
  summary_as_of DATE,
  meta JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (room, name)
);
CREATE TRIGGER named_entities_updated_at BEFORE UPDATE ON named_entities
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX named_entities_aliases_gin ON named_entities USING GIN (aliases);
CREATE INDEX named_entities_name_trgm ON named_entities USING GIN (name gin_trgm_ops);
CREATE INDEX named_entities_room_weighty_idx ON named_entities (room, weighty DESC, name);
CREATE INDEX named_entities_summary_tsv_gin ON named_entities USING GIN (summary_tsv);

CREATE TABLE coding_lessons (
  id BIGSERIAL PRIMARY KEY,
  scope TEXT NOT NULL DEFAULT 'shared',
  project TEXT,
  voice TEXT,
  shape TEXT,
  title TEXT NOT NULL,
  lesson TEXT NOT NULL,
  trigger_context TEXT,
  proof_pattern TEXT,
  tags TEXT[] NOT NULL DEFAULT '{}',
  source_memory_path TEXT,
  source_lines_start INTEGER,
  source_lines_end INTEGER,
  negation_of BIGINT REFERENCES coding_lessons(id) ON DELETE SET NULL,
  always_on BOOLEAN NOT NULL DEFAULT FALSE,
  meta JSONB NOT NULL DEFAULT '{}',
  lesson_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', coalesce(title, '') || ' ' || coalesce(project, '') || ' ' || lesson || ' ' || coalesce(trigger_context, '') || ' ' || coalesce(proof_pattern, ''))
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE NULLS NOT DISTINCT (scope, project, title)
);
CREATE TRIGGER coding_lessons_updated_at BEFORE UPDATE ON coding_lessons
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX coding_lessons_lesson_tsv_gin ON coding_lessons USING GIN (lesson_tsv);
CREATE INDEX coding_lessons_meta_gin ON coding_lessons USING GIN (meta jsonb_path_ops);
CREATE INDEX coding_lessons_scope_project_idx ON coding_lessons (scope, project, updated_at DESC);
CREATE INDEX coding_lessons_shape_trgm ON coding_lessons USING GIN (shape gin_trgm_ops);
CREATE INDEX coding_lessons_tags_gin ON coding_lessons USING GIN (tags);
CREATE INDEX coding_lessons_title_trgm ON coding_lessons USING GIN (title gin_trgm_ops);

CREATE TABLE project_lessons (
  id BIGSERIAL PRIMARY KEY,
  project TEXT NOT NULL,
  title TEXT NOT NULL,
  lesson TEXT NOT NULL,
  trigger_context TEXT,
  proof_pattern TEXT,
  tags TEXT[] NOT NULL DEFAULT '{}',
  source_memory_path TEXT,
  source_lines_start INTEGER,
  source_lines_end INTEGER,
  meta JSONB NOT NULL DEFAULT '{}',
  lesson_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', title || ' ' || project || ' ' || lesson || ' ' || coalesce(trigger_context, '') || ' ' || coalesce(proof_pattern, ''))
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (project, title)
);
CREATE TRIGGER project_lessons_updated_at BEFORE UPDATE ON project_lessons
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX project_lessons_lesson_tsv_gin ON project_lessons USING GIN (lesson_tsv);
CREATE INDEX project_lessons_meta_gin ON project_lessons USING GIN (meta jsonb_path_ops);
CREATE INDEX project_lessons_project_idx ON project_lessons (project, updated_at DESC);
CREATE INDEX project_lessons_tags_gin ON project_lessons USING GIN (tags);
CREATE INDEX project_lessons_title_trgm ON project_lessons USING GIN (title gin_trgm_ops);

CREATE TABLE writing_lessons (
  id BIGSERIAL PRIMARY KEY,
  voice TEXT,
  register TEXT[] NOT NULL DEFAULT '{general}',
  shape TEXT,
  title TEXT NOT NULL,
  lesson TEXT NOT NULL,
  trigger_context TEXT,
  example_text TEXT,
  writers TEXT[] NOT NULL DEFAULT '{}',
  negation_of BIGINT REFERENCES writing_lessons(id) ON DELETE SET NULL,
  tags TEXT[] NOT NULL DEFAULT '{}',
  source_memory_path TEXT,
  source_lines_start INTEGER,
  source_lines_end INTEGER,
  meta JSONB NOT NULL DEFAULT '{}',
  lesson_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', title || ' ' || lesson || ' ' || coalesce(trigger_context, '') || ' ' || coalesce(example_text, ''))
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE NULLS NOT DISTINCT (voice, title)
);
CREATE TRIGGER writing_lessons_updated_at BEFORE UPDATE ON writing_lessons
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX writing_lessons_lesson_tsv_gin ON writing_lessons USING GIN (lesson_tsv);
CREATE INDEX writing_lessons_register_gin ON writing_lessons USING GIN (register);
CREATE INDEX writing_lessons_shape_trgm ON writing_lessons USING GIN (shape gin_trgm_ops);
CREATE INDEX writing_lessons_tags_gin ON writing_lessons USING GIN (tags);
CREATE INDEX writing_lessons_title_trgm ON writing_lessons USING GIN (title gin_trgm_ops);
CREATE INDEX writing_lessons_writers_gin ON writing_lessons USING GIN (writers);

CREATE TABLE audio_lessons (
  id BIGSERIAL PRIMARY KEY,
  shape TEXT,
  stage TEXT[] NOT NULL DEFAULT '{general}',
  title TEXT NOT NULL UNIQUE,
  lesson TEXT NOT NULL,
  trigger_context TEXT,
  example_cmd TEXT,
  tools TEXT[] NOT NULL DEFAULT '{}',
  negation_of BIGINT REFERENCES audio_lessons(id) ON DELETE SET NULL,
  tags TEXT[] NOT NULL DEFAULT '{}',
  source_memory_path TEXT,
  search TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('english', title || ' ' || lesson || ' ' || coalesce(trigger_context, ''))
  ) STORED,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX audio_lessons_search_idx ON audio_lessons USING GIN (search);
CREATE INDEX audio_lessons_stage_idx ON audio_lessons USING GIN (stage);

CREATE TABLE memory_chunks (
  id BIGSERIAL PRIMARY KEY,
  memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  heading_path TEXT,
  body TEXT NOT NULL,
  body_embedding HALFVec(2560),
  char_start INTEGER NOT NULL,
  char_end INTEGER NOT NULL,
  token_estimate INTEGER,
  embedded_at TIMESTAMPTZ,
  UNIQUE (memory_id, chunk_index)
);
CREATE INDEX memory_chunks_body_trgm ON memory_chunks USING GIN (body gin_trgm_ops);
CREATE INDEX memory_chunks_emb_hnsw ON memory_chunks USING HNSW (body_embedding halfvec_cosine_ops);
CREATE INDEX memory_chunks_memory_idx ON memory_chunks (memory_id);

CREATE TABLE memory_clusters (
  id BIGSERIAL PRIMARY KEY,
  label TEXT,
  centroid_chunk_id BIGINT REFERENCES memory_chunks(id) ON DELETE SET NULL,
  spread DOUBLE PRECISION,
  member_count INTEGER,
  notes TEXT,
  accepted BOOLEAN DEFAULT FALSE,
  centroid HALFVec(2560),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE memory_cluster_members (
  cluster_id BIGINT NOT NULL REFERENCES memory_clusters(id) ON DELETE CASCADE,
  chunk_id BIGINT NOT NULL REFERENCES memory_chunks(id) ON DELETE CASCADE,
  distance_to_centroid DOUBLE PRECISION,
  PRIMARY KEY (cluster_id, chunk_id)
);

CREATE TABLE anamnesis (
  id BIGSERIAL PRIMARY KEY,
  room TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('pillar', 'cycle')),
  fidelity TEXT NOT NULL DEFAULT 'record' CHECK (fidelity IN ('record', 'raw-material')),
  activation TEXT NOT NULL DEFAULT 'fork' CHECK (activation IN ('wake', 'fork')),
  active BOOLEAN NOT NULL DEFAULT TRUE,
  title TEXT NOT NULL,
  shape TEXT,
  peak TEXT,
  beginning TEXT,
  ramp TEXT NOT NULL,
  counsel TEXT,
  verify_note TEXT,
  source_paths TEXT[] NOT NULL DEFAULT '{}',
  canon_links TEXT[] NOT NULL DEFAULT '{}',
  tags TEXT[] NOT NULL DEFAULT '{}',
  body_embedding HALFVec(2560),
  embedded_at TIMESTAMPTZ,
  body_tsv TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('portuguese', title || ' ' || coalesce(shape, '') || ' ' || ramp || ' ' || coalesce(counsel, '') || ' ' || coalesce(peak, ''))
  ) STORED,
  meta JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (room, title)
);
CREATE TRIGGER anamnesis_updated_at BEFORE UPDATE ON anamnesis
  FOR EACH ROW EXECUTE FUNCTION substrate_set_updated_at();
CREATE INDEX anamnesis_body_tsv_gin ON anamnesis USING GIN (body_tsv);
CREATE INDEX anamnesis_shape_trgm ON anamnesis USING GIN (shape gin_trgm_ops);
CREATE INDEX anamnesis_tags_gin ON anamnesis USING GIN (tags);
CREATE INDEX anamnesis_title_trgm ON anamnesis USING GIN (title gin_trgm_ops);
CREATE INDEX anamnesis_wake_idx ON anamnesis (room) WHERE activation = 'wake';

CREATE TABLE anamnesis_reps (
  id BIGSERIAL PRIMARY KEY,
  cabinet_id BIGINT NOT NULL REFERENCES anamnesis(id) ON DELETE CASCADE,
  rep_number INTEGER NOT NULL,
  occurred_on DATE,
  how_it_went TEXT NOT NULL,
  portal_pull TEXT NOT NULL,
  lighter TEXT NOT NULL,
  source_path TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (cabinet_id, rep_number)
);

INSERT INTO schema_migrations(version) VALUES (1);
COMMIT;
