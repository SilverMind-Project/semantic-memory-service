-- 002_description_embedding.sql
-- Add text embedding column for scene description semantic search
-- Uses 384-dimensional vectors from sentence-transformers/all-MiniLM-L6-v2

ALTER TABLE scene_observations
ADD COLUMN description_embedding vector(384);

CREATE INDEX idx_scene_obs_description_embedding
ON scene_observations USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
