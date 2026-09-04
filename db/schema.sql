-- Schema for the image-blog matching capstone.
-- Runs automatically on first container start (docker-entrypoint-initdb.d).
-- Plain float[] arrays for embeddings, not pgvector — fine at ~50 images
-- per the brief's own scope note.

CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    category TEXT NOT NULL,
    attributes TEXT[] NOT NULL DEFAULT '{}',
    caption TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    low_confidence BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'tagged',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_images_category ON images (category);
CREATE INDEX idx_images_subject ON images (subject);

CREATE TABLE image_embeddings (
    image_id INTEGER PRIMARY KEY REFERENCES images (id) ON DELETE CASCADE,
    embedding REAL[] NOT NULL,
    model_name TEXT NOT NULL
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,  -- e.g. "post_01", matches posts.json ids
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    expected_category TEXT NOT NULL,
    expected_subject TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_posts_expected_subject ON posts (expected_subject);

CREATE TABLE post_embeddings (
    post_id INTEGER PRIMARY KEY REFERENCES posts (id) ON DELETE CASCADE,
    embedding REAL[] NOT NULL,
    model_name TEXT NOT NULL
);

CREATE TABLE suggestions (
    id SERIAL PRIMARY KEY,
    post_id INTEGER NOT NULL REFERENCES posts (id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images (id) ON DELETE CASCADE,
    similarity REAL NOT NULL,
    decision TEXT NOT NULL,  -- 'approved' | 'rejected' | 'no_confident_match'
    reason TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX idx_suggestions_post_id ON suggestions (post_id);
CREATE INDEX idx_suggestions_image_id ON suggestions (image_id);
CREATE INDEX idx_suggestions_review_status ON suggestions (review_status);

CREATE TABLE cost_log (
    id SERIAL PRIMARY KEY,
    call_type TEXT NOT NULL,  -- 'vision_tagging' | 'embedding'
    reference TEXT,           -- e.g. image file_path or post external_id
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd NUMERIC(10, 6) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_cost_log_call_type ON cost_log (call_type);

CREATE TABLE batch_jobs (
    id SERIAL PRIMARY KEY,
    job_type TEXT NOT NULL,  -- 'vision_tagging' | 'embedding'
    status TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'completed' | 'failed'
    total_items INTEGER NOT NULL DEFAULT 0,
    succeeded_items INTEGER NOT NULL DEFAULT 0,
    failed_items INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);