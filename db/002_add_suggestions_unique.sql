-- Run manually against the already-running container (schema.sql only
-- auto-applies on first container init, not on later edits).
ALTER TABLE suggestions ADD CONSTRAINT uq_suggestions_post_image UNIQUE (post_id, image_id);