"""
One-time migration: loads existing JSON files (tagged_images.json,
posts.json, image_embeddings.json, post_embeddings.json) into Postgres.

Idempotent-ish: uses ON CONFLICT DO NOTHING on the natural unique keys
(images.file_path, posts.external_id), so re-running is safe but won't
update already-migrated rows — for real updates, truncate and rerun.
"""
import json
from src.db import get_cursor


def migrate_images():
    with open("data/tagged_images.json", "r", encoding="utf-8") as f:
        tagged_images = json.load(f)

    with get_cursor() as cur:
        for file_path, data in tagged_images.items():
            cur.execute(
                """
                INSERT INTO images (file_path, subject, category, attributes, caption, confidence, low_confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO NOTHING
                RETURNING id
                """,
                (
                    file_path,
                    data["subject"],
                    data["category"],
                    data.get("attributes", []),
                    data["caption"],
                    data["confidence"],
                    data.get("low_confidence", False),
                ),
            )

    print(f"Migrated {len(tagged_images)} images.")


def migrate_posts():
    with open("data/posts.json", "r", encoding="utf-8") as f:
        posts = json.load(f)

    with get_cursor() as cur:
        for post in posts:
            cur.execute(
                """
                INSERT INTO posts (external_id, title, body, expected_category, expected_subject)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (external_id) DO NOTHING
                """,
                (post["id"], post["title"], post["body"], post["expected_category"], post["expected_subject"]),
            )

    print(f"Migrated {len(posts)} posts.")


def migrate_image_embeddings():
    with open("data/image_embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    with get_cursor() as cur:
        for file_path, data in embeddings.items():
            cur.execute("SELECT id FROM images WHERE file_path = %s", (file_path,))
            row = cur.fetchone()
            if row is None:
                print(f"  Skipping embedding for {file_path}: no matching image row")
                continue

            cur.execute(
                """
                INSERT INTO image_embeddings (image_id, embedding, model_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (image_id) DO NOTHING
                """,
                (row["id"], data["embedding"], data["model_name"]),
            )

    print(f"Migrated {len(embeddings)} image embeddings.")


def migrate_post_embeddings():
    with open("data/post_embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    with get_cursor() as cur:
        for external_id, data in embeddings.items():
            cur.execute("SELECT id FROM posts WHERE external_id = %s", (external_id,))
            row = cur.fetchone()
            if row is None:
                print(f"  Skipping embedding for {external_id}: no matching post row")
                continue

            cur.execute(
                """
                INSERT INTO post_embeddings (post_id, embedding, model_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (post_id) DO NOTHING
                """,
                (row["id"], data["embedding"], data["model_name"]),
            )

    print(f"Migrated {len(embeddings)} post embeddings.")


if __name__ == "__main__":
    migrate_images()
    migrate_posts()
    migrate_image_embeddings()
    migrate_post_embeddings()
    print("Migration complete.")