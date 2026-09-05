"""
Backfills cost_log with $0.00 entries for the 56 embedding calls already
made (42 image + 14 post). Honest to do retroactively, unlike vision costs:
local sentence-transformers embeddings have zero API cost, reproducibly and
verifiably, regardless of when they were computed -- there's no "real"
number that could have been different.
"""
import json
from src.db import get_cursor


def backfill_embedding_costs():
    with open("data/image_embeddings.json", "r", encoding="utf-8") as f:
        image_embeddings = json.load(f)

    with open("data/post_embeddings.json", "r", encoding="utf-8") as f:
        post_embeddings = json.load(f)

    with get_cursor() as cur:
        for file_path in image_embeddings:
            cur.execute(
                """
                INSERT INTO cost_log (call_type, reference, input_tokens, output_tokens, cost_usd)
                VALUES ('embedding', %s, NULL, NULL, 0.0)
                """,
                (file_path,),
            )
        for external_id in post_embeddings:
            cur.execute(
                """
                INSERT INTO cost_log (call_type, reference, input_tokens, output_tokens, cost_usd)
                VALUES ('embedding', %s, NULL, NULL, 0.0)
                """,
                (external_id,),
            )

    print(f"Backfilled {len(image_embeddings)} image + {len(post_embeddings)} post embedding cost entries ($0.00 each, local model).")


if __name__ == "__main__":
    backfill_embedding_costs()