import json
import numpy as np

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity between two embedding vectors, in [-1, 1]."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_embeddings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rank_images_for_post(post_id: str, post_embeddings: dict, image_embeddings: dict, top_k: int = 5) -> list[tuple[str, float]]:
    """Return the top_k images ranked by cosine similarity to the given post."""
    post_vector = post_embeddings[post_id]["embedding"]

    scored = []
    for image_id, data in image_embeddings.items():
        score = cosine_similarity(post_vector, data["embedding"])
        scored.append((image_id, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]

def compute_dynamic_threshold(top_similarity: float, margin: float = 0.15) -> float:
    """
    Accept candidates within `margin` of this post's best similarity score,
    rather than against a fixed global bar. Real data shows absolute cosine
    scores vary a lot by post writing style (e.g. ~0.5-0.6 for visual/descriptive
    posts vs ~0.2-0.3 for behavior/abstract posts like dog training), so a
    single global threshold can't separate "good match" from "bad match"
    across all posts. margin=0.15 is a starting point, pending validation
    against a labeled eval set.
    Known limitation: a fixed margin doesn't scale well for posts with low
    top-similarity scores (e.g. dog posts), which can let same-category but
    wrong-subject images (e.g. a wolf approved for a deer post) slip past.
    _CONFUSABLE_PAIRS only guards a hand-picked set of pairs, not all
    possible confusions. To be validated/tuned against a labeled eval set.
    """
    return max(0.0, top_similarity - margin)