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