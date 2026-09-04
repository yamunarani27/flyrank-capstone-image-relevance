import json
from pathlib import Path
from src.embeddings import get_embedding

POSTS_PATH = "data/posts.json"
POST_EMBEDDINGS_PATH = "data/post_embeddings.json"

def embed_posts():
    with open(POSTS_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    if Path(POST_EMBEDDINGS_PATH).exists():
        with open(POST_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
    else:
        embeddings = {}

    for post in posts:
        post_id = post.get("id")
        if post_id in embeddings:
            continue

        text = f"{post.get('title', '')}. {post.get('body', '')}"
        vector = get_embedding(text)
        embeddings[post_id] = {
            "embedding": vector,
            "model_name": "all-MiniLM-L6-v2",
        }
        print(f"Embedded post {post_id}")

    with open(POST_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Done. {len(embeddings)} total post embeddings saved to {POST_EMBEDDINGS_PATH}")

if __name__ == "__main__":
    embed_posts()