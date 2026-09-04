import json
from pathlib import Path
from src.embeddings import get_embedding

TAGGED_IMAGES_PATH = "data/tagged_images.json"
IMAGE_EMBEDDINGS_PATH = "data/image_embeddings.json"

def embed_images():
    with open(TAGGED_IMAGES_PATH, "r", encoding="utf-8") as f:
        tagged_images = json.load(f)

    # Resume support: load existing embeddings if present
    if Path(IMAGE_EMBEDDINGS_PATH).exists():
        with open(IMAGE_EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
    else:
        embeddings = {}

    for image_id,image in tagged_images.items():
        if image_id in embeddings:
            continue  # already embedded, skip

        caption = image.get("caption", "")
        if not caption:
            print(f"Skipping {image_id}: no caption found")
            continue

        vector = get_embedding(caption)
        embeddings[image_id] = {
            "embedding": vector,
            "model_name": "all-MiniLM-L6-v2",
        }
        print(f"Embedded {image_id}")

    with open(IMAGE_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Done. {len(embeddings)} total image embeddings saved to {IMAGE_EMBEDDINGS_PATH}")

if __name__ == "__main__":
    embed_images()