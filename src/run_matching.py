import json
from src.similarity import load_embeddings, rank_images_for_post, compute_dynamic_threshold
from src.guard import evaluate_match
from src.schema import ImageMetadata

post_embeddings = load_embeddings("data/post_embeddings.json")
image_embeddings = load_embeddings("data/image_embeddings.json")

with open("data/posts.json", "r", encoding="utf-8") as f:
    posts = json.load(f)

with open("data/tagged_images.json", "r", encoding="utf-8") as f:
    tagged_images = json.load(f)

for post in posts:
    post_id = post["id"]
    ranked = rank_images_for_post(post_id, post_embeddings, image_embeddings, top_k=5)

    top_similarity = ranked[0][1]
    dynamic_threshold = compute_dynamic_threshold(top_similarity)

    print(f"\n{post['title']} (dynamic threshold: {dynamic_threshold:.3f})")

    for image_id, score in ranked:
        image_data = tagged_images[image_id]
        candidate = ImageMetadata(**image_data)  # adjust if your constructor differs

        result = evaluate_match(
            post_expected_category=post["expected_category"],
            post_expected_subject=post["expected_subject"],
            candidate=candidate,
            similarity=score,
            similarity_threshold=dynamic_threshold,
        )
        print(f"  {image_id}: {result.decision.value} — {result.reason}")