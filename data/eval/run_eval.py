"""
Runs the labeled eval set and reports two numbers:

1. Top-1 precision: of all posts, what fraction had a correct image ranked
   first (before the guard even runs) — this measures ranking quality.
2. Mismatch rejection rate: for a curated set of known-wrong pairings
   (fox/wolf, dog/wolf, deer/wolf), what fraction does the guard correctly
   reject — this measures the safety layer, independent of ranking.

Both numbers belong in README.md and EVIDENCE.md as your quality proof.
"""
import json
from src.similarity import load_embeddings, rank_images_for_post, compute_dynamic_threshold
from src.guard import evaluate_match
from src.schema import ImageMetadata

LABELS_PATH = "data/eval/labels.json"
POSTS_PATH = "data/posts.json"
TAGGED_IMAGES_PATH = "data/tagged_images.json"

# All C(5,2)=10 same-category (animal) pairings, one representative post
# and one representative image per species. Landscape is excluded here since
# it's a different category, already caught by the category-mismatch check
# (see test_hard_category_mismatch_is_rejected_not_just_low_confidence),
# not the subject-confusable check this eval targets.
# (post_id, wrong_image_id, why it's a meaningful negative test)
MISMATCH_TEST_CASES = [
    ("post_01", "wolf_01.jpg",  "fox post vs wolf image"),
    ("post_01", "bear_01.jpg",  "fox post vs bear image"),
    ("post_01", "deer_01.jpg",  "fox post vs deer image"),
    ("post_01", "dog_01.jpg",   "fox post vs dog image"),
    ("post_04", "bear_01.jpg",  "wolf post vs bear image"),
    ("post_04", "deer_01.jpg",  "wolf post vs deer image"),
    ("post_08", "wolf_03.jpg",  "deer post vs wolf image (original known gap, re-tested)"),
    ("post_04", "dog_01.jpg",   "wolf post vs dog image"),
    ("post_06", "deer_01.jpg",  "bear post vs deer image"),
    ("post_06", "dog_01.jpg",   "bear post vs dog image"),
    ("post_08", "dog_01.jpg",   "deer post vs dog image"),
]


def compute_top1_precision():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    post_embeddings = load_embeddings("data/post_embeddings.json")
    image_embeddings = load_embeddings("data/image_embeddings.json")

    correct = 0
    total = 0
    misses = []

    for post_id, entry in labels.items():
        if not entry["correct_images"]:
            continue
        ranked = rank_images_for_post(post_id, post_embeddings, image_embeddings, top_k=1)
        top_image_id = ranked[0][0]
        total += 1

        if top_image_id in entry["correct_images"]:
            correct += 1
        else:
            misses.append((post_id, entry["title"], top_image_id))

    precision = correct / total if total else 0.0
    print(f"Top-1 precision: {correct}/{total} = {precision:.2%}")
    if misses:
        print("Misses:")
        for post_id, title, top_image_id in misses:
            print(f"  {post_id} ({title}): top-1 was {top_image_id}, not in correct set")

    return precision


def compute_mismatch_rejection_rate():
    with open(POSTS_PATH, "r", encoding="utf-8") as f:
        posts = {p["id"]: p for p in json.load(f)}

    with open(TAGGED_IMAGES_PATH, "r", encoding="utf-8") as f:
        tagged_images = json.load(f)

    all_subjects=[data["subject"] for data in tagged_images.values()]
    post_embeddings = load_embeddings("data/post_embeddings.json")
    image_embeddings = load_embeddings("data/image_embeddings.json")

    rejected = 0
    total = len(MISMATCH_TEST_CASES)

    print("\nMismatch rejection test cases:")
    for post_id, wrong_image_id, description in MISMATCH_TEST_CASES:
        post = posts[post_id]
        candidate = ImageMetadata(**tagged_images[wrong_image_id])

        ranked = rank_images_for_post(post_id, post_embeddings, image_embeddings, top_k=len(image_embeddings))
        top_similarity = ranked[0][1]
        dynamic_threshold = compute_dynamic_threshold(top_similarity)

        similarity = None
        for image_id, score in ranked:
            if image_id == wrong_image_id:
                similarity = score
                break

        result = evaluate_match(
            post_expected_category=post["expected_category"],
            post_expected_subject=post["expected_subject"],
            candidate=candidate,
            similarity=similarity,
            similarity_threshold=dynamic_threshold,
            all_subjects=all_subjects,
        )

        status = "REJECTED" if not result.approved else "APPROVED (FAIL)"
        if not result.approved:
            rejected += 1
        print(f"  [{status}] {description}: {result.reason}")

    rate = rejected / total if total else 0.0
    print(f"\nMismatch rejection rate: {rejected}/{total} = {rate:.2%}")
    return rate


if __name__ == "__main__":
    print("=== Top-1 precision ===")
    precision = compute_top1_precision()

    print("\n=== Mismatch rejection ===")
    rejection_rate = compute_mismatch_rejection_rate()

    print(f"\nSummary: top-1 precision {precision:.2%}, mismatch rejection {rejection_rate:.2%}")