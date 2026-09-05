"""
Builds data/eval/labels.json — the ground-truth eval set.

Correct images for a post are those whose filename prefix matches the
post's expected_subject (e.g. expected_subject="dog" -> all dog_*.jpg
files). This is more reliable than matching against the free-text
`subject` caption field, since breed-level captions ("golden retriever
puppy", "beagle") don't literally contain the word "dog" — but the
corpus's own filename convention (dog_01.jpg, wolf_02.jpg, ...) is a
clean, unambiguous ground truth grouping.

Run once to generate; hand-edit data/eval/labels.json afterward if you
want to exclude a borderline image or add a note.
"""
import json

TAGGED_IMAGES_PATH = "data/tagged_images.json"
POSTS_PATH = "data/posts.json"
OUTPUT_PATH = "data/eval/labels.json"


def build_labels():
    with open(TAGGED_IMAGES_PATH, "r", encoding="utf-8") as f:
        tagged_images = json.load(f)

    with open(POSTS_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    labels = {}
    for post in posts:
        post_id = post["id"]
        expected_subject = post["expected_subject"].lower()

        correct_images = [
            image_id
            for image_id in tagged_images
            if image_id.lower().startswith(f"{expected_subject}_")
        ]

        labels[post_id] = {
            "title": post["title"],
            "expected_subject": post["expected_subject"],
            "correct_images": correct_images,
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"Built {OUTPUT_PATH} with {len(labels)} posts.")
    for post_id, entry in labels.items():
        note = " (intentionally no match — excluded from precision)" if not entry["correct_images"] else ""
        print(f"  {post_id}: {len(entry['correct_images'])} correct image(s) — {entry['title']}{note}")


if __name__ == "__main__":
    build_labels()