# Design Doc — AI Image Understanding & Content Matching Engine

## Problem

Given a small library of images and a set of blog posts, automatically tag each image by what it actually depicts, and recommend the best-matching image for each post based on meaning — not filenames or keywords. Critically,the system must know when it doesn't have a good match and say so, rather than guess. Wrong recommendations are worse than no recommendation.

## Non-goal

This is not a general-purpose image search engine — it does not aim to scale beyond a small, bounded corpus (~50 images) or support arbitrary open-ended queries. No UI is built; the review workflow is API endpoints only. Comparing multiple vision or embedding models is out of scope — one of each is enough.

## Image metadata schema

```json
{
    "subject": "red fox",
    "category": "animal",
    "attributes": ["orange fur", "wild", "forest"],
    "caption": "A red fox standing in a forest",
    "confidence": 0.94,
    "low_confidence": false
}
```
Validated via Pydantic (`src/schema.py::ImageMetadata`) — invalid vision responses are never trusted; low-confidence tags are flagged, not discarded.

## Matching strategy

Image captions and post text are embedded independently via `sentence-transformers` (`all-MiniLM-L6-v2`, local, no API cost) and ranked by cosine similarity. The acceptance threshold is computed dynamically per post (top similarity − a margin), not a fixed global value — testing showed real similarity scores vary too widely by post writing style (behavioral prose scores far lower than visually descriptive prose) for one fixed number to work across all posts.

## Guard rules

Evaluated in order, first match wins:
1. Corpus-existence check — if the post's expected subject has zero representation anywhere in the tagged corpus, reject outright (similarity is meaningless when nothing of that subject exists at all).
2. Category mismatch — reject if categories don't match.
3. Low confidence — reject if the vision model's own confidence is below threshold.
4. Confusable subjects — reject if the pairing is a known visually-similar-but-wrong species pair (`_CONFUSABLE_PAIRS`: fox/wolf, dog/wolf, deer/wolf).
5. Similarity threshold — "no confident match" if below the post's dynamic threshold.
6. Otherwise, approved.

## Database design

Postgres (`db/schema.sql`): `images`, `image_embeddings`, `posts`, `post_embeddings`, `suggestions`, `cost_log`, `batch_jobs` — indexed on category, subject, post/image foreign keys, and review status.

## Dataset

42 images across 6 categories (fox, wolf, bear, deer, dog, landscape), Unsplash/Pexels license-free, committed directly to the repo.
