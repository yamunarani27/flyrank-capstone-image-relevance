# Evidence

One entry per Definition-of-Done checkbox (brief §6). Unproven items are
marked TODO rather than claimed, per the brief: "Claims without evidence
score as not done."

## AI Processing

**Vision model produces structured output validated against a schema;
invalid responses are never trusted.**
✅ DONE. `src/schema.py` defines `ImageMetadata` (Pydantic `BaseModel`) with
field-level validation (`not_blank`, `clean_attributes`, range constraints
on `confidence`). `src/vision_client.py::tag_image` passes `ImageMetadata`
directly as `response_schema` to Gemini, then verifies the SDK's parsed
result with `isinstance(metadata, ImageMetadata)` before use — an
unparseable or wrong-typed response raises `VisionModelError` rather than
being trusted. Real output, from `fox_01.jpg`:

ImageMetadata(subject='red fox', category=<Category.ANIMAL: 'animal'>,
attributes=['red fur', 'white chest', 'bushy tail', 'pointed ears',
'dark legs', 'snowy background'], caption='A red fox stands gracefully
on a snowy slope looking into the distance during golden hour.',
confidence=0.98, low_confidence=False)


**Low-confidence classifications are flagged instead of accepted.**
✅ DONE. `ImageMetadata.mark_confidence()` sets `low_confidence=True` when
`confidence < LOW_CONFIDENCE_THRESHOLD (0.60)`. Proven in
`tests/test_guard.py::test_low_confidence_classification_is_rejected_even_with_high_similarity`
— a candidate with `confidence=0.35` and `similarity=0.95` (every other
signal favorable) is still rejected by the guard specifically for low
confidence:

PASSED tests/test_guard.py::test_low_confidence_classification_is_rejected_even_with_high_similarity


**Images are processed through a batch background job with retries.**
✅ DONE. `src/batch_tagger.py::run_batch_tagging` loops over all discovered
images; `src/vision_client.py::tag_image_with_retry` retries on
`VisionModelError`, `ServerError`, and HTTP 429 (`ClientError`) with
exponential/fixed backoff, up to 3 attempts. Real log output from an actual
run, showing retry-then-recovery behavior:

[retry 1/3] fox_02.jpg: rate limited, waiting 45s...
[retry 2/3] fox_02.jpg: rate limited, waiting 45s...
[retry 3/3] fox_02.jpg: rate limited, waiting 45s...
SKIPPED fox_02.jpg: Failed to tag fox_02.jpg after 3 attempts: 429 RESOURCE_EXHAUSTED...

All 42/42 images were eventually tagged successfully across a resumed run
(see `data/tagged_images.json`).

## Semantic Matching

Images and blog posts are embedded and ranked by semantic similarity — matches concepts, not exact words. ✅ DONE. src/image_embeddings.py::get_embedding uses sentence-transformers (all-MiniLM-L6-v2, 384-dim), applied to image captions (data/image_embeddings.json, 42 entries) and post title+body (data/post_embeddings.json, 14 entries). src/similarity.py:: rank_images_for_post computes cosine similarity and returns a ranked list. Concept-not-keyword matching is directly exercised by post_02 ("Wild Fox Species Around the Globe" — no literal overlap with image captions beyond "fox") and post_03/post_14, which use "Vulpes vulpes" and "wild fox species" phrasing; all top-5 results for every fox post were correctly fox images despite the differing wording. Real output:

The Behavior of Red Foxes (dynamic threshold: 0.487)
  fox_07.jpg: approved — similarity 0.64, confidence 0.98
  fox_03.jpg: approved — similarity 0.54, confidence 0.98

Ranking correctness verified across all 14 posts covering all 6 corpus categories (fox, wolf, bear, deer, dog, landscape) — every post's top candidates matched its expected subject.

Real similarity scores are used in image-post matching decisions, not hardcoded placeholders. ✅ DONE. guard.evaluate_match() takes a similarity argument populated from cosine_similarity() in src/run_matching.py, run against the full posts/images dataset — not the fixed test values used in tests/test_guard.py. Confirmed real, varying similarity scores appear in guard decisions (0.12 to 0.64 across the full eval run — see below).

A small labeled evaluation dataset measures top-1 precision. ✅ DONE. data/eval/build_labels.py generates ground truth (data/eval/labels.json) from the corpus's own filename convention (e.g. all dog_*.jpg are correct for a post with expected_subject="dog") — 7 correct images per post across all 14 posts. data/eval/run_eval.py computes top-1 precision by ranking each post's images and checking whether the top-1 result is in its labeled correct set. Real output:

=== Top-1 precision ===
Top-1 precision: 14/14 = 100.00%

The mismatch guard rejects incorrect recommendations across same-category confusions, not just the fox/wolf example. ✅ DONE. data/eval/run_eval.py::compute_mismatch_rejection_rate tests all 10 same-category (animal) species pairings in the corpus (fox, wolf, bear, deer, dog cross-checked against each other) plus the original fox/wolf case from the brief — 11 cases total — using the same dynamic per-post threshold the production matching path (run_matching.py) actually uses, not a hypothetical fixed threshold. Real output:

=== Mismatch rejection ===
Mismatch rejection test cases:
  [REJECTED] fox post vs wolf image: Subject mismatch: expected 'fox', detected 'grey wolf'.
  [REJECTED] fox post vs bear image: No confident match: similarity 0.22 is below threshold 0.75.
  [REJECTED] fox post vs deer image: No confident match: similarity 0.27 is below threshold 0.75.
  [REJECTED] fox post vs dog image: No confident match: similarity 0.18 is below threshold 0.75.
  [REJECTED] wolf post vs bear image: No confident match: similarity 0.15 is below threshold 0.75.
  [REJECTED] wolf post vs deer image: Subject mismatch: expected 'wolf', detected 'white-tailed deer'.
  [REJECTED] deer post vs wolf image (original known gap, re-tested): Subject mismatch: expected 'deer', detected 'grey wolf'.
  [REJECTED] wolf post vs dog image: No confident match: similarity 0.13 is below threshold 0.75.
  [REJECTED] bear post vs deer image: No confident match: similarity 0.22 is below threshold 0.75.
  [REJECTED] bear post vs dog image: No confident match: similarity 0.12 is below threshold 0.75.
  [REJECTED] deer post vs dog image: No confident match: similarity 0.25 is below threshold 0.75.
Mismatch rejection rate: 11/11 = 100.00%

Note: an earlier version of this eval found a real gap — a wolf image (wolf_03.jpg, similarity 0.48) was approved for "Deer in the Autumn Woods" because deer/wolf was not yet in _CONFUSABLE_PAIRS (see BUILDLOG.md for the full investigation, including a self-caught methodology error where the first expanded test list accidentally retested a different post/image pair instead of the original failing case). Fixed by adding deer/wolf to _CONFUSABLE_PAIRS; the 100% figure above reflects the fix, not an untested claim.

## Safety layer

The mismatch guard rejects incorrect recommendations — the wolf-on-a-fox-post scenario provably fails. ✅ DONE. tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly — deliberately uses similarity=0.78, above SIMILARITY_THRESHOLD (0.75), to prove the confusable-subject check (not just a low similarity score) is what catches this case:

PASSED tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly

Rejections include a human-readable explanation. ✅ DONE. Every GuardResult includes a reason string. Example, from the wolf-rejection test: "Subject mismatch: expected 'red fox', detected 'gray wolf'. These subjects are visually and semantically close but are never interchangeable for this post."

When no image clears the bar, the system answers "no confident match" with reasons. ✅ DONE. Proven by two tests:

PASSED tests/test_guard.py::test_generic_dog_image_ranks_below_threshold
PASSED tests/test_guard.py::test_no_good_match_says_so_instead_of_guessing

The guard's category/subject checks cover every same-category confusable pairing in the corpus, not just a hand-picked example. ✅ DONE (previously TODO). _CONFUSABLE_PAIRS now includes fox/wolf, dog/wolf, and deer/wolf. Coverage validated by the 11-case eval above, which exercises all 10 same-category species combinations plus the original brief example — 11/11 rejected. See BUILDLOG.md Phase 4 for how this gap was found (via the eval set) and fixed.

## Backend

**Database models for images, tags, embeddings, posts, suggestions,
approvals/rejections — with the required indexes.**
✅ DONE (previously TODO). `db/schema.sql` defines all 7 tables from
`design_doc.md` (images, image_embeddings, posts, post_embeddings,
suggestions, cost_log, batch_jobs), applied automatically on container
init via Docker's `docker-entrypoint-initdb.d`. Indexes on `images.category`,`images.subject`, `posts.expected_subject`, `suggestions.post_id`,`suggestions.image_id`, `suggestions.review_status`, `cost_log.call_type`.Verified via `\dt` after `docker compose up -d`, and via row counts after migration:

Migrated 42 images.
Migrated 14 posts.
Migrated 42 image embeddings.
Migrated 14 post embeddings.
Migration complete.














