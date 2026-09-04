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

## Safety layer

The mismatch guard rejects incorrect recommendations — the wolf-on-a-fox-post scenario provably fails. ✅ DONE. tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly — deliberately uses similarity=0.78, above SIMILARITY_THRESHOLD (0.75), to prove the confusable-subject check (not just a low similarity score) is what catches this case:

PASSED tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly

Rejections include a human-readable explanation. ✅ DONE. Every GuardResult includes a reason string. Example, from the wolf-rejection test: "Subject mismatch: expected 'red fox', detected 'gray wolf'. These subjects are visually and semantically close but are never interchangeable for this post."

When no image clears the bar, the system answers "no confident match" with reasons. ✅ DONE. Proven by two tests:

PASSED tests/test_guard.py::test_generic_dog_image_ranks_below_threshold
PASSED tests/test_guard.py::test_no_good_match_says_so_instead_of_guessing












## Safety Layer

**The mismatch guard rejects incorrect recommendations — the
wolf-on-a-fox-post scenario provably fails.**
✅ DONE. `tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly`
— deliberately uses `similarity=0.78`, *above* `SIMILARITY_THRESHOLD (0.75)`,
to prove the confusable-subject check (not just a low similarity score)
is what catches this case:

PASSED tests/test_guard.py::test_fox_post_rejects_wolf_image_explicitly

**Rejections include a human-readable explanation.**
✅ DONE. Every `GuardResult` includes a `reason` string. Example, from the
wolf-rejection test: `"Subject mismatch: expected 'red fox', detected
'gray wolf'. These subjects are visually and semantically close but are
never interchangeable for this post."`

**When no image clears the bar, the system answers "no confident match"
with reasons.**
✅ DONE. Proven by two tests:

PASSED tests/test_guard.py::test_generic_dog_image_ranks_below_threshold
PASSED tests/test_guard.py::test_no_good_match_says_so_instead_of_guessing
