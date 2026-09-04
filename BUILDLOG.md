# Build Log

Honest record of where AI (Claude) helped, where it was wrong, and what
changed as a result. Written incrementally during the build, not
reconstructed afterward.

## Phase 1 — Design

- First commit bundled more than intended (schema, guard core, design doc,  and the full 40-image corpus all in one commit) while establishing the initial project structure. Subsequent commits are scoped more tightly to one coherent change each, per the brief's own commit guidance.

- `.gitignore` initially excluded `data/images/*` under the assumption that image files shouldn't be committed (standard practice for large/generated datasets). This was wrong for this project: the brief explicitly requires the corpus be committed directly (or a download script provided) so evaluators can reproduce results. Caught before the first real commit of images; the exclusion was removed and the corpus committed directly.

- The mismatch guard's `_CONFUSABLE_PAIRS` list (fox/wolf, dog/wolf) was a deliberate design addition beyond the three signals (category, confidence, similarity) originally sketched. Caption-embedding similarity between close relatives (two canids) can score deceptively high — high enough to pass a naive similarity threshold — so a small curated negative list was added as a fourth signal specifically to catch this. Verified this works via `test_fox_post_rejects_wolf_image_explicitly`, which deliberately uses
  a similarity score (0.78) *above* the plain threshold (0.75) to prove the confusable-pairs check, not the threshold, is what catches the wolf case.

## Phase 2 — Vision pipeline

- Initial vision client used `google-generativeai` (the older SDK pattern most tutorials still show). Corrected before writing real code: that package's Gemini Developer API support ended August 2025. Switched to the current `google-genai` SDK (`from google import genai`) before any client code was written.

- Initial model choice `gemini-2.5-flash` returned a 404 on first real API call: "no longer available to new users... use models/gemini-3.6-flash."Switched immediately; caught by actually running the code against a live key rather than assuming the model name was current.

- Cost tracking initially used placeholder pricing (Gemini 3 Flash line:$0.50/1M input, $3.00/1M output) because the official pricing page   couldn't be fetched directly in the build environment, and `3.6`-specific rates weren't independently confirmable via search at the time. I (the user) verified the actual rate directly on ai.google.dev: $0.75/1M input, $3.75/1M output through Dec 31 2026 (rising to $1.50/$7.50 on Jan 1 2027).`cost_tracker.py` was corrected to the verified figures before any real batch run.

- `requirements.txt` was missing `python-dotenv` and `google-genai` after they were `pip install`-ed directly — caught and fixed in a follow-up commit before it could cause an import error for anyone else cloning the repo and running `pip install -r requirements.txt`.

- The SDK prints a warning ("Direct use of automatic function calling
  (AFC)...") on every `generate_content` call, despite no tools/functions being passed. Cause not fully confirmed — doesn't appear to affect correctness of output or cost tracking (verified by checking actual `ImageMetadata` results and `cost_log` entries were correct on calls that printed the warning). Left as an open, acknowledged gap rather than a chased-down root cause, given time constraints.

- A Pylance type error (`Cannot access attribute "mark_confidence" for class 
"BaseModel" | "dict" | "Enum"`) appeared because the SDK types
  `response.parsed` generically. Fixed with an explicit `isinstance(metadata,ImageMetadata)` check rather than a `# type: ignore` comment — the `isinstance` check both satisfies the type checker *and* adds a real runtime guard against a genuinely possible (if unlikely) wrong-typed response, which `# type: ignore` would not.

- First full batch-tagging run had no persistence layer yet (`batch_tagger.py` only held results in an in-memory list). The run was interrupted by Gemini free-tier rate limiting partway through, and all in-progress results — roughly 11-12 successfully tagged images — were lost, since nothing had been written to disk. Persistence (`data/tagged_images.json`, written after every successful tag, with automatic skip-on-resume) was added immediately after, before attempting the batch again.

- Discovered mid-batch that the Gemini free tier's daily quota was cut from 250 to 20 requests/day/project/model as of December 2025 — confirmed via search after hitting repeated 429 `RESOURCE_EXHAUSTED` errors (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Initial retry logic treated all `ClientError`s (4xx) as non-retryable; corrected to special-case HTTP 429 specifically (wait ~45s, retry) since a rate limit is transient, unlike a genuinely broken request (bad model name, bad key).

- Observed quota behavior did not cleanly match the documented "20/day,
  resets at midnight Pacific" model — during one batch run, some requests succeeded interspersed with 429 failures, suggesting a rolling/leaky-bucket regeneration rather than a hard daily wall. This is an observation, not a confirmed mechanism — noted here as an honest uncertainty rather than stated as fact elsewhere in the project.

- To finish tagging within a reasonable timeframe, a second API key was
  created under a separate Google Cloud project on the same Google account (not a second account) — a legitimate use of per-project quota, distinct from creating multiple accounts specifically to multiply free-tier allowance, which would edge toward circumventing intended per-developer limits. All 42 images were successfully tagged across the two projects' combined quota; the key was swapped back to the original project afterward with no code changes required, since `.env` alone controls which key is active.

## Phase 3 — Matching engine

-Chose a local sentence-transformers model (all-MiniLM-L6-v2, 384-dim) over the Gemini embedding API specifically to avoid a repeat of the Phase 2 quota problem. Embeddings run locally, offline, with no per-day request cap — a meaningful difference given embeddings get regenerated any time the pipeline is re-tested.

-Considered embedding a combined string (caption + subject + category + attributes) for each image before settling on caption-only. Re-reading the brief's own fox/wolf example clarified this: the guard's "expected fox, detected wolf" rejection is a structured category comparison, not a semantic-similarity judgment — a fox caption and a wolf caption can be embedding-similar (same setting, same sentence shape) despite being a category mismatch. Blending subject/category into the embedding text would conflate the two signals the guard is meant to keep separate. Kept caption-only for embeddings; subject/category remain explicit fields compared directly in guard.evaluate_match().

-get_embedding()'s first implementation returned list(model.encode(text)) — a Python list of numpy float32 values, not native floats. This passes silently until the first json.dump() call, which raises TypeError: Object of type float32 is not JSON serializable. Fixed by using .tolist() on the numpy array instead, which recursively converts to native Python types. Caught via a manual sanity check (type(vec[0])) before running the full batch, not via the error itself.

-First version of the image-embedding loop assumed data/tagged_images.json was a list of image objects (for image in tagged_images: image.get(...)) and raised AttributeError: 'str' object has no attribute 'get' on the first run. The file is actually a dict keyed by filename. Fixed by iterating .items() instead.

-No blog post content existed yet at the start of Phase 3 — nothing in the brief or prior phases produced sample posts. Authored 14 posts (data/posts.json) spanning all corpus categories (fox, wolf, bear, deer, dog, landscape), including posts phrased with scientific names and synonyms ("Vulpes vulpes," "wild fox species") to exercise semantic rather than keyword matching, plus one post worded to specifically probe the fox/wolf mismatch case described in the brief.

-Successfully generated and persisted 42 image embeddings (data/image_embeddings.json) and 14 post embeddings (data/post_embeddings.json), each entry storing the 384-float vector alongside model_name for traceability if the embedding model changes later.

-Ran real cosine-similarity ranking (src/similarity.py) across all 14 posts before wiring anything into the guard, specifically to check whether the placeholder SIMILARITY_THRESHOLD = 0.75 was realistic. It wasn't: real scores for genuinely correct matches ranged from ~0.20 (behavior/methodology posts, e.g. "Training Tips for a Happy Dog") to ~0.64 (visually descriptive posts, e.g. fox/deer/landscape). A fixed global threshold of 0.75 would have rejected every single match in the dataset, correct or not — caught by testing against real data before wiring, not assumed from the placeholder value.

-Added a similarity_threshold parameter to evaluate_match() (default SIMILARITY_THRESHOLD, so existing tests are unaffected) and a compute_dynamic_threshold(top_similarity, margin=0.15) helper in similarity.py, which sets each post's acceptance bar relative to its own top similarity score rather than a fixed global number.

-First wiring attempt (run_matching.py) produced every result as no_confident_match, always citing "below threshold 0.75" even though the printed per-post dynamic threshold was clearly different each time (e.g. 0.15, 0.41). Root cause: the similarity < SIMILARITY_THRESHOLD comparison inside evaluate_match() (and the matching f-string) had not actually been updated to reference the new similarity_threshold parameter — the parameter existed in the signature but was never used in the body, so the function silently fell back to the old global constant. Fixed by replacing both occurrences of SIMILARITY_THRESHOLD with similarity_threshold inside the similarity-check branch.
-After the fix, the dynamic-threshold approach worked well for higher-scoring posts (fox, deer, landscape) but surfaced a real, unresolved gap for low-scoring posts: a fixed margin=0.15 pushes the effective threshold very low when the top score is already low (e.g. dog posts dropped to a ~0.14-0.15 threshold), low enough that wrong-subject animal images can clear it. Observed concretely: a wolf image (wolf_03.jpg, similarity 0.48) was approved for a deer post ("Deer in the Autumn Woods," dynamic threshold 0.427), since wolf/deer is not in _CONFUSABLE_PAIRS and nothing else in the guard catches a same-category, wrong-subject, non-listed pairing. Left as a known, documented limitation rather than patched with a guessed fix — to be addressed once a labeled eval set justifies a better margin or threshold formula (e.g. a margin that scales with top score rather than a fixed absolute value).

- Found and fixed a real bug in `_is_known_confusable()` while re-checking `guard.py` before committing: the check compared normalized subject strings for exact equality against `_CONFUSABLE_PAIRS`, but real tagged subjects are phrases like `"grey wolf"` or `"red fox"`, not the bare `"wolf"`/`"fox"` entries actually listed in the set — so `frozenset({"dog",
  "grey wolf"})` never matched `frozenset({"dog", "wolf"})`, and the dog/wolf pair silently never fired for any real wolf image. This explained why `wolf_03.jpg` and `wolf_04.jpg` had been approved for dog posts despite `dog`/`wolf` already being listed as confusable. Fixed by checking whether each word in a pair appears as a substring of either subject,
  rather than requiring exact string equality. Verified against all 6 existing guard tests (all still pass) and against a full 14-post rerun: `wolf_03.jpg`/`wolf_04.jpg` now correctly `rejected` under the dog posts.

- Confirmed one related case is still open, not fixed by the above: on the same full rerun, `wolf_03.jpg` (similarity 0.48) remains `approved` under "Deer in the Autumn Woods." This is not the same bug — `deer`/`wolf` was never added to `_CONFUSABLE_PAIRS` in the first place, so there was
  nothing for the substring fix to catch. Left open deliberately rather than guessing and adding more pairs ad hoc; whether to expand `_CONFUSABLE_PAIRS` further or address this via the threshold/margin formula is a decision for the labeled eval set, not a guess.

