# flyrank-capstone-imagerelevance


# AI Image Understanding & Content Matching Engine

A system that looks at an image library, understands what's in each image, tags it, and matches each image to the blog post it actually belongs to — based on meaning, not filenames or keywords. A post about red foxes gets the red-fox photo. A similar-looking wolf photo is refused, with a stated reason. If nothing in the library is a good enough match, the system says so instead of guessing.

Built as a FlyRank Internship backend-track capstone.

## Why this exists

The hard part of a system like this isn't finding a plausible image — it's knowing when the best available candidate still isn't good enough, and explaining why. This project's mismatch guard combines category checks, a curated confusable-subject list, confidence scores, and similarity thresholds to make that call, and a labeled evaluation set measures whether it actually works.

## Architecture

```
Images ─(batch job, retries, cost-tracked)─► Vision Model (Gemini) ─► {tags, caption, confidence}
   │                                                                        │
   │                                                          embed(caption) via sentence-transformers
   │                                                                        │
   ▼                                                                        ▼
Postgres: images, image_embeddings                          Postgres: posts, post_embeddings
   │                                                                        │
   └───────────────────────────┬────────────────────────────────────────────┘
                                ▼
                    GET /posts/{id}/images
                                │
                    Cosine Similarity Ranking
                                │
                    Mismatch Guard (category + confusable-pairs +
                    corpus-existence + confidence + dynamic threshold)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                    ▼
    Suggested image (ranked,               "No confident match"
       explained, persisted)                  + explanation
                                │
                    Review API: approve / reject
```

Two parallel pipelines (vision tagging, embeddings) feed a shared Postgres schema; every candidate passes through the guard before a human ever sees it.

## Tech stack

- **Language / framework:** Python, FastAPI
- **Vision model:** Gemini Flash (free tier)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`), run fully locally — zero API cost, no rate limits
- **Database:** PostgreSQL via Docker
- **Schema validation:** Pydantic
- **Testing:** pytest

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop (running)
- A Gemini API key ([aistudio.google.com](https://aistudio.google.com)) — free tier, no card required

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Copy `.env.example` to `.env` and fill in real values:
```bash
cp .env.example .env
```
You need a `GEMINI_API_KEY` and the `POSTGRES_*` / `DATABASE_URL` values (any local username/password works — this only runs on your machine).

### 4. Start Postgres
```bash
docker compose up -d
docker compose ps   # confirm "healthy"
```
This automatically creates all tables from `db/schema.sql` on first run. Then apply one follow-up migration (only needed once, since it was added after the container's first init):
```bash
Get-Content db/002_add_suggestions_unique.sql | docker exec -i capstone_postgres psql -U capstone -d image_matching
```

### 5. Seed the data
The image corpus (42 images, `data/images/`), tagged metadata, embeddings, and sample posts are all committed to this repo — no re-tagging needed to run the system. Load them into Postgres:
```bash
python -m src.migrate_json_to_db
python -m src.backfill_embedding_costs
```

### 6. Run the API
```bash
uvicorn src.api:app --reload --port 8000
```

### 7. Run the tests
```bash
pytest tests/ -v
```
18 tests: schema validation, mismatch rejection, confidence flagging (6), full Review API integration (10), matching-accuracy regression against the labeled eval set (2).

### 8. Run the eval script
```bash
python -m data.eval.build_labels
python -m data.eval.run_eval
```

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/posts` | List all posts |
| GET | `/posts/{external_id}/images?top_k=5` | Rank images for a post, guard-evaluate, persist as suggestions |
| GET | `/posts/{external_id}/best-match` | Single clean verdict — match found, or "no confident match" + why |
| GET | `/posts/{external_id}/force/{image_file_path}` | Force a specific image as a candidate (probe/demo tool, doesn't persist) |
| GET | `/suggestions/{id}` | Inspect one suggestion |
| GET | `/suggestions?review_status=pending` | List/filter suggestions |
| POST | `/suggestions/{id}/approve` | Approve a suggestion |
| POST | `/suggestions/{id}/reject` | Reject a suggestion |

Example — force a wolf onto a fox post:
```bash
curl "http://localhost:8000/posts/post_01/force/wolf_01.jpg"
```
```json
{"decision":"rejected","reason":"Subject mismatch: expected 'fox', detected 'grey wolf'. These subjects are visually and semantically close but are never interchangeable for this post.", ...}
```

## Data model

Postgres tables (`db/schema.sql`): `images`, `image_embeddings`, `posts`, `post_embeddings`, `suggestions`, `cost_log`, `batch_jobs` — indexed on the columns actually queried (category, subject, post/image foreign keys, review status).

## Evaluation results

Measured against a labeled ground-truth set (`data/eval/labels.json`, generated from the corpus's own filename convention):

- **Top-1 precision: 14/14 = 100%** — across all 6 real corpus categories (fox, wolf, bear, deer, dog, landscape). A 15th post ("Understanding Domestic Cat Behavior") is deliberately excluded from this figure: the corpus has zero cat images by design, so it has no possible correct answer, and scoring it would misrepresent a correct "nothing to find" outcome as a ranking failure. It's used instead to verify the no-confident-match path (see below).
- **Mismatch rejection: 11/11 = 100%** — every same-category confusable pairing in the corpus (fox/wolf/bear/deer/dog cross-checked, 10 pairs) plus the brief's original fox/wolf example, all correctly rejected using the same dynamic per-post threshold the live API actually uses.
- **Genuine no-match case:** `GET /posts/post_15/best-match` (the cat post) correctly returns `match_found: false` — closing a real gap found during testing (see Limitations).

Reproduce with `python -m data.eval.run_eval`.

## Cost tracking

Every vision API call is logged to `cost_log` with real token counts and cost (Gemini 3.6 Flash paid-tier rates, $0.75/$3.75 per 1M tokens). Embedding calls are logged at $0.00 — legitimately free, since embeddings run locally via `sentence-transformers` with no external API involved.

```
 call_type | count |   sum
-----------+-------+----------
 vision    |     1 | 0.002863
 embedding |    57 | 0.000000
```

*(Only 1 vision entry because cost persistence was added after the original 42-image batch run — see Limitations.)*

## Known limitations

These are real findings from building and testing this system, not hidden:

- **Original batch-tagging costs weren't persisted.** The cost-tracking mechanism (`CostLog`) was in-memory only during the original 42-image tagging run; nothing was written to Postgres until this gap was found and fixed later. Those original costs are unrecoverable. Fixed going forward — every new vision call now persists automatically (verified with a real test call, see BUILDLOG.md).
- **Pure similarity thresholds can't separate all cases.** Testing with a deliberately unmatchable post (no cat images in the corpus) showed that a low-scoring correct match (a real dog post, ~0.29 similarity) and a low-scoring incorrect match (the cat post against a dog image, ~0.28 similarity) can be nearly indistinguishable by cosine similarity alone, when both posts are written in similar abstract/behavioral prose. Fixed with a dedicated corpus-existence check in the guard (rejects outright if the requested subject has zero representation in the tagged corpus at all), not by tuning the threshold further — no threshold could have separated these two cases reliably.
- **`_CONFUSABLE_PAIRS` is a curated list, not exhaustive.** It currently covers fox/wolf, dog/wolf, and deer/wolf — found and added incrementally via eval testing, not derived from a general rule. A new species pair not yet tested could still slip through on similarity alone if it's not in the list and the corpus-existence check doesn't apply (i.e., the subject does exist, just as the wrong species).
- **One vision model, one embedding model.** Per the brief's realistic scope — comparing models was treated as a stretch goal, not core.
- **API integration tests run against the real dev database**, not an isolated test database — proportionate to this project's scale, but means test runs mutate real `suggestions` rows (approve/reject cycle in `test_approve_reject_workflow_persists`).
- **A vision-tagging quirk was found, not fixed:** `wolf_07.jpg` was tagged by Gemini with subject `"coyote"`, not `"wolf"`. It's still correctly rejected on similarity grounds when tested against fox/dog/bear posts, so the outcome is correct — but it's an example of the AI mislabeling data in a way that happened not to matter here.

## Project structure

```
├── docker-compose.yml
├── capstone.yaml
├── LICENSE
├── requirements.txt
├── .env.example
├── db/
│   ├── schema.sql
│   └── 002_add_suggestions_unique.sql
├── docs/
│   └── design_doc.md
├── data/
│   ├── images/               (42-image corpus)
│   ├── posts.json            (15 sample posts)
│   ├── tagged_images.json
│   ├── image_embeddings.json
│   ├── post_embeddings.json
│   └── eval/
│       ├── build_labels.py
│       ├── labels.json
│       └── run_eval.py
├── src/
│   ├── api.py                (Review API)
│   ├── config.py
│   ├── db.py                 (Postgres connection + cost logging)
│   ├── batch_tagger.py       (vision batch job with retries)
│   ├── vision_client.py
│   ├── cost_tracker.py
│   ├── embeddings.py         (shared get_embedding())
│   ├── image_embeddings.py
│   ├── post_embeddings.py
│   ├── run_matching.py       (console ranking/guard runner)
│   ├── similarity.py         (cosine similarity, dynamic threshold)
│   ├── guard.py              (mismatch guard)
│   ├── schema.py             (ImageMetadata, Pydantic validation)
│   ├── migrate_json_to_db.py
│   └── backfill_embedding_costs.py
├── tests/
│   ├── test_guard.py
│   ├── test_api.py
│   └── test_matching_accuracy.py
├── BUILDLOG.md
├── EVIDENCE.md
└── README.md
```

## AI usage

This project was built with AI assistance throughout — see `BUILDLOG.md` for an honest, incremental record of where it helped, where it was wrong, and what changed as a result. See `EVIDENCE.md` for pasted proof against every Definition-of-Done checkbox.

## License

MIT — see `LICENSE`.