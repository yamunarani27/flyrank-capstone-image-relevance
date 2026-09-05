"""
Review API: rank images for a post, run them through the mismatch guard,
persist the results as suggestions, and let a human approve/reject/inspect.

Endpoints:
  GET  /posts                          — list all posts
  GET  /posts/{external_id}/images     — rank + guard-evaluate images for a post, persist as suggestions
  GET  /suggestions/{suggestion_id}    — inspect one suggestion (why it was approved/rejected)
  GET  /suggestions?review_status=...  — list suggestions, optionally filtered
  POST /suggestions/{suggestion_id}/approve
  POST /suggestions/{suggestion_id}/reject
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from src.db import get_cursor
from src.similarity import cosine_similarity, compute_dynamic_threshold
from src.guard import evaluate_match
from src.schema import ImageMetadata

app = FastAPI(title="Image-Post Matching Review API")


class SuggestionOut(BaseModel):
    id: int
    post_id: int
    image_id: int
    image_file_path: str
    similarity: float
    decision: str
    reason: str
    review_status: str


class PostOut(BaseModel):
    id: int
    external_id: str
    title: str
    expected_subject: str
    expected_category: str


@app.get("/posts", response_model=list[PostOut])
def list_posts():
    with get_cursor() as cur:
        cur.execute("SELECT id, external_id, title, expected_subject, expected_category FROM posts ORDER BY id")
        return cur.fetchall()


@app.get("/posts/{external_id}/images", response_model=list[SuggestionOut])
def get_images_for_post(external_id: str, top_k: int = 5):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.expected_category, p.expected_subject, pe.embedding
            FROM posts p
            JOIN post_embeddings pe ON pe.post_id = p.id
            WHERE p.external_id = %s
            """,
            (external_id,),
        )
        post = cur.fetchone()
        if post is None:
            raise HTTPException(status_code=404, detail=f"Post '{external_id}' not found")

        cur.execute(
            """
            SELECT i.id, i.file_path, i.subject, i.category, i.attributes, i.caption,
                   i.confidence, i.low_confidence, ie.embedding
            FROM images i
            JOIN image_embeddings ie ON ie.image_id = i.id
            """
        )
        images = cur.fetchall()
        all_subjects = [img["subject"] for img in images]

    ranked = sorted(
        (
            (img, cosine_similarity(post["embedding"], img["embedding"]))
            for img in images
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_k]

    if not ranked:
        raise HTTPException(status_code=404, detail="No images available to rank")

    dynamic_threshold = compute_dynamic_threshold(ranked[0][1])

    results = []
    with get_cursor() as cur:
        for img, similarity in ranked:
            candidate = ImageMetadata(
                subject=img["subject"],
                category=img["category"],
                attributes=img["attributes"],
                caption=img["caption"],
                confidence=img["confidence"],
            )
            guard_result = evaluate_match(
                post_expected_category=post["expected_category"],
                post_expected_subject=post["expected_subject"],
                candidate=candidate,
                similarity=similarity,
                similarity_threshold=dynamic_threshold,
                all_subjects=all_subjects,
            )

            cur.execute(
                """
                INSERT INTO suggestions (post_id, image_id, similarity, decision, reason)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (post_id, image_id)
                DO UPDATE SET similarity = EXCLUDED.similarity,
                              decision = EXCLUDED.decision,
                              reason = EXCLUDED.reason
                RETURNING id, post_id, image_id, similarity, decision, reason, review_status
                """,
                (post["id"], img["id"], similarity, guard_result.decision.value, guard_result.reason),
            )
            row = cur.fetchone()
            results.append({**row, "image_file_path": img["file_path"]})

    return results

@app.get("/posts/{external_id}/best-match")
def get_best_match(external_id: str):
    """
    Returns a single clean verdict for a post's best candidate — the
    "does this post have a good match at all" question the brief's demo
    script (Probe 4) asks for, without needing to inspect a ranked list.
    """
    results = get_images_for_post(external_id, top_k=1)
    top = results[0]

    if top["decision"] == "approved":
        return {"match_found": True, **top}

    return {
        "match_found": False,
        "message": "No confident match found.",
        "reason": top["reason"],
        "closest_candidate": top["image_file_path"],
        "similarity": top["similarity"],
    }

@app.get("/suggestions/{suggestion_id}", response_model=SuggestionOut)
def get_suggestion(suggestion_id: int):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.post_id, s.image_id, i.file_path AS image_file_path,
                   s.similarity, s.decision, s.reason, s.review_status
            FROM suggestions s
            JOIN images i ON i.id = s.image_id
            WHERE s.id = %s
            """,
            (suggestion_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")
        return row


@app.get("/suggestions", response_model=list[SuggestionOut])
def list_suggestions(review_status: Optional[str] = Query(default=None)):
    with get_cursor() as cur:
        if review_status:
            cur.execute(
                """
                SELECT s.id, s.post_id, s.image_id, i.file_path AS image_file_path,
                       s.similarity, s.decision, s.reason, s.review_status
                FROM suggestions s
                JOIN images i ON i.id = s.image_id
                WHERE s.review_status = %s
                ORDER BY s.id
                """,
                (review_status,),
            )
        else:
            cur.execute(
                """
                SELECT s.id, s.post_id, s.image_id, i.file_path AS image_file_path,
                       s.similarity, s.decision, s.reason, s.review_status
                FROM suggestions s
                JOIN images i ON i.id = s.image_id
                ORDER BY s.id
                """
            )
        return cur.fetchall()


@app.post("/suggestions/{suggestion_id}/approve", response_model=SuggestionOut)
def approve_suggestion(suggestion_id: int):
    return _set_review_status(suggestion_id, "approved")


@app.post("/suggestions/{suggestion_id}/reject", response_model=SuggestionOut)
def reject_suggestion(suggestion_id: int):
    return _set_review_status(suggestion_id, "rejected")


def _set_review_status(suggestion_id: int, status: str):
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE suggestions
            SET review_status = %s, reviewed_at = now()
            WHERE id = %s
            RETURNING id, post_id, image_id, similarity, decision, reason, review_status
            """,
            (status, suggestion_id),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

        cur.execute("SELECT file_path FROM images WHERE id = %s", (row["image_id"],))
        image = cur.fetchone()
        return {**row, "image_file_path": image["file_path"]}

@app.get("/posts/{external_id}/force/{image_file_path}")
def force_candidate(external_id: str, image_file_path: str):
    """
    Force a specific image as a candidate for a specific post, bypassing
    ranking entirely — for demos and targeted testing (e.g. "force the
    wolf as a candidate for the fox post"). Does not persist a suggestion
    row, since this is a deliberate what-if probe, not a real ranked
    recommendation.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT p.id, p.expected_category, p.expected_subject, pe.embedding
            FROM posts p
            JOIN post_embeddings pe ON pe.post_id = p.id
            WHERE p.external_id = %s
            """,
            (external_id,),
        )
        post = cur.fetchone()
        if post is None:
            raise HTTPException(status_code=404, detail=f"Post '{external_id}' not found")

        cur.execute(
            """
            SELECT i.id, i.file_path, i.subject, i.category, i.attributes, i.caption,
                   i.confidence, i.low_confidence, ie.embedding
            FROM images i
            JOIN image_embeddings ie ON ie.image_id = i.id
            WHERE i.file_path = %s
            """,
            (image_file_path,),
        )
        image = cur.fetchone()
        if image is None:
            raise HTTPException(status_code=404, detail=f"Image '{image_file_path}' not found")

        # Need the post's top similarity across all images to compute the
        # same dynamic threshold the real ranking endpoint uses — otherwise
        # this forced check would use a different (unfair) bar.
        cur.execute(
            """
            SELECT i.subject,ie.embedding
            FROM images i
            JOIN image_embeddings ie ON ie.image_id = i.id
            """
        )
        all_images = cur.fetchall()

    similarities = [cosine_similarity(post["embedding"], img["embedding"]) for img in all_images]
    all_subjects = [img["subject"] for img in all_images]
    dynamic_threshold = compute_dynamic_threshold(max(similarities))

    forced_similarity = cosine_similarity(post["embedding"], image["embedding"])

    candidate = ImageMetadata(
        subject=image["subject"],
        category=image["category"],
        attributes=image["attributes"],
        caption=image["caption"],
        confidence=image["confidence"],
    )

    result = evaluate_match(
        post_expected_category=post["expected_category"],
        post_expected_subject=post["expected_subject"],
        candidate=candidate,
        similarity=forced_similarity,
        similarity_threshold=dynamic_threshold,
        all_subjects=all_subjects,
    )

    return {
        "post_external_id": external_id,
        "forced_image": image_file_path,
        "similarity": round(forced_similarity, 4),
        "dynamic_threshold": round(dynamic_threshold, 4),
        "decision": result.decision.value,
        "reason": result.reason,
    }