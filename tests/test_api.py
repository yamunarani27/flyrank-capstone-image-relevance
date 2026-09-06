"""
Integration tests for the Review API. These run against the real dev
Postgres database (docker compose up -d + migration must already have
been run) — not an isolated test DB. Acceptable at this project's scale;
documented here rather than hidden. Tests assume the standard 14-post,
42-image, post_15-cat-post dataset already exists (see
src/migrate_json_to_db.py).
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_list_posts_returns_all_posts():
    response = client.get("/posts")
    assert response.status_code == 200
    posts = response.json()
    assert len(posts) == 15  # 14 real + post_15 (intentional no-match)
    assert any(p["external_id"] == "post_01" for p in posts)


def test_list_posts_unknown_returns_404():
    response = client.get("/posts/post_99/images")
    assert response.status_code == 404


def test_fox_post_ranks_fox_images_first():
    response = client.get("/posts/post_01/images")
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 5
    assert all(r["image_file_path"].startswith("fox_") for r in results)
    assert all(r["decision"] == "approved" for r in results)


def test_force_wolf_on_fox_post_is_rejected():
    response = client.get("/posts/post_01/force/wolf_01.jpg")
    assert response.status_code == 200
    result = response.json()
    assert result["decision"] == "rejected"
    assert "subject mismatch" in result["reason"].lower() or "wolf" in result["reason"].lower()


def test_force_unknown_image_returns_404():
    response = client.get("/posts/post_01/force/nonexistent.jpg")
    assert response.status_code == 404


def test_best_match_for_real_post_finds_a_match():
    response = client.get("/posts/post_01/best-match")
    assert response.status_code == 200
    result = response.json()
    assert result["match_found"] is True


def test_best_match_for_unmatchable_post_reports_no_confidence():
    response = client.get("/posts/post_15/best-match")
    assert response.status_code == 200
    result = response.json()
    assert result["match_found"] is False
    assert "no confident match" in result["message"].lower()
    assert "cat" in result["reason"].lower()


def test_approve_reject_workflow_persists():
    # Rank a post first, so at least one real suggestion row exists
    ranked = client.get("/posts/post_02/images").json()
    suggestion_id = ranked[0]["id"]

    approve_response = client.post(f"/suggestions/{suggestion_id}/approve")
    assert approve_response.status_code == 200
    assert approve_response.json()["review_status"] == "approved"

    fetch_response = client.get(f"/suggestions/{suggestion_id}")
    assert fetch_response.json()["review_status"] == "approved"

    reject_response = client.post(f"/suggestions/{suggestion_id}/reject")
    assert reject_response.status_code == 200
    assert reject_response.json()["review_status"] == "rejected"


def test_approve_unknown_suggestion_returns_404():
    response = client.post("/suggestions/999999/approve")
    assert response.status_code == 404


def test_list_suggestions_filter_by_review_status():
    response = client.get("/suggestions", params={"review_status": "rejected"})
    assert response.status_code == 200
    results = response.json()
    assert all(r["review_status"] == "rejected" for r in results)