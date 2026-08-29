from src.guard import Decision, evaluate_match
from src.schema import Category, ImageMetadata


def make_image(subject, category, confidence, attributes=None, caption=None):
    return ImageMetadata(
        subject=subject,
        category=category,
        attributes=attributes or [],
        caption=caption or f"A photo of a {subject}",
        confidence=confidence,
    )

def test_fox_post_approves_fox_image():
    fox = make_image("red fox", Category.ANIMAL, confidence=0.94)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=fox,
        similarity=0.91,
    )
    assert result.decision is Decision.APPROVED
    assert result.approved

def test_fox_post_rejects_wolf_image_explicitly():
    wolf = make_image("gray wolf", Category.ANIMAL, confidence=0.90)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=wolf,
        similarity=0.78,
    )
    assert result.decision is Decision.REJECTED
    assert "Subject mismatch" in result.reason
    assert not result.approved

def test_low_confidence_classification_is_rejected_even_with_high_similarity():
    uncertain_fox = make_image("red fox", Category.ANIMAL, confidence=0.35)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=uncertain_fox,
        similarity=0.95,
    )
    assert result.decision is Decision.REJECTED
    assert "Low classification confidence" in result.reason

def test_hard_category_mismatch_is_rejected_not_just_low_confidence():
    landscape = make_image("mountain range", Category.LANDSCAPE, confidence=0.97)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=landscape,
        similarity=0.30,
    )
    assert result.decision is Decision.REJECTED
    assert "Category mismatch" in result.reason
    assert not result.category_match

def test_generic_dog_image_ranks_below_threshold():
    dog = make_image("dog", Category.ANIMAL, confidence=0.88)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=dog,
        similarity=0.40,
    )
    assert result.decision is Decision.NO_CONFIDENT_MATCH
    assert not result.approved

def test_no_good_match_says_so_instead_of_guessing():
    weak_candidate = make_image("deer", Category.ANIMAL, confidence=0.80)
    result = evaluate_match(
        post_expected_category="animal",
        post_expected_subject="red fox",
        candidate=weak_candidate,
        similarity=0.50,
    )
    assert result.decision is Decision.NO_CONFIDENT_MATCH
    assert not result.approved
    assert result.reason  # must always explain itself