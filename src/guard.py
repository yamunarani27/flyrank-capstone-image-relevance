from enum import Enum
from dataclasses import dataclass
from src.schema import ImageMetadata, LOW_CONFIDENCE_THRESHOLD

class Decision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NO_CONFIDENT_MATCH = "no_confident_match"

SIMILARITY_THRESHOLD = 0.75
MIN_ACCEPTABLE_CONFIDENCE = LOW_CONFIDENCE_THRESHOLD

_CONFUSABLE_PAIRS: set[frozenset[str]] = {
    frozenset({"red fox", "gray wolf"}),
    frozenset({"fox", "wolf"}),
    frozenset({"dog", "wolf"}),
    frozenset({"deer", "wolf"}),
}


def _normalize_subject(subject: str) -> str:
    return subject.strip().lower()


def _is_known_confusable(expected_subject: str, candidate_subject: str) -> bool:
    a, b = _normalize_subject(expected_subject), _normalize_subject(candidate_subject)
    if a == b:
        return False
    for pair in _CONFUSABLE_PAIRS:
        x, y = tuple(pair)
        if (x in a or x in b) and (y in a or y in b):
            return True
    return False
    
@dataclass(frozen=True)
class GuardResult:
    decision: Decision
    reason: str
    similarity: float
    image_confidence: float
    category_match: bool

    @property
    def approved(self) -> bool:
        return self.decision is Decision.APPROVED

def evaluate_match(
    *,
    post_expected_category: str,
    post_expected_subject: str,
    candidate: ImageMetadata,
    similarity: float,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> GuardResult:
    category_match = candidate.category.value == post_expected_category

    if not category_match:
        return GuardResult(
            decision=Decision.REJECTED,
            reason=(
                f"Category mismatch: expected '{post_expected_category}', "
                f"detected '{candidate.category.value}' "
                f"(subject: '{candidate.subject}')."
            ),
            similarity=similarity,
            image_confidence=candidate.confidence,
            category_match=False,
        )

    if candidate.confidence < MIN_ACCEPTABLE_CONFIDENCE:
        return GuardResult(
            decision=Decision.REJECTED,
            reason=(
                f"Low classification confidence ({candidate.confidence:.2f} "
                f"< {MIN_ACCEPTABLE_CONFIDENCE:.2f}) for subject "
                f"'{candidate.subject}': the vision model itself was not "
                f"sure enough to trust this tag."
            ),
            similarity=similarity,
            image_confidence=candidate.confidence,
            category_match=True,
        )
    
    if _is_known_confusable(post_expected_subject, candidate.subject):
        return GuardResult(
            decision=Decision.REJECTED,
            reason=(
                f"Subject mismatch: expected '{post_expected_subject}', "
                f"detected '{candidate.subject}'. These subjects are "
                f"visually and semantically close but are never "
                f"interchangeable for this post."
            ),
            similarity=similarity,
            image_confidence=candidate.confidence,
            category_match=True,
        )

    if similarity < similarity_threshold:
        return GuardResult(
            decision=Decision.NO_CONFIDENT_MATCH,
            reason=(
                f"No confident match: similarity {similarity:.2f} is below "
                f"threshold {SIMILARITY_THRESHOLD:.2f}. Closest candidate "
                f"was '{candidate.subject}', which does not clearly match "
                f"the post topic '{post_expected_subject}'."
            ),
            similarity=similarity,
            image_confidence=candidate.confidence,
            category_match=True,
        )

    return GuardResult(
        decision=Decision.APPROVED,
        reason=(
            f"Approved: '{candidate.subject}' matches category "
            f"'{post_expected_category}' with similarity {similarity:.2f} "
            f"and confidence {candidate.confidence:.2f}."
        ),
        similarity=similarity,
        image_confidence=candidate.confidence,
        category_match=True,
    )