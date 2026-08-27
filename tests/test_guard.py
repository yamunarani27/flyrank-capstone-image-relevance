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