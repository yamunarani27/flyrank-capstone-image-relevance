from enum import Enum
from pydantic import BaseModel, Field,field_validator

class Category(str, Enum):
    ANIMAL = "animal"
    PLANT = "plant"
    LANDSCAPE = "landscape"
    OBJECT = "object"
    PEOPLE = "people"
    OTHER = "other"

LOW_CONFIDENCE_THRESHOLD = 0.60

class ImageMetadata(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    category: Category
    attributes: list[str] = Field(default_factory=list, max_length=20)
    caption: str = Field(..., min_length=1, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("subject", "caption")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank or whitespace-only")
        return v

    @field_validator("attributes")
    @classmethod
    def clean_attributes(cls, v: list[str]) -> list[str]:
        return [a.strip() for a in v if a.strip()]

    low_confidence: bool = False

    def mark_confidence(self) -> "ImageMetadata":
        self.low_confidence = self.confidence < LOW_CONFIDENCE_THRESHOLD
        return self

class VisionModelError(Exception):
    """Raised when the vision model's raw response fails schema validation."""