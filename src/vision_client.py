from google import genai
from pathlib import Path
from src.config import GEMINI_API_KEY
from src.cost_tracker import CostLog
from src.schema import ImageMetadata, VisionModelError
import time
from google.genai import errors as genai_errors

VISION_MODEL_NAME = "gemini-3.6-flash"

_client = genai.Client(api_key=GEMINI_API_KEY)
cost_log = CostLog()

def _load_image_part(image_path: Path):
    image_bytes = image_path.read_bytes()
    mime_type = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return {"inline_data": {"mime_type": mime_type, "data": image_bytes}}

def tag_image(image_path: Path) -> ImageMetadata:
    image_part = _load_image_part(image_path)

    response = _client.models.generate_content(
        model=VISION_MODEL_NAME,
        contents=[
            "Identify the single main subject of this image. Return its "
            "specific subject (e.g. 'red fox', not just 'animal'), a broad "
            "category, a short list of notable visual attributes, a one "
            "sentence caption, and your confidence (0 to 1) in this "
            "classification.",
            image_part,
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": ImageMetadata,
        },
    )

    metadata = response.parsed

    if not isinstance(metadata, ImageMetadata):
        raise VisionModelError(
            f"Vision model returned unparseable or wrong-typed output: {metadata!r}"
        )

    usage = response.usage_metadata
    if usage is not None:
        output_tokens = usage.candidates_token_count + (usage.thoughts_token_count or 0)
        cost_log.record(
            call_type="vision",
            target=image_path.name,
            input_tokens=usage.prompt_token_count,
            output_tokens=output_tokens,
        )

    return metadata.mark_confidence()

def tag_image_with_retry(image_path: Path, max_retries: int = 3) -> ImageMetadata:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return tag_image(image_path)
        except VisionModelError as e:
            last_error = e
            print(f"[retry {attempt}/{max_retries}] {image_path.name}: schema validation failed: {e}")
        except genai_errors.ServerError as e:
            last_error = e
            print(f"[retry {attempt}/{max_retries}] {image_path.name}: server error: {e}")
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                last_error = e
                print(f"[retry {attempt}/{max_retries}] {image_path.name}: rate limited, waiting 45s...")
                time.sleep(45)
                continue
            raise

        if attempt < max_retries:
            time.sleep(2 ** attempt)  # 2s, 4s, 8s...

    raise VisionModelError(
        f"Failed to tag {image_path.name} after {max_retries} attempts: {last_error}"
    )
