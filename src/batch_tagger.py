from pathlib import Path
from dataclasses import dataclass
from src.vision_client import tag_image_with_retry, cost_log
from src.schema import ImageMetadata
import time,json
from src.db import log_cost

def discover_images(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )

@dataclass
class TaggingResult:
    image_path: Path
    metadata: ImageMetadata | None
    error: str | None

def run_batch_tagging(image_dir: Path, results_path: Path = Path("data/tagged_images.json")) -> list[TaggingResult]:
    images = discover_images(image_dir)
    completed = load_completed_results(results_path)

    results: list[TaggingResult] = []

    for i, image_path in enumerate(images, start=1):
        if image_path.name in completed:
            print(f"[{i}/{len(images)}] SKIPPING {image_path.name} (already tagged)")
            existing = ImageMetadata(**completed[image_path.name])
            results.append(TaggingResult(image_path=image_path, metadata=existing, error=None))
            continue

        print(f"[{i}/{len(images)}] Tagging {image_path.name}...")
        try:
            metadata = tag_image_with_retry(image_path)
            save_result(results_path, image_path.name, metadata)
            results.append(TaggingResult(image_path=image_path, metadata=metadata, error=None))

            latest_entry = cost_log.entries[-1]
            log_cost(
                call_type=latest_entry.call_type,
                reference=latest_entry.target,
                input_tokens=latest_entry.input_tokens,
                output_tokens=latest_entry.output_tokens,
                cost_usd=latest_entry.cost_usd,
            )
        except Exception as e:
            print(f"  SKIPPED {image_path.name}: {e}")
            results.append(TaggingResult(image_path=image_path, metadata=None, error=str(e)))

        time.sleep(3.5)

    return results

def load_completed_results(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return json.load(f)


def save_result(results_path: Path, image_name: str, metadata: ImageMetadata) -> None:
    results = load_completed_results(results_path)
    results[image_name] = metadata.model_dump()
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_batch_tagging(Path("data/images"))