import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.paths import DATASET_DIR, REPORTS_DIR, SUPPORTED_EXTENSIONS, TARGET_FOLDERS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_dataset(dataset_dir: Path, include_hashes: bool = True) -> dict:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    folders = defaultdict(lambda: {"count": 0, "bytes": 0, "dimensions": defaultdict(int)})
    extensions = defaultdict(int)
    errors = []
    hashes = defaultdict(list)
    image_count = 0

    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image_count += 1
        folder = path.parent.name
        extension = path.suffix.lower()
        extensions[extension] += 1
        folders[folder]["count"] += 1
        folders[folder]["bytes"] += path.stat().st_size
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions = f"{image.width}x{image.height}"
            folders[folder]["dimensions"][dimensions] += 1
        except (OSError, UnidentifiedImageError) as error:
            errors.append({"file": str(path.relative_to(dataset_dir)), "error": str(error)})
            continue

        if include_hashes:
            hashes[sha256_file(path)].append(str(path.relative_to(dataset_dir)))

    if image_count == 0:
        raise ValueError(f"No supported images found in {dataset_dir}")

    duplicates = [
        {"sha256": digest, "files": paths}
        for digest, paths in hashes.items()
        if len(paths) > 1
    ]
    folder_report = {
        name: {
            "count": data["count"],
            "size_mb": round(data["bytes"] / (1024 * 1024), 2),
            "dimensions": dict(sorted(data["dimensions"].items())),
            "target_note_folder": name in TARGET_FOLDERS,
        }
        for name, data in sorted(folders.items())
    }

    return {
        "dataset": str(dataset_dir.resolve()),
        "image_count": image_count,
        "extensions": dict(sorted(extensions.items())),
        "folders": folder_report,
        "corrupt_images": errors,
        "corrupt_image_count": len(errors),
        "exact_duplicates": duplicates if include_hashes else None,
        "exact_duplicate_group_count": len(duplicates) if include_hashes else None,
        "hashes_calculated": include_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the supplied image dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip exact duplicate checks.")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "dataset_analysis.json")
    args = parser.parse_args()

    report = analyze_dataset(args.dataset, include_hashes=not args.skip_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
