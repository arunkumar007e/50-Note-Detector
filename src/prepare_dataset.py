import argparse
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.annotate import label_path_for
from src.paths import (
    ANNOTATIONS_DIR,
    DATASET_DIR,
    SUPPORTED_EXTENSIONS,
    TARGET_FOLDERS,
    YOLO_DATA_DIR,
)

CAPTURE_TIME = re.compile(r"IMG(\d{8})(\d{6})", re.IGNORECASE)


def read_target_label(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reviewed INR 50 bounding box: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one note box in {path}, found {len(lines)}")
    fields = lines[0].split()
    if len(fields) != 5 or fields[0] != "0":
        raise ValueError(f"Expected one YOLO class-0 box in {path}")
    values = [float(value) for value in fields[1:]]
    center_x, center_y, width, height = values
    if not all(0 <= value <= 1 for value in values) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid normalized YOLO coordinates in {path}")
    if center_x - width / 2 < 0 or center_x + width / 2 > 1:
        raise ValueError(f"Box extends outside the image in {path}")
    if center_y - height / 2 < 0 or center_y + height / 2 > 1:
        raise ValueError(f"Box extends outside the image in {path}")
    return lines[0] + "\n"


def capture_group(path: Path) -> str:
    match = CAPTURE_TIME.search(path.stem)
    if not match:
        return f"{path.parent.name}:{path.stem}"
    captured = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return captured.strftime("%Y%m%d%H%M")


def split_capture_groups(paths: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for path in paths:
        groups[capture_group(path)].append(path)
    if len(groups) < 3:
        raise ValueError(
            "The images contain fewer than three capture-time groups; "
            "cannot create train, validation, and test splits without leakage."
        )

    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
    folder_totals = defaultdict(int)
    for path in paths:
        folder_totals[path.parent.name] += 1
    targets = {
        folder: {split: count * fractions[split] for split in fractions}
        for folder, count in folder_totals.items()
    }
    assigned = {name: [] for name in fractions}
    assigned_by_folder = {
        folder: {split: 0 for split in fractions} for folder in folder_totals
    }

    for index, (_, group) in enumerate(group_items):
        group_counts = defaultdict(int)
        for path in group:
            group_counts[path.parent.name] += 1

        missing_splits = [split for split in fractions if not assigned[split]]
        remaining_groups = len(group_items) - index
        candidates = missing_splits if remaining_groups == len(missing_splits) else list(fractions)

        def score(split: str) -> float:
            return sum(
                min(
                    max(targets[folder][split] - assigned_by_folder[folder][split], 0),
                    count,
                )
                / max(targets[folder][split], 1)
                for folder, count in group_counts.items()
            )

        destination = max(candidates, key=lambda split: (score(split), rng.random()))
        assigned[destination].extend(group)
        for folder, count in group_counts.items():
            assigned_by_folder[folder][destination] += count

    return assigned


def create_dataset(
    dataset_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, int]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    images_by_folder = defaultdict(list)
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images_by_folder[path.parent.name].append(path)
    if not images_by_folder:
        raise ValueError(f"No supported images found in {dataset_dir}")

    for folder in TARGET_FOLDERS:
        if folder not in images_by_folder:
            raise ValueError(f"Required target folder is missing: {dataset_dir / folder}")

    rng = random.Random(seed)
    all_images = [path for paths in images_by_folder.values() for path in paths]
    splits = split_capture_groups(all_images, rng)

    labels = {}
    for image_path in all_images:
        if image_path.parent.name in TARGET_FOLDERS:
            labels[image_path] = read_target_label(
                label_path_for(image_path, dataset_dir, labels_dir)
            )

    for split_name in ("train", "val", "test"):
        if not any(path in labels for path in splits[split_name]):
            raise ValueError(f"The {split_name} split has no labeled INR 50 examples.")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output path or move the existing generated dataset first."
        )

    for split_name, paths in splits.items():
        for image_path in paths:
            filename = f"{image_path.parent.name}__{image_path.name}"
            image_destination = output_dir / "images" / split_name / filename
            label_destination = output_dir / "labels" / split_name / f"{Path(filename).stem}.txt"
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                image_destination.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, image_destination)

            label_destination.write_text(labels.get(image_path, ""), encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    root = output_dir.resolve().as_posix()
    data_yaml.write_text(
        f'path: "{root}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: note_50\n",
        encoding="utf-8",
    )
    return {name: len(paths) for name, paths in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware YOLO train/val/test data.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--labels", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=YOLO_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = create_dataset(args.dataset, args.labels, args.output, args.seed)
    print(f"Prepared dataset at {args.output.resolve()}")
    print("Image counts: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
