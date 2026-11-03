import random
from pathlib import Path

import pytest

from src.prepare_dataset import (
    capture_group,
    create_dataset,
    read_target_label,
    split_capture_groups,
)


def test_read_target_label_rejects_missing_label(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_target_label(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "1 0.5 0.5 0.2 0.2",
        "0 1.1 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 0.05 0.5 0.2 0.2",
        "0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1",
    ],
)
def test_read_target_label_rejects_invalid_boxes(tmp_path: Path, contents: str) -> None:
    label = tmp_path / "label.txt"
    label.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_target_label(label)


def test_read_target_label_accepts_normalized_box(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    assert read_target_label(label) == "0 0.5 0.5 0.4 0.2\n"


def test_capture_group_keeps_same_minute_photos_together(tmp_path: Path) -> None:
    first = tmp_path / "INR50_Front" / "IMG20261101161201.jpg"
    second = tmp_path / "INR50_Front" / "IMG20261101161259.jpg"
    assert capture_group(first) == capture_group(second)


def test_split_capture_groups_keeps_all_same_minute_images_together(
    tmp_path: Path,
) -> None:
    images = [
        tmp_path / folder / f"IMG2026110112{minute:02d}00.jpg"
        for folder in ("INR50_Front", "INR50_Back", "INR100_Front")
        for minute in range(8)
    ]
    splits = split_capture_groups(images, random.Random(42))
    assignments = {
        capture_group(path): split_name
        for split_name, paths in splits.items()
        for path in paths
    }

    assert set(splits) == {"train", "val", "test"}
    assert len(assignments) == 8
    assert all(
        assignments[capture_group(path)] == split_name
        for split_name, paths in splits.items()
        for path in paths
    )
    assert all(
        any(path.parent.name.startswith("INR50_") for path in paths)
        for paths in splits.values()
    )


def test_create_dataset_requires_reviewed_target_boxes_before_writing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "Dataset"
    for folder, filename in (
        ("INR50_Front", "IMG20261101120100.jpg"),
        ("INR50_Back", "IMG20261101120200.jpg"),
        ("INR10_Front", "IMG20261101120300.jpg"),
    ):
        image_dir = dataset_dir / folder
        image_dir.mkdir(parents=True)
        (image_dir / filename).write_bytes(b"test fixture")

    output_dir = tmp_path / "yolo_data"

    with pytest.raises(FileNotFoundError, match="Missing reviewed"):
        create_dataset(dataset_dir, tmp_path / "annotations", output_dir)

    assert not output_dir.exists()
