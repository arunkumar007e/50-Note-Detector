import argparse
from pathlib import Path

import cv2

from src.paths import ANNOTATIONS_DIR, DATASET_DIR, SUPPORTED_EXTENSIONS, TARGET_FOLDERS


def label_path_for(image_path: Path, dataset_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_dir)
    return labels_dir / relative.parent / f"{image_path.stem}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw one bounding box around each new INR 50 note in the supplied photos."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--labels", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing labels.")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset}")

    images = sorted(
        path
        for folder in sorted(TARGET_FOLDERS)
        for path in (args.dataset / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise ValueError("No images were found in INR50_Front or INR50_Back.")

    cv2.namedWindow("Label the new INR 50 note", cv2.WINDOW_NORMAL)
    try:
        for index, image_path in enumerate(images, start=1):
            output_path = label_path_for(image_path, args.dataset, args.labels)
            if output_path.exists() and not args.overwrite:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"OpenCV could not read image: {image_path}")

            print(f"[{index}/{len(images)}] Draw a tight box around the whole note; press Enter.")
            print("Press Esc to leave this image unlabeled; Ctrl+C stops the session.")
            image_height, image_width = image.shape[:2]
            display_scale = min(1200 / image_width, 800 / image_height, 1.0)
            display_image = cv2.resize(
                image,
                (round(image_width * display_scale), round(image_height * display_scale)),
                interpolation=cv2.INTER_AREA,
            )
            x, y, width, height = cv2.selectROI(
                "Label the new INR 50 note",
                display_image,
                showCrosshair=True,
                fromCenter=False,
            )
            if width <= 0 or height <= 0:
                continue

            x = max(0, min(image_width - 1, round(x / display_scale)))
            y = max(0, min(image_height - 1, round(y / display_scale)))
            width = max(1, min(image_width - x, round(width / display_scale)))
            height = max(1, min(image_height - y, round(height / display_scale)))
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            normalized_width = width / image_width
            normalized_height = height / image_height
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                f"0 {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}\n",
                encoding="utf-8",
            )
    finally:
        cv2.destroyAllWindows()

    labeled_count = sum(
        1 for image_path in images
        if label_path_for(image_path, args.dataset, args.labels).is_file()
    )
    print(f"Labels present: {labeled_count}/{len(images)}")
    print(f"Label review directory: {args.labels}")
    if labeled_count != len(images):
        print("Skipped images remain unlabeled; dataset preparation will require every target label.")


if __name__ == "__main__":
    main()
