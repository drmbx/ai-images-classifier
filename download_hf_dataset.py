import os
from pathlib import Path
from typing import Tuple

from datasets import load_dataset


DATASET_NAME = "Hemg/AI-Generated-vs-Real-Images-Datasets"

# Mapping from dataset label to our folder name
LABEL_TO_CLASS = {
    "0AiArtData": "ai",
    "1RealData": "real",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[list[int], list[int], list[int]]:
    """Return indices for train/val/test splits."""
    from sklearn.model_selection import train_test_split

    all_indices = list(range(n))
    train_size = train_ratio
    remaining_ratio = 1.0 - train_ratio
    val_size = val_ratio / remaining_ratio

    train_idx, temp_idx = train_test_split(all_indices, train_size=train_size, shuffle=True, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_size, shuffle=True, random_state=42)
    return train_idx, val_idx, test_idx


def main() -> None:
    root = Path("data")
    print(f"Downloading dataset '{DATASET_NAME}' from Hugging Face...")
    ds = load_dataset(DATASET_NAME, split="train")

    n = len(ds)
    print(f"Loaded {n} samples. Creating train/val/test splits (70/15/15)...")
    train_idx, val_idx, test_idx = split_indices(n)

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    for split_name, indices in splits.items():
        for class_name in ("ai", "real"):
            ensure_dir(root / split_name / class_name)

        print(f"Saving {len(indices)} samples to '{split_name}'...")
        for i, idx in enumerate(indices):
            sample = ds[idx]
            image = sample["image"]
            label = sample["label"]

            # Приводим к RGB, чтобы корректно сохранять как JPEG
            if image.mode != "RGB":
                image = image.convert("RGB")

            if isinstance(label, int):
                # If labels are integers, map 0/1 to ai/real by hand
                class_name = "ai" if label == 0 else "real"
            else:
                class_name = LABEL_TO_CLASS.get(label)
                if class_name is None:
                    raise ValueError(f"Unknown label value: {label!r}")

            out_dir = root / split_name / class_name
            out_path = out_dir / f"{idx}.jpg"
            image.save(out_path)

            if (i + 1) % 1000 == 0:
                print(f"  Saved {i + 1}/{len(indices)} images for split '{split_name}'")

    print("Done. Dataset saved under 'data/train|val|test/ai|real'.")


if __name__ == "__main__":
    main()


