from pathlib import Path
from typing import Tuple

import hydra
from datasets import load_dataset
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Mapping from dataset label to our folder name
LABEL_TO_CLASS = {
    "0AiArtData": "ai",
    "1RealData": "real",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_indices(
    n: int, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[list[int], list[int], list[int]]:
    """Return indices for train/val/test splits."""
    from sklearn.model_selection import train_test_split

    all_indices = list(range(n))
    train_size = train_ratio
    remaining_ratio = 1.0 - train_ratio
    if remaining_ratio != 0:
        val_size = val_ratio / remaining_ratio
        train_idx, temp_idx = train_test_split(
            all_indices, train_size=train_size, shuffle=True, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, shuffle=True, random_state=42
        )
    else:
        train_idx = all_indices
        val_idx, test_idx = [], []
    return train_idx, val_idx, test_idx


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "conf/download"),
    config_name="default",
)
def download_data(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset_name
    root = Path(PROJECT_ROOT / cfg.output_root)
    print(f"Downloading dataset '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name, split="train")

    n = len(ds)
    print(
        f"Loaded {n} samples. Creating train/val/test splits ({cfg.train_ratio}/{cfg.val_ratio})..."
    )
    train_idx, val_idx, test_idx = split_indices(n, cfg.train_ratio, cfg.val_ratio)

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
    download_data()
