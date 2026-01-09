from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@hydra.main(
    version_base=None,
    config_path=str(PROJECT_ROOT / "conf/download"),
    config_name="default",
)
def download_data(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset_name
    output_root = PROJECT_ROOT / cfg.output_root

    print(f"Downloading dataset '{dataset_name}' from Hugging Face...")
    dataset = load_dataset(dataset_name, split="train")

    (output_root / "ai").mkdir(parents=True, exist_ok=True)
    (output_root / "real").mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(dataset)} samples. Saving raw images...")

    for idx, sample in enumerate(dataset):
        image = sample["image"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        class_name = "ai" if sample["label"] == 0 else "real"
        out_path = output_root / class_name / f"{idx:06d}.jpg"
        image.save(out_path, format="JPEG", quality=100)

        if (idx + 1) % 1000 == 0:
            print(f"  Saved {idx + 1}/{len(dataset)} images")

    print(f"Done. Raw dataset saved under '{output_root}'.")


if __name__ == "__main__":
    download_data()
