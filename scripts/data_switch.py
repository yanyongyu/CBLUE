import json
import random
import shutil
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets"

ORIGIN_DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets_origin"
FEW_SHOT_DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets_few_shot"

FEW_SHOT_SAMPLES = 100


def init():
    if ORIGIN_DATASET_DIR.exists() or FEW_SHOT_DATASET_DIR.exists():
        print("Dataset already exists. Skip init.")
        return

    # generate few-shot dataset
    print("Generating few-shot dataset...")
    generate_dataset()


def generate_dataset():
    FEW_SHOT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in (
        "CMeEE",
        "CHIP-CTC",
        "CHIP-CDN",
        "CHIP-STS",
        "KUAKE-QIC",
        "KUAKE-QQR",
        "KUAKE-QTR",
    ):
        origin_dataset_dir = DATASET_DIR / dataset
        target_dataset_dir = FEW_SHOT_DATASET_DIR / dataset
        shutil.copytree(origin_dataset_dir, target_dataset_dir)

        train_file = target_dataset_dir / f"{dataset}_train.json"
        data: List[Any] = json.loads(train_file.read_text(encoding="utf-8"))
        few_shot_data = random.sample(data, FEW_SHOT_SAMPLES)
        train_file.write_text(
            json.dumps(few_shot_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for dataset in ("CMeIE",):
        origin_dataset_dir = DATASET_DIR / dataset
        target_dataset_dir = FEW_SHOT_DATASET_DIR / dataset
        shutil.copytree(origin_dataset_dir, target_dataset_dir)

        train_file = target_dataset_dir / f"{dataset}_train.jsonl"
        data = train_file.read_text(encoding="utf-8").splitlines()
        few_shot_data = random.sample(data, FEW_SHOT_SAMPLES)
        train_file.write_text("\n".join(few_shot_data), encoding="utf-8")


def switch_dataset():
    if ORIGIN_DATASET_DIR.exists() and not FEW_SHOT_DATASET_DIR.exists():
        DATASET_DIR.rename(FEW_SHOT_DATASET_DIR)
        ORIGIN_DATASET_DIR.rename(DATASET_DIR)
        print("Using origin dataset.")
    elif FEW_SHOT_DATASET_DIR.exists() and not ORIGIN_DATASET_DIR.exists():
        DATASET_DIR.rename(ORIGIN_DATASET_DIR)
        FEW_SHOT_DATASET_DIR.rename(DATASET_DIR)
        print("Using few-shot dataset.")
    else:
        raise RuntimeError("Dataset not valid!")


def main():
    init()
    switch_dataset()


if __name__ == "__main__":
    main()
