import json
import random
import shutil
from pathlib import Path
from itertools import chain
from collections import Counter
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets"

ORIGIN_DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets_origin"
FEW_SHOT_DATASET_DIR = PROJECT_ROOT / "CBLUEDatasets_few_shot"

FEW_SHOT_SAMPLES = 100
FEW_SHOT_CATEGORY_KEY = {
    "CMeEE": "entities.*.type",
    "CMeIE": "spo_list.*.predicate",
    "CHIP-CTC": "label",
    "CHIP-STS": "label",
    "CHIP-CDN": "normalized_result",
    "KUAKE-QIC": "label",
    "KUAKE-QQR": "label",
    "KUAKE-QTR": "label",
}


def init():
    if ORIGIN_DATASET_DIR.exists() or FEW_SHOT_DATASET_DIR.exists():
        print("Dataset already exists. Skip init.")
        return

    # generate few-shot dataset
    print("Generating few-shot dataset...")
    generate_dataset()


def _load_data(dir: Path, task: str) -> List[Any]:
    dataset_dir = dir / task
    if task == "CMeIE":
        train_file = dataset_dir / f"{task}_train.jsonl"
        return [
            json.loads(line)
            for line in train_file.read_text(encoding="utf-8").splitlines()
        ]
    else:
        train_file = dataset_dir / f"{task}_train.json"
        return json.loads(train_file.read_text(encoding="utf-8"))


def _save_data(dir: Path, task: str, data: List[Any]):
    dataset_dir = dir / task
    if task == "CMeIE":
        train_file = dataset_dir / f"{task}_train.jsonl"
        train_file.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in data),
            encoding="utf-8",
        )
    else:
        train_file = dataset_dir / f"{task}_train.json"
        train_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _get_sample_category(task: str, sample: dict) -> str:
    category_key = FEW_SHOT_CATEGORY_KEY[task]
    path = category_key.split(".")

    def _iter(d: Any, path: List[str]):
        key, *path = path
        if key == "*":
            return [_iter(v, path) if path else v for v in d]
        else:
            return _iter(d[key], path) if path else d[key]

    category = _iter(sample, path)
    if isinstance(category, list):
        category = Counter(category).most_common(1)[0][0]
    return category


def generate_dataset():
    FEW_SHOT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in FEW_SHOT_CATEGORY_KEY:
        origin_dataset_dir = DATASET_DIR / dataset
        target_dataset_dir = FEW_SHOT_DATASET_DIR / dataset
        shutil.copytree(origin_dataset_dir, target_dataset_dir)

        data = _load_data(FEW_SHOT_DATASET_DIR, dataset)

        categories: Dict[str, List[Any]] = {}
        for sample in data:
            categories.setdefault(
                _get_sample_category(dataset, sample), []
            ).append(sample)

        total_categories = len(categories)
        size, remain = divmod(FEW_SHOT_SAMPLES, total_categories)
        samples = {
            category: random.sample(samples, size + (i < remain))
            for i, (category, samples) in enumerate(categories.items())
        }

        data = sorted(chain.from_iterable(samples.values()), key=data.index)
        assert len(data) == FEW_SHOT_SAMPLES

        _save_data(FEW_SHOT_DATASET_DIR, dataset, data)


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
