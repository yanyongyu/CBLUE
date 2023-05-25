import json
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("result_dir")


def convert(result_dir: str):
    dir = Path(result_dir)

    # convert CMeEE v1 to v2
    file = dir / "CMeEE_test.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    for sample in data:
        for entity in sample["entities"]:
            entity["end_idx"] = entity["end_idx"] + 1

    new_file = dir / "CMeEE-V2_test.json"
    new_file.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # convert CMeIE v1 to v2
    file = dir / "CMeIE_test.jsonl"
    new_file = dir / "CMeIE-V2_test.jsonl"
    new_file.write_text(file.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    args = parser.parse_args()
    convert(**vars(args))
