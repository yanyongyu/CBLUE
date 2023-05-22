import json
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("result-dir")


def convert(result_dir: str):
    dir = Path(result_dir)

    # convert CMeEE v1 to v2
    file = dir / "CMeEE_test.json"
    data = json.loads(file.read_text())
    for sample in data:
        for entity in sample["entities"]:
            entity["end_idx"] = entity["end_idx"] + 1
    file.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    args = parser.parse_args()
    convert(**vars(args))
