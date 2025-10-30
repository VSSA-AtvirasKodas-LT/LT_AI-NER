from jsonl_converter import convert
from glob import glob
from pathlib import Path
import os


def main():
    os.makedirs("data/conll_train", exist_ok=True)
    os.makedirs("data/conll_test", exist_ok=True)

    for file in glob("data/lt_train/*.jsonl"):
        p = Path(file)
        convert(file, f"data/conll_train/{p.stem}.conll")

    for file in glob("data/lt_test/*.jsonl"):
        p = Path(file)
        convert(file, f"data/conll_test/{p.stem}.conll")


if __name__ == "__main__":
    print("Keičiami jsonl failai į conll failus...")
    main()