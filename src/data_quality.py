import json
from pathlib import Path
import pandas as pd


def count_missing(fields):
    return {k: int(v) for k, v in fields.items()}


def main():
    out = {}

    # MCL-WiC crosslingual
    base = Path("datasets/mcl_wic/data/MCL-WiC/test/crosslingual")
    gold_base = Path("datasets/mcl_wic/data_gold")
    for path in sorted(base.glob("*.data")):
        data = json.loads(path.read_text())
        gold_path = gold_base / path.name.replace(".data", ".gold")
        gold = json.loads(gold_path.read_text()) if gold_path.exists() else []
        missing_counts = {
            "lemma": sum(1 for x in data if not x.get("lemma")),
            "sentence1": sum(1 for x in data if not x.get("sentence1")),
            "sentence2": sum(1 for x in data if not x.get("sentence2")),
            "ranges1": sum(1 for x in data if not x.get("ranges1")),
            "ranges2": sum(1 for x in data if not x.get("ranges2")),
        }
        out[path.name] = {
            "num_examples": len(data),
            "num_gold": len(gold),
            "missing": count_missing(missing_counts),
        }

    # MCL-WiC training (for polysemy lexicon)
    train_data = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.data")
    train_gold = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.gold")
    if train_data.exists():
        data = json.loads(train_data.read_text())
        gold = json.loads(train_gold.read_text()) if train_gold.exists() else []
        out["training.en-en.data"] = {
            "num_examples": len(data),
            "num_gold": len(gold),
        }

    # MUSE dictionary sizes (subset used by experiment)
    dict_dir = Path("datasets/muse_dictionaries/dictionaries")
    muse_stats = {}
    for fname in ["en-fr.0-5000.txt", "en-ru.0-5000.txt", "en-zh.0-5000.txt", "en-ar.0-5000.txt"]:
        path = dict_dir / fname
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            muse_stats[fname] = count
    out["muse_dictionaries"] = muse_stats

    Path("results").mkdir(exist_ok=True)
    with open("results/data_quality.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
