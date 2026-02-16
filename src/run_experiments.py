import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )

@dataclass
class MCLWICSample:
    sample_id: str
    lemma: str
    pos: str
    sentence1: str
    sentence2: str
    start1: int
    end1: int
    start2: int
    end2: int
    label: str  # "T" or "F"


def _parse_range(range_str: str) -> Tuple[int, int]:
    # Expected format: "start-end" or multiple spans "start-end,start-end"
    spans = []
    for chunk in range_str.split(","):
        parts = chunk.split("-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected range format: {range_str}")
        spans.append((int(parts[0]), int(parts[1])))
    starts = [s for s, _ in spans]
    ends = [e for _, e in spans]
    return min(starts), max(ends)


def load_mcl_wic_pair(data_path: Path, gold_path: Path) -> List[MCLWICSample]:
    data = json.loads(data_path.read_text())
    gold = {item["id"]: item["tag"] for item in json.loads(gold_path.read_text())}
    samples = []
    for item in data:
        label = gold.get(item["id"], None)
        if label is None:
            continue

        # Handle both multilingual (start1/end1) and cross-lingual (ranges1/ranges2) formats
        if "start1" in item and "end1" in item:
            start1, end1 = int(item["start1"]), int(item["end1"])
            start2, end2 = int(item["start2"]), int(item["end2"])
        elif "ranges1" in item and "ranges2" in item:
            start1, end1 = _parse_range(item["ranges1"])
            start2, end2 = _parse_range(item["ranges2"])
        else:
            raise KeyError("Missing span fields in MCL-WiC item")

        samples.append(
            MCLWICSample(
                sample_id=item["id"],
                lemma=item["lemma"],
                pos=item["pos"],
                sentence1=item["sentence1"],
                sentence2=item["sentence2"],
                start1=start1,
                end1=end1,
                start2=start2,
                end2=end2,
                label=label,
            )
        )
    return samples


def load_mcl_wic_training(data_path: Path, gold_path: Path) -> List[MCLWICSample]:
    return load_mcl_wic_pair(data_path, gold_path)


def build_polysemy_lexicon(samples: Iterable[MCLWICSample]) -> Dict[str, str]:
    # Mark lemma as polysemous if it appears with both T and F labels
    lemma_labels: Dict[str, set] = {}
    for s in samples:
        lemma_labels.setdefault(s.lemma, set()).add(s.label)
    lexicon = {}
    for lemma, labels in lemma_labels.items():
        if len(labels) >= 2:
            lexicon[lemma] = "polysemous"
        else:
            lexicon[lemma] = "monosemous"
    return lexicon


def find_dictionary_file(base_dir: Path, lang_pair: str) -> Path:
    # Try exact, then subsets
    candidates = [
        base_dir / f"{lang_pair}.txt",
        base_dir / f"{lang_pair}.0-5000.txt",
        base_dir / f"{lang_pair}.5000-6500.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try reverse pair
    src, tgt = lang_pair.split("-")
    rev = f"{tgt}-{src}"
    candidates = [
        base_dir / f"{rev}.txt",
        base_dir / f"{rev}.0-5000.txt",
        base_dir / f"{rev}.5000-6500.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No dictionary file found for {lang_pair}")


def load_dictionary_pairs(path: Path, limit: int = 2000) -> List[Tuple[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
            if len(pairs) >= limit:
                break
    return pairs


def mean_pool_span(hidden_states: torch.Tensor, offsets: List[Tuple[int, int]], start: int, end: int) -> torch.Tensor:
    # Identify token indices overlapping the character span
    idxs = []
    for i, (s, e) in enumerate(offsets):
        if s == e == 0:
            continue
        if s < end and e > start:
            idxs.append(i)
    if not idxs:
        # fallback to closest token by start
        distances = [abs(s - start) for s, e in offsets]
        if distances:
            idxs = [int(np.argmin(distances))]
        else:
            idxs = [0]
    vec = hidden_states[idxs].mean(dim=0)
    return vec


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def get_static_word_embedding(word: str, tokenizer, embedding_matrix: torch.Tensor) -> torch.Tensor:
    tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    if input_ids.numel() == 0:
        return torch.zeros(embedding_matrix.shape[1])
    vecs = embedding_matrix[input_ids]
    return vecs.mean(dim=0)


def compute_static_similarity(
    model, tokenizer, pairs: List[Tuple[str, str]], polysemy_map: Dict[str, str]
) -> pd.DataFrame:
    embedding_matrix = model.get_input_embeddings().weight.detach().cpu()
    records = []
    for src, tgt in tqdm(pairs, desc="Static sim"):
        src_vec = get_static_word_embedding(src, tokenizer, embedding_matrix)
        tgt_vec = get_static_word_embedding(tgt, tokenizer, embedding_matrix)
        sim = cosine_sim(src_vec, tgt_vec)
        poly = polysemy_map.get(src, "unknown")
        records.append({"src": src, "tgt": tgt, "cosine": sim, "polysemy": poly, "pair_type": "translation"})
    # mismatched baseline by shuffling targets
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    for (src, _), (_, tgt) in tqdm(zip(pairs, shuffled), total=len(pairs), desc="Static sim (mismatch)"):
        src_vec = get_static_word_embedding(src, tokenizer, embedding_matrix)
        tgt_vec = get_static_word_embedding(tgt, tokenizer, embedding_matrix)
        sim = cosine_sim(src_vec, tgt_vec)
        poly = polysemy_map.get(src, "unknown")
        records.append({"src": src, "tgt": tgt, "cosine": sim, "polysemy": poly, "pair_type": "mismatch"})
    return pd.DataFrame(records)


def batch_contextual_embeddings(
    model,
    tokenizer,
    sentences: List[str],
    spans: List[Tuple[int, int]],
    device: torch.device,
    max_length: int = 128,
    batch_size: int = 64,
) -> List[torch.Tensor]:
    outputs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Contextual batches"):
        batch_sents = sentences[i : i + batch_size]
        batch_spans = spans[i : i + batch_size]
        enc = tokenizer(
            batch_sents,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state  # [B, T, H]
        for j in range(hidden.size(0)):
            h = hidden[j].detach().cpu()
            offs = offsets[j].tolist()
            start, end = batch_spans[j]
            vec = mean_pool_span(h, offs, start, end)
            outputs.append(vec)
    return outputs


def compute_contextual_similarity(samples: List[MCLWICSample], model, tokenizer, device, batch_size: int) -> pd.DataFrame:
    sentences1 = [s.sentence1 for s in samples]
    spans1 = [(s.start1, s.end1) for s in samples]
    sentences2 = [s.sentence2 for s in samples]
    spans2 = [(s.start2, s.end2) for s in samples]

    emb1 = batch_contextual_embeddings(model, tokenizer, sentences1, spans1, device, batch_size=batch_size)
    emb2 = batch_contextual_embeddings(model, tokenizer, sentences2, spans2, device, batch_size=batch_size)

    records = []
    for s, v1, v2 in zip(samples, emb1, emb2):
        sim = cosine_sim(v1, v2)
        records.append(
            {
                "id": s.sample_id,
                "lemma": s.lemma,
                "pos": s.pos,
                "label": s.label,
                "cosine": sim,
            }
        )
    return pd.DataFrame(records)


def stats_summary(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
    summary = {}
    for group, sub in df.groupby(group_col):
        summary[group] = {
            "count": int(len(sub)),
            "mean": float(sub[value_col].mean()),
            "std": float(sub[value_col].std()),
            "min": float(sub[value_col].min()),
            "max": float(sub[value_col].max()),
        }
    return summary


def main():
    setup_logging(Path("logs/experiment.log"))
    logging.info("Starting experiments")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Model
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Batch size based on GPU memory (24GB -> 64-128)
    batch_size = 64 if device.type == "cuda" else 16

    # Load MCL-WiC training for polysemy lexicon
    train_data = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.data")
    train_gold = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.gold")
    train_samples = load_mcl_wic_training(train_data, train_gold)
    polysemy_map = build_polysemy_lexicon(train_samples)
    logging.info(f"Polysemy lexicon size: {len(polysemy_map)}")

    # Static similarity from MUSE dictionaries
    dict_dir = Path("datasets/muse_dictionaries/dictionaries")
    lang_pairs = ["en-fr", "en-ru", "en-zh", "en-ar"]
    static_frames = []
    for lp in lang_pairs:
        try:
            dict_path = find_dictionary_file(dict_dir, lp)
        except FileNotFoundError:
            logging.warning(f"Skipping {lp}: dictionary not found")
            continue
        pairs = load_dictionary_pairs(dict_path, limit=2000)
        df = compute_static_similarity(model, tokenizer, pairs, polysemy_map)
        df["lang_pair"] = lp
        static_frames.append(df)
    static_df = pd.concat(static_frames, ignore_index=True)
    static_df.to_csv("results/static_similarity.csv", index=False)

    # Contextual similarity on MCL-WiC cross-lingual test sets
    cross_base = Path("datasets/mcl_wic/data/MCL-WiC/test/crosslingual")
    gold_base = Path("datasets/mcl_wic/data_gold")
    contextual_frames = []
    for lp in lang_pairs:
        data_path = cross_base / f"test.{lp}.data"
        gold_path = gold_base / f"test.{lp}.gold"
        if not data_path.exists() or not gold_path.exists():
            logging.warning(f"Skipping {lp}: data or gold missing")
            continue
        samples = load_mcl_wic_pair(data_path, gold_path)
        df = compute_contextual_similarity(samples, model, tokenizer, device, batch_size=batch_size)
        df["lang_pair"] = lp
        contextual_frames.append(df)
    contextual_df = pd.concat(contextual_frames, ignore_index=True)
    contextual_df.to_csv("results/contextual_similarity.csv", index=False)

    # Threshold-based WiC prediction (simple baseline): optimize on dev.en-en
    dev_data = Path("datasets/mcl_wic/data/MCL-WiC/dev/multilingual/dev.en-en.data")
    dev_gold = Path("datasets/mcl_wic/data/MCL-WiC/dev/multilingual/dev.en-en.gold")
    if dev_data.exists() and dev_gold.exists():
        dev_samples = load_mcl_wic_pair(dev_data, dev_gold)
        dev_df = compute_contextual_similarity(dev_samples, model, tokenizer, device, batch_size=batch_size)
        # Search threshold on dev
        best_thr, best_acc = 0.0, 0.0
        for thr in np.linspace(-1, 1, 81):
            preds = ["T" if c >= thr else "F" for c in dev_df["cosine"]]
            acc = accuracy_score(dev_df["label"], preds)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
    else:
        best_thr, best_acc = 0.0, None

    # Apply threshold to cross-lingual
    contextual_df["pred"] = ["T" if c >= best_thr else "F" for c in contextual_df["cosine"]]
    overall_acc = accuracy_score(contextual_df["label"], contextual_df["pred"])

    # Polysemy grouping for contextual results
    contextual_df["polysemy"] = contextual_df["lemma"].map(polysemy_map).fillna("unknown")

    # Statistical tests
    stats = {
        "static_similarity": stats_summary(static_df, "pair_type", "cosine"),
        "contextual_similarity": stats_summary(contextual_df, "label", "cosine"),
        "polysemy_contextual": stats_summary(contextual_df, "polysemy", "cosine"),
    }

    # t-test and Mann–Whitney for contextual T vs F
    t_vals = contextual_df[contextual_df["label"] == "T"]["cosine"]
    f_vals = contextual_df[contextual_df["label"] == "F"]["cosine"]
    ttest = ttest_ind(t_vals, f_vals, equal_var=False)
    mwu = mannwhitneyu(t_vals, f_vals, alternative="two-sided")
    stats["contextual_ttest"] = {"t": float(ttest.statistic), "p": float(ttest.pvalue)}
    stats["contextual_mwu"] = {"u": float(mwu.statistic), "p": float(mwu.pvalue)}

    # Static similarity polysemy comparison
    poly_static = static_df[static_df["polysemy"] == "polysemous"]["cosine"]
    mono_static = static_df[static_df["polysemy"] == "monosemous"]["cosine"]
    if len(poly_static) > 0 and len(mono_static) > 0:
        ttest_static = ttest_ind(poly_static, mono_static, equal_var=False)
        stats["static_polysemy_ttest"] = {"t": float(ttest_static.statistic), "p": float(ttest_static.pvalue)}

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "dev_threshold": best_thr,
        "dev_accuracy": best_acc,
        "crosslingual_accuracy": float(overall_acc),
        "stats": stats,
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Basic plots
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 4))
    sns.histplot(data=contextual_df, x="cosine", hue="label", bins=40, stat="density", common_norm=False)
    plt.title("Contextual cosine similarity by label (T vs F)")
    plt.tight_layout()
    plt.savefig("figures/contextual_similarity_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=contextual_df, x="label", y="cosine")
    plt.title("Contextual similarity by label")
    plt.tight_layout()
    plt.savefig("figures/contextual_similarity_box.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=static_df, x="pair_type", y="cosine")
    plt.title("Static similarity: translation vs mismatch")
    plt.tight_layout()
    plt.savefig("figures/static_similarity_box.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=contextual_df, x="polysemy", y="cosine")
    plt.title("Contextual similarity by polysemy class")
    plt.tight_layout()
    plt.savefig("figures/contextual_similarity_polysemy.png", dpi=200)
    plt.close()

    logging.info("Experiments completed")


if __name__ == "__main__":
    main()
