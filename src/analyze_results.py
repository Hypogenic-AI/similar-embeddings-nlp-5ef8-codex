import json
from pathlib import Path
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


def main():
    static_df = pd.read_csv("results/static_similarity.csv")
    context_df = pd.read_csv("results/contextual_similarity.csv")

    # Load polysemy map from training data
    import json as _json
    train_data = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.data")
    train_gold = Path("datasets/mcl_wic/data/MCL-WiC/training/training.en-en.gold")
    data = _json.loads(train_data.read_text())
    gold = {item["id"]: item["tag"] for item in _json.loads(train_gold.read_text())}
    lemma_labels = {}
    for item in data:
        tag = gold.get(item["id"])
        if tag is None:
            continue
        lemma_labels.setdefault(item["lemma"], set()).add(tag)
    poly_map = {lemma: ("polysemous" if len(labels) > 1 else "monosemous") for lemma, labels in lemma_labels.items()}

    context_df["polysemy"] = context_df["lemma"].map(poly_map).fillna("unknown")

    # Summary tables
    summary = {
        "static_pair_type": static_df.groupby("pair_type")["cosine"].agg(["count", "mean", "std"]).reset_index(),
        "static_polysemy": static_df.groupby("polysemy")["cosine"].agg(["count", "mean", "std"]).reset_index(),
        "context_label": context_df.groupby("label")["cosine"].agg(["count", "mean", "std"]).reset_index(),
        "context_polysemy": context_df.groupby("polysemy")["cosine"].agg(["count", "mean", "std"]).reset_index(),
        "context_lang_label": context_df.groupby(["lang_pair", "label"])["cosine"].agg(["count", "mean"]).reset_index(),
    }

    # Statistical tests
    t_vals = context_df[context_df["label"] == "T"]["cosine"]
    f_vals = context_df[context_df["label"] == "F"]["cosine"]
    ttest = ttest_ind(t_vals, f_vals, equal_var=False)
    mwu = mannwhitneyu(t_vals, f_vals, alternative="two-sided")

    stats = {
        "context_ttest": {"t": float(ttest.statistic), "p": float(ttest.pvalue)},
        "context_mwu": {"u": float(mwu.statistic), "p": float(mwu.pvalue)},
    }

    # Save tables
    Path("results").mkdir(exist_ok=True)
    for name, df in summary.items():
        df.to_csv(f"results/{name}.csv", index=False)

    with open("results/analysis_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Update contextual with polysemy for convenience
    context_df.to_csv("results/contextual_similarity_with_polysemy.csv", index=False)


if __name__ == "__main__":
    main()
