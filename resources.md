# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 9

| Title | Authors | Year | File | Key Info |
|---|---|---|---|---|
| Word Translation Without Parallel Data | Conneau et al. | 2017 | papers/2017_conneau_word_translation_without_parallel_data.pdf | Unsupervised/supervised alignment (MUSE) |
| A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings | Artetxe et al. | 2018 | papers/2018_artetxe_robust_self_learning.pdf | VecMap baseline |
| Cross-lingual alignment of contextual word embeddings | Schuster et al. | 2019 | papers/2019_schuster_cross_lingual_alignment_contextual_embeddings.pdf | Contextual alignment |
| Context-Aware Cross-Lingual Mapping | Aldarmaki and Diab | 2019 | papers/2019_aldarmaki_context_aware_cross_lingual_mapping.pdf | Context-aware mapping |
| Cross-lingual Language Model Pretraining (XLM) | Lample and Conneau | 2019 | papers/2019_lample_conneau_xlm.pdf | Multilingual pretraining |
| Unsupervised Cross-lingual Representation Learning at Scale (XLM-R) | Conneau et al. | 2019 | papers/2019_conneau_xlm_r.pdf | Large-scale multilingual model |
| Towards Multi-Sense Cross-Lingual Alignment of Contextual Embeddings | Liu et al. | 2022 | papers/2022_liu_multi_sense_cross_lingual_alignment.pdf | Sense-aware alignment |
| XL-WiC: A Multilingual Benchmark for Word-in-Context Disambiguation | Raganato et al. | 2020 | papers/2020_xlwic.pdf | WiC benchmark |
| SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context Disambiguation | Martelli et al. | 2021 | papers/2021_mcl_wic_task.pdf | Task definition and dataset |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| MUSE Dictionaries | MUSE (Facebook Research) | ~103 MB extracted | Bilingual lexicon induction | datasets/muse_dictionaries/ | 110 dictionaries across language pairs |
| XL-WiC | XL-WiC project site | ~32 MB extracted | Word-in-context disambiguation | datasets/xlwic/ | WordNet and Wiktionary variants |
| MCL-WiC | SapienzaNLP | ~8 MB extracted | Multilingual/cross-lingual WiC | datasets/mcl_wic/ | SemEval-2021 Task 2 |

See datasets/README.md for detailed descriptions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| MUSE | https://github.com/facebookresearch/MUSE | Unsupervised/supervised alignment | code/muse/ | Includes dictionaries and evaluation scripts |
| VecMap | https://github.com/artetxem/vecmap | Cross-lingual mapping baseline | code/vecmap/ | Strong unsupervised baseline |
| XLM | https://github.com/facebookresearch/XLM | Multilingual pretraining | code/xlm/ | Includes XLM/XLM-R references |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

Search Strategy
- Attempted paper-finder (service unavailable/timeout), then used targeted manual search for cross-lingual embedding alignment and multilingual contextual models.
- Prioritized foundational alignment methods and sense-focused benchmarks.

Selection Criteria
- Direct relevance to cross-lingual embedding similarity and polysemy.
- Widely used baselines with available code.
- Benchmarks that evaluate sense consistency across languages.

Challenges Encountered
- The full MUSE aligned vectors archive is very large (~6.7 GB). Only dictionaries were downloaded.

Gaps and Workarounds
- Did not download aligned vectors; use dictionaries with local fastText vectors or smaller language subsets if needed.

## Recommendations for Experiment Design

1. Primary dataset(s): MUSE dictionaries for translation-based similarity; XL-WiC and MCL-WiC for sense-aware evaluation.
2. Baseline methods: MUSE supervised and unsupervised alignment; VecMap unsupervised mapping; contextual embeddings from XLM/XLM-R.
3. Evaluation metrics: Translation retrieval accuracy; WiC accuracy/F1; cross-lingual similarity correlation.
4. Code to adapt/reuse: MUSE and VecMap for alignment baselines; XLM for contextual embedding extraction.

## Research Execution Notes (2026-02-15)
- Implemented experiments in `src/run_experiments.py` using XLM-R embeddings.
- Data quality checks: `src/data_quality.py` (outputs in `results/data_quality.json`).
- Analysis summaries: `src/analyze_results.py` (tables in `results/*.csv`).
- Plots saved in `figures/` and metrics in `results/metrics.json`.
- Full report: `REPORT.md`.
