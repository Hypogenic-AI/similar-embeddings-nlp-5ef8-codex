## 1. Executive Summary
We evaluated whether multilingual embeddings align meanings across languages and whether polysemy weakens that alignment using MCL-WiC cross-lingual data and MUSE dictionaries with XLM-R embeddings. Translation pairs showed much higher static similarity than mismatched pairs, and contextual similarity was slightly higher for same-sense (T) than different-sense (F) pairs across languages, though the effect was small. Practically, multilingual contextual embeddings appear to preserve shared senses, but polysemy only weakly modulates similarity in this setup.

## 2. Goal
We tested the hypothesis that words with similar meanings across languages have similar embeddings in multilingual models, and that polysemy reduces similarity when only one sense aligns. This matters for cross-lingual retrieval, bilingual lexicon induction, and semantic search. The expected impact is clearer guidance on when multilingual embeddings can be trusted for sense-level matching.

## 3. Data Construction

### Dataset Description
- MCL-WiC (SemEval-2021 Task 2): cross-lingual word-in-context disambiguation data for en-ar, en-fr, en-ru, en-zh. Each example provides two sentences in different languages with target word spans and a gold label (T/F) for same sense.
- MUSE Dictionaries: bilingual word pairs for en-fr, en-ru, en-zh, en-ar used for static similarity baselines.

### Example Samples
MCL-WiC (cross-lingual) example format:
```text
id: test.en-fr.0
lemma: gently
sentence1: ... treated more gently in the 2008 Chairman's working paper.
sentence2: Pendant cette décennie, ... augmenté modérément ...
ranges1: 116-122
ranges2: 73-83
label: T/F
```

### Data Quality
From `results/data_quality.json`:
- Cross-lingual test sets: 1000 examples each for en-ar, en-fr, en-ru, en-zh; 0 missing spans or sentences.
- Training en-en set: 8000 examples (used for polysemy lexicon).
- MUSE dictionary files sizes: en-fr 10872, en-ru 10887, en-zh 8728, en-ar 11571.

### Preprocessing Steps
1. Parsed MCL-WiC JSON files and joined gold labels.
2. Converted `ranges1/ranges2` (multi-span possible) to a single span by taking min start / max end.
3. For embeddings, tokenized sentences and pooled target-span tokens via offset mappings.
4. For static embeddings, averaged XLM-R input embeddings for the wordpiece tokens.

### Train/Val/Test Splits
- Polysemy lexicon derived from MCL-WiC training `en-en` set.
- Threshold selection on MCL-WiC dev `en-en` (multilingual) set.
- Main evaluation on cross-lingual test sets (en-ar, en-fr, en-ru, en-zh).

## 4. Experiment Description

### Methodology
#### High-Level Approach
We compared static vs contextual similarities in a multilingual encoder (XLM-R). Static similarity used input embedding averages for MUSE translation pairs. Contextual similarity used target-span embeddings from MCL-WiC and compared T vs F labels.

#### Why This Method?
It directly aligns with the hypothesis: if embeddings capture shared meaning, translation pairs and same-sense pairs should be more similar than mismatched or different-sense pairs. XLM-R is a widely used multilingual model with strong cross-lingual alignment.

### Implementation Details
#### Tools and Libraries
- torch 2.10.0+cu128
- transformers 5.1.0
- datasets 4.5.0
- numpy 2.4.2, pandas 3.0.0, scipy 1.17.0, seaborn 0.13.2

#### Algorithms/Models
- Model: `xlm-roberta-base`
- Contextual embeddings: last hidden state, mean pooled over target-span tokens
- Static embeddings: mean of input embedding vectors for wordpiece tokens

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---|---|
| max_length | 128 | default (covers WiC sentence lengths) |
| batch_size | 64 | GPU memory-based (24GB) |
| similarity | cosine | standard metric |
| threshold grid | [-1..1], step 0.025 | dev set search |

#### Training Procedure or Analysis Pipeline
1. Load datasets and gold labels.
2. Build polysemy lexicon from training (lemmas with both T and F labels).
3. Compute static similarities for translation and mismatched pairs.
4. Compute contextual similarities for cross-lingual WiC pairs.
5. Fit threshold on dev en-en and evaluate on cross-lingual test.
6. Run statistical tests and generate plots.

### Experimental Protocol
#### Reproducibility Information
- Runs: 1 (deterministic inference)
- Seed: 42
- Hardware: NVIDIA GeForce RTX 3090 (24GB)
- Batch size: 64
- Mixed precision: not used (inference-only)

#### Evaluation Metrics
- Cosine similarity: measures embedding alignment
- WiC accuracy (thresholded similarity): interpretable baseline for sense matching
- Statistical tests: Welch t-test and Mann–Whitney U on T vs F similarities

### Raw Results
#### Tables
Static similarity (MUSE dictionaries, 4 languages, 2k pairs each):
| Pair Type | Count | Mean | Std |
|---|---:|---:|---:|
| translation | 8000 | 0.5131 | 0.1646 |
| mismatch | 8000 | 0.3747 | 0.0937 |

Contextual similarity (MCL-WiC cross-lingual test):
| Label | Count | Mean | Std |
|---|---:|---:|---:|
| T (same sense) | 2000 | 0.9847 | 0.0051 |
| F (different sense) | 2000 | 0.9822 | 0.0059 |

#### Visualizations
- `figures/contextual_similarity_hist.png`
- `figures/contextual_similarity_box.png`
- `figures/contextual_similarity_polysemy.png`
- `figures/static_similarity_box.png`

#### Output Locations
- Results JSON: `results/metrics.json`
- Plots: `figures/`
- Tables: `results/*.csv`

## 5. Result Analysis

### Key Findings
1. Translation pairs are substantially more similar than mismatched pairs in static embeddings (mean 0.513 vs 0.375; Cohen’s d ≈ 1.03).
2. Contextual similarity is higher for same-sense (T) than different-sense (F) pairs, but the effect is small (mean 0.9847 vs 0.9822; Cohen’s d ≈ 0.46; p < 1e-46).
3. Polysemy classification from MCL-WiC training shows minimal effect on static similarity (d ≈ -0.03), and a small effect on contextual similarity (polysemous slightly higher than monosemous).

### Hypothesis Testing Results
- H1 (translation pairs higher similarity): Supported.
- H2 (same-sense contexts higher similarity): Supported with small effect size.
- H3 (polysemy reduces similarity in static embeddings): Not supported in this setup.

Statistical tests (contextual T vs F):
- Welch t-test: t = 14.57, p = 7.56e-47
- Mann–Whitney U: U = 2,541,088.5, p = 1.15e-49

### Comparison to Baselines
- Static baseline clearly separates translation vs mismatched pairs.
- Thresholded WiC accuracy (dev-trained threshold) on cross-lingual test: 0.53 (slightly above chance).

### Surprises and Insights
- Contextual similarity values are extremely high for both T and F, suggesting that raw cosine similarity may be too coarse for sense discrimination without additional normalization.

### Error Analysis
- Many F examples still have high cosine similarity, indicating overlap in contextual representations even when senses differ.
- This may be due to averaging over wordpiece spans and the overall sentence context dominating the target token representation.

### Limitations
- Static embeddings derived from XLM-R input embeddings, not standalone monolingual vectors (e.g., fastText).
- Polysemy is approximated by presence of both T and F labels in MCL-WiC training, which is incomplete.
- Span handling merges multiple ranges into a single contiguous span, which may blur multiword targets.
- No fine-tuning or alignment-specific training was performed.

## 6. Conclusions
Multilingual embeddings in XLM-R show strong alignment for translation pairs and slightly higher similarity for cross-lingual same-sense pairs, indicating that shared meaning is reflected in embeddings. However, polysemy does not strongly reduce similarity in static embeddings and the contextual effect size is modest. This suggests that raw cosine similarity is informative but insufficient for robust sense discrimination without additional modeling.

### Implications
- Cross-lingual semantic retrieval can rely on multilingual embeddings for broad meaning alignment.
- Sense-level distinction likely requires more than simple cosine similarity (e.g., probing, alignment, or contrastive objectives).

### Confidence in Findings
Moderate. The experiments are consistent across four language pairs, but the polysemy proxy and static baseline are limited. Stronger evidence would come from sense-annotated lexicons and multi-model comparisons.

## 7. Next Steps
### Immediate Follow-ups
1. Replace static baseline with fastText or aligned MUSE vectors to separate model effects from token-embedding artifacts.
2. Evaluate with XL-WiC to test consistency across a different benchmark.

### Alternative Approaches
- Use contrastive probing on contextual embeddings to improve sense separability.

### Broader Extensions
- Extend to more language families and scripts.

### Open Questions
- How do alignment-aware fine-tuned models (e.g., multilingual sentence encoders) change sense-level similarity?

## References
- MCL-WiC (SemEval-2021 Task 2)
- XL-WiC
- MUSE dictionaries
- XLM-R (Conneau et al., 2019)
