## Motivation & Novelty Assessment

### Why This Research Matters
Multilingual models are widely used for cross-lingual retrieval, translation, and multilingual applications, yet it is unclear how well they preserve meaning similarity across languages when words are polysemous. Understanding whether embeddings align on shared senses (vs. averaging across senses) directly impacts bilingual lexicon induction, semantic search, and cross-lingual transfer quality.

### Gap in Existing Work
Prior work focuses on cross-lingual alignment and evaluates on translation or WiC-style tasks separately, but there is limited direct analysis of how polysemy affects cross-lingual embedding similarity within the same multilingual model under controlled conditions.

### Our Novel Contribution
We test whether cross-lingual embedding similarity persists for polysemous words when only one sense aligns across languages, contrasting static alignment baselines with multilingual contextual embeddings using sense-aware benchmarks.

### Experiment Justification
- Experiment 1: Static embedding alignment (MUSE/VecMap) on bilingual dictionaries to establish baseline cross-lingual similarity for monosemous vs. polysemous words.
- Experiment 2: Contextual embedding similarity and WiC-style evaluation (XL-WiC, MCL-WiC) to test whether shared senses yield higher similarity in multilingual contextual models.
- Experiment 3: Sense-conditional similarity analysis by grouping polysemous words by sense match vs. mismatch to quantify the effect of polysemy on alignment.

## Research Question
Do words with similar meanings in different languages have similar embeddings in multilingual models, and how does polysemy affect cross-lingual embedding similarity when only one sense aligns?

## Background and Motivation
Cross-lingual embedding similarity underpins bilingual lexicon induction and multilingual transfer. While multilingual pretraining improves alignment, polysemy may blur similarity if embeddings aggregate multiple senses. This research aims to quantify that effect using standard dictionaries and sense-aware benchmarks.

## Hypothesis Decomposition
- H1: For monosemous translation pairs, multilingual embeddings are more similar than for unrelated pairs.
- H2: For polysemous words where one sense aligns across languages, contextual embeddings in matching contexts yield higher similarity than in mismatched contexts.
- H3: Static alignment methods show reduced similarity for polysemous pairs compared to monosemous pairs.

## Proposed Methodology

### Approach
Combine static alignment baselines (MUSE/VecMap) with multilingual contextual embeddings (XLM-R) and evaluate similarity under controlled sense conditions using XL-WiC/MCL-WiC.

### Experimental Steps
1. Load MUSE dictionaries and identify monosemous vs. polysemous candidates (by WordNet/Wiktionary metadata from XL-WiC/MCL-WiC).
2. Compute static embedding similarities and translation retrieval accuracy for monosemous vs. polysemous subsets.
3. Extract contextual embeddings from a multilingual transformer (e.g., XLM-R) for XL-WiC/MCL-WiC contexts; compute cosine similarity across languages for matched vs. mismatched senses.
4. Compare similarity distributions and WiC accuracy between conditions; perform statistical tests and effect sizes.

### Baselines
- Static alignment: VecMap and/or MUSE bilingual alignment.
- Contextual: Multilingual transformer embeddings without additional alignment.

### Evaluation Metrics
- Cosine similarity distributions between cross-lingual word pairs.
- Translation retrieval accuracy (top-1/top-5) on MUSE dictionaries.
- WiC accuracy/F1 on XL-WiC and MCL-WiC.

### Statistical Analysis Plan
- Two-sample t-test or Mann–Whitney U for similarity distributions.
- McNemar’s test for paired WiC predictions across conditions.
- Effect sizes (Cohen’s d) and 95% confidence intervals.
- Significance threshold α = 0.05 with FDR correction for multiple comparisons.

## Expected Outcomes
Support for the hypothesis would be higher similarity for monosemous pairs and for polysemous pairs only in sense-matched contexts. Static embeddings are expected to show weaker discrimination between sense-match and sense-mismatch.

## Timeline and Milestones
- Phase 2 (Setup/EDA): 1–2 hours
- Phase 3 (Implementation): 2–3 hours
- Phase 4 (Experiments): 2–3 hours
- Phase 5 (Analysis): 1–2 hours
- Phase 6 (Documentation): 1 hour

## Potential Challenges
- Sense annotation coverage across languages may be limited.
- GPU memory constraints for large batch embedding extraction.
- Aligning wordpieces to target tokens in contextual models.

## Success Criteria
- Completed experiments with reproducible scripts.
- Statistically significant differences between sense-matched and mismatched conditions (or well-justified null results).
- Comprehensive REPORT.md with plots, tables, and error analysis.
