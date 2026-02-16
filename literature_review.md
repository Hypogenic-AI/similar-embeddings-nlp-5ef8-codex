# Literature Review

## Research Area Overview
This project studies whether words with similar meanings across languages have similar embeddings, especially in multilingual models and in the presence of polysemy. The literature spans cross-lingual word embedding alignment (static embeddings) and multilingual contextual representation learning, plus sense-aware benchmarks (WiC variants) that explicitly test meaning consistency across languages.

## Key Papers

### Word Translation Without Parallel Data (Conneau et al., 2017)
- Authors: Conneau et al.
- Year: 2017
- Source: arXiv:1710.04087
- Key Contribution: Unsupervised and supervised alignment methods (MUSE) for mapping monolingual embeddings into a shared space.
- Methodology: Adversarial training + Procrustes refinement; dictionary induction with CSLS.
- Datasets Used: MUSE bilingual dictionaries; fastText monolingual embeddings.
- Results: Strong bilingual lexicon induction without parallel data.
- Code Available: Yes (MUSE).
- Relevance to Our Research: Provides baseline for measuring cross-lingual similarity of word embeddings.

### A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings (Artetxe et al., 2018)
- Authors: Artetxe et al.
- Year: 2018
- Source: ACL 2018 (P18-1073)
- Key Contribution: Self-learning framework for unsupervised cross-lingual mapping (VecMap).
- Methodology: Iterative refinement with dictionary induction, normalization, and CSLS.
- Datasets Used: Bilingual dictionaries and word translation benchmarks.
- Results: Competitive unsupervised alignment across language pairs.
- Code Available: Yes (VecMap).
- Relevance to Our Research: Strong baseline for cross-lingual alignment of static embeddings.

### Cross-lingual alignment of contextual word embeddings (Schuster et al., 2019)
- Authors: Schuster et al.
- Year: 2019
- Source: arXiv:1902.09492
- Key Contribution: Aligns contextual embeddings across languages; demonstrates zero-shot dependency parsing gains.
- Methodology: Learn a linear mapping between contextual embedding spaces using parallel data.
- Datasets Used: Parallel corpora and UD parsing benchmarks.
- Results: Improved cross-lingual transfer with aligned contextual representations.
- Code Available: Not primary; references standard toolchains.
- Relevance to Our Research: Directly addresses cross-lingual similarity of contextual embeddings.

### Context-Aware Cross-Lingual Mapping (Aldarmaki and Diab, 2019)
- Authors: Aldarmaki and Diab
- Year: 2019
- Source: arXiv:1903.03243
- Key Contribution: Context-aware mapping to address sense variation in cross-lingual embeddings.
- Methodology: Learn mappings using contextualized representations; evaluate alignment quality.
- Datasets Used: Cross-lingual lexical tasks and contextual datasets.
- Results: Improved mapping when context is considered.
- Code Available: Not central; paper focuses on approach.
- Relevance to Our Research: Aligns with hypothesis about polysemy and embedding similarity.

### Cross-lingual Language Model Pretraining (Lample and Conneau, 2019)
- Authors: Lample and Conneau
- Year: 2019
- Source: arXiv:1901.07291
- Key Contribution: XLM introduces MLM and TLM objectives for cross-lingual pretraining.
- Methodology: Transformer pretraining with multilingual corpora; use of parallel data for TLM.
- Datasets Used: Wikipedia, parallel corpora; evaluation on XNLI and MT.
- Results: Strong cross-lingual transfer; improved multilingual representations.
- Code Available: Yes (XLM).
- Relevance to Our Research: Provides multilingual contextual embeddings for similarity tests.

### Unsupervised Cross-lingual Representation Learning at Scale (Conneau et al., 2019)
- Authors: Conneau et al.
- Year: 2019
- Source: arXiv:1911.02116
- Key Contribution: XLM-R scales multilingual pretraining without parallel data.
- Methodology: Large-scale masked LM on multilingual corpora.
- Datasets Used: CommonCrawl corpora; evaluation on XNLI, MLQA, etc.
- Results: State-of-the-art multilingual performance.
- Code Available: Yes (via XLM repo / later Hugging Face models).
- Relevance to Our Research: Key multilingual model family for embedding similarity analysis.

### Towards Multi-Sense Cross-Lingual Alignment of Contextual Embeddings (Liu et al., 2022)
- Authors: Liu et al.
- Year: 2022
- Source: COLING 2022
- Key Contribution: Explicit multi-sense alignment for contextual embeddings across languages.
- Methodology: Sense-aware alignment objectives; evaluation on sense disambiguation tasks.
- Datasets Used: Sense-aware benchmarks (WiC-style datasets).
- Results: Better alignment for polysemous words.
- Code Available: Not central; paper details method.
- Relevance to Our Research: Directly investigates polysemy and cross-lingual embedding similarity.

### XL-WiC: A Multilingual Benchmark for Word-in-Context Disambiguation (Raganato et al., 2020)
- Authors: Raganato et al.
- Year: 2020
- Source: arXiv:2010.06478
- Key Contribution: Introduces XL-WiC benchmark spanning multiple languages.
- Methodology: Word-in-context classification using multilingual datasets.
- Datasets Used: XL-WiC dataset (WordNet and Wiktionary variants).
- Results: Establishes baseline performance across languages.
- Code Available: Yes (dataset and scorer).
- Relevance to Our Research: Provides evaluation for cross-lingual sense consistency.

### SemEval-2021 Task 2: MCL-WiC (Martelli et al., 2021)
- Authors: Martelli et al.
- Year: 2021
- Source: SemEval 2021
- Key Contribution: Defines multilingual and cross-lingual WiC task with shared benchmarks.
- Methodology: Task setup and evaluation for multilingual/cross-lingual sense disambiguation.
- Datasets Used: MCL-WiC dataset.
- Results: Task baselines and shared-task outcomes.
- Code Available: Dataset in task repo.
- Relevance to Our Research: Primary dataset for cross-lingual sense alignment evaluation.

## Common Methodologies
- Linear mapping of embedding spaces using Procrustes alignment and CSLS retrieval.
- Self-learning / iterative dictionary induction for unsupervised alignment.
- Contextual embedding alignment using parallel data or shared multilingual pretraining.
- Sense-aware evaluation with WiC-style binary classification tasks.

## Standard Baselines
- Procrustes alignment with bilingual dictionaries (supervised MUSE / VecMap).
- Unsupervised alignment with adversarial initialization + CSLS refinement.
- Multilingual contextual models (XLM, XLM-R) without explicit alignment.

## Evaluation Metrics
- Bilingual lexicon induction accuracy (top-1/top-5 translation retrieval).
- Cross-lingual word similarity (Spearman correlation on word pair similarity tasks).
- WiC accuracy or F1 on sense disambiguation tasks.

## Datasets in the Literature
- MUSE bilingual dictionaries for word translation and mapping evaluation.
- XL-WiC for multilingual word-in-context disambiguation.
- MCL-WiC for multilingual and cross-lingual sense disambiguation.

## Gaps and Opportunities
- Limited analysis of how polysemy affects alignment quality across languages in multilingual models.
- Few direct comparisons between static alignment methods and contextual multilingual models on identical sense-focused tasks.

## Recommendations for Our Experiment
- Recommended datasets: MUSE dictionaries for translation-based similarity; XL-WiC and MCL-WiC for sense-sensitive evaluation.
- Recommended baselines: MUSE supervised and unsupervised alignment; VecMap unsupervised mapping; XLM/XLM-R contextual embeddings.
- Recommended metrics: Translation retrieval accuracy; WiC accuracy; cross-lingual similarity correlations.
- Methodological considerations: Control for polysemy by grouping by sense; compare static vs contextual embeddings under identical evaluation protocol.
