# Outline: Cross-Lingual Embedding Similarity and Polysemy

## Title
- Emphasize cross-lingual alignment and weak polysemy effect in XLM-R

## Abstract
- Problem: cross-lingual meaning alignment and polysemy
- Gap: limited analysis of polysemy impact on similarity
- Approach: static (MUSE) vs contextual (MCL-WiC) similarities with XLM-R
- Results: translation pairs higher (0.513 vs 0.375), T>F contextual (0.9847 vs 0.9822), small effect sizes
- Significance: cosine similarity aligns meaning broadly but is weak for sense separation

## Introduction
- Hook: cross-lingual retrieval depends on embedding alignment
- Importance: multilingual models power search and transfer
- Gap: polysemy effect on cross-lingual similarity under controlled sense conditions
- Approach: compare static and contextual similarity using MUSE + MCL-WiC, XLM-R
- Quantitative preview: translation d≈1.03; contextual d≈0.46; WiC accuracy 0.53
- Contributions (3-4 bullets)

## Related Work
- Static alignment (MUSE, VecMap)
- Contextual alignment (Schuster, Aldarmaki & Diab)
- Multilingual pretraining (XLM, XLM-R)
- Sense-aware evaluation (XL-WiC, MCL-WiC, multi-sense alignment)
- Positioning: measurement-focused analysis of polysemy effects

## Methodology
- Problem formulation: cosine similarity for translation vs mismatch; T vs F contexts
- Datasets: MUSE, MCL-WiC (en-ar, en-fr, en-ru, en-zh)
- Preprocessing: span resolution, offset mapping pooling
- Embeddings: XLM-R last hidden mean; static from input embeddings
- Metrics: cosine, WiC accuracy, Welch t-test, Mann–Whitney U, Cohen's d
- Experimental protocol: dev threshold on en-en; test on cross-lingual

## Results
- Table: static similarity translation vs mismatch
- Table: contextual similarity T vs F
- Figures: static box, contextual hist/box, polysemy plot
- Stats: t=14.57, p=7.56e-47; U=2,541,088.5, p=1.15e-49
- Baseline WiC accuracy 0.53

## Discussion
- Interpretation: strong alignment for translations; weak sense separation
- Error analysis: high cosine for F due to context pooling
- Limitations: static embeddings from XLM-R input, polysemy proxy, merged spans, no fine-tuning
- Implications: need contrastive/sense-aware methods

## Conclusion
- Summarize findings and contributions
- Future work: fastText/MUSE vectors, XL-WiC, contrastive probing

## Figures/Tables
- figures: contextual_similarity_hist.png, contextual_similarity_box.png, contextual_similarity_polysemy.png, static_similarity_box.png
- tables: static_similarity.tex, contextual_similarity.tex
