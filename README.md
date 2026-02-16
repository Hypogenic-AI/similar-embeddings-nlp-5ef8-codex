# Multilingual Embedding Similarity and Polysemy

This project evaluates whether words with similar meanings across languages have similar embeddings, and how polysemy affects that similarity using MCL-WiC and MUSE datasets with XLM-R embeddings.

## Key Findings
- Translation pairs show much higher static similarity than mismatched pairs.
- Same-sense cross-lingual contexts have slightly higher similarity than different-sense contexts.
- Polysemy shows minimal effect on static similarity in this setup.

## How to Reproduce
1. Create and activate environment:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
2. Run experiments:
```bash
python src/run_experiments.py
python src/data_quality.py
python src/analyze_results.py
```

## File Structure
- `src/run_experiments.py`: main experiments and plots
- `src/data_quality.py`: data quality checks
- `src/analyze_results.py`: summary tables and stats
- `results/`: metrics and tables
- `figures/`: plots
- `REPORT.md`: full report

See `REPORT.md` for full methodology and analysis.
