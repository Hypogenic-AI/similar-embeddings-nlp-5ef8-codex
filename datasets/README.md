# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below to
reproduce the local data state.

## Dataset 1: MUSE Bilingual Dictionaries

Overview
- Source: https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
- Size (downloaded): ~59 MB compressed, ~103 MB extracted
- Format: Plain text dictionary pairs (source word, target word) per line
- Task: Bilingual lexicon induction / word translation evaluation
- Languages: Many (110 dictionaries across language pairs)
- License: See MUSE repository for details

Download Instructions
- Direct download:
  - `wget -O datasets/muse_dictionaries.tar.gz https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz`
  - `mkdir -p datasets/muse_dictionaries && tar -xzf datasets/muse_dictionaries.tar.gz -C datasets/muse_dictionaries`

Loading the Dataset
- Each dictionary file is a whitespace-separated word pair list in `datasets/muse_dictionaries/dictionaries/`.

Sample Data
- See `datasets/muse_dictionaries/samples/muse_zh_en_sample.txt`.

Notes
- The full aligned embeddings in MUSE are very large (~6.7 GB) and were not downloaded.

## Dataset 2: XL-WiC

Overview
- Source: https://pilehvar.github.io/xlwic/
- Size (downloaded): ~18 MB compressed, ~32 MB extracted
- Format: Plain text files for train/dev/test per language
- Task: Multilingual word-in-context disambiguation
- Splits: Train/valid (English) and multilingual test sets
- License: See dataset README in `datasets/xlwic/xlwic_datasets/README.txt`

Download Instructions
- Direct download:
  - `wget -O datasets/xlwic_datasets.zip https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip`
  - `mkdir -p datasets/xlwic && unzip -q datasets/xlwic_datasets.zip -d datasets/xlwic`

Loading the Dataset
- English train/valid: `datasets/xlwic/xlwic_datasets/wic_english/`
- Multilingual WordNet-based: `datasets/xlwic/xlwic_datasets/xlwic_wn/`
- Wiktionary-based: `datasets/xlwic/xlwic_datasets/xlwic_wikt/`

Sample Data
- See `datasets/xlwic/samples/xlwic_train_en_sample.txt`.

## Dataset 3: MCL-WiC (SemEval-2021 Task 2)

Overview
- Source: https://github.com/SapienzaNLP/mcl-wic
- Size (downloaded): ~3 MB compressed (repo zip), ~8 MB extracted (all datasets)
- Format: Plain text `.data` and `.gold` files
- Task: Multilingual and cross-lingual word-in-context disambiguation
- Splits: train/dev/test, multilingual and cross-lingual pairs
- License: See `datasets/mcl_wic/all_datasets/MCL-WiC/LICENSE.txt`

Download Instructions
- Direct download:
  - `wget -O datasets/mcl_wic.zip https://github.com/SapienzaNLP/mcl-wic/archive/refs/heads/master.zip`
  - `mkdir -p datasets/mcl_wic && unzip -q datasets/mcl_wic.zip -d datasets/mcl_wic`
  - `mkdir -p datasets/mcl_wic/all_datasets && unzip -q datasets/mcl_wic/mcl-wic-master/SemEval-2021_MCL-WiC_all-datasets.zip -d datasets/mcl_wic/all_datasets`

Loading the Dataset
- Main dataset folder: `datasets/mcl_wic/all_datasets/MCL-WiC/`

Sample Data
- See `datasets/mcl_wic/samples/mcl_wic_training_en_en_sample.txt`.
