# Hybrid Climate Claim Fact-Checking Pipeline

A two-stage automated fact-checking system that retrieves supporting evidence from a 1.2M-passage corpus and classifies climate-related claims as `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, or `DISPUTED`.

Built for COMP90042 (Natural Language Processing), University of Melbourne.

## Overview

Manual fact-checking can't keep up with the volume of climate misinformation online. This project tackles the problem as two coupled subtasks:

1. **Evidence retrieval** — given a claim, find the most relevant passages out of 1.2M+ candidates
2. **Claim classification** — given a claim and its retrieved evidence, decide whether the claim is supported, refuted, disputed, or unverifiable

The two stages aren't independent: bad retrieval feeds bad evidence to a good classifier, and a weak classifier can fail even with perfect evidence. This pipeline treats both as first-class problems rather than optimizing one in isolation.

## Approach

**Retrieval** combines four complementary signals, fused with Reciprocal Rank Fusion (k=60) and reranked with a cross-encoder:
- BM25 (sparse, lexical)
- Dense retrieval via `all-mpnet-base-v2` embeddings + FAISS (`IndexFlatIP`)
- TF-IDF (bigrams, sublinear scaling)
- Query-expanded BM25 (top BM25 terms added back into the query)
- Final reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 25 fused candidates → top 5 kept

**Classification** compares three models, all trained with label smoothing (ε=0.1) and inverse-frequency class weights to handle severe class imbalance (DISPUTED has only 124 training examples vs. 519 for SUPPORTS):
- BiLSTM over `all-mpnet-base-v2` embeddings, reshaped into a 12×64 sequence, with attention pooling
- `bert-base-uncased`, fine-tuned on claim + top-3 evidence passages
- `roberta-base`, same setup as BERT — this was the best performer and became the primary model

DeBERTa-v3-small/large with LoRA were also tried but underperformed RoBERTa and were dropped.

## Results

| Component | Result |
|---|---|
| Best retriever (hybrid + RRF + rerank) | F-score 0.1850 (top-5, dev) |
| Best classifier (RoBERTa-base, gold evidence) | 0.7143 dev accuracy |
| Full end-to-end pipeline | F=0.1682, A=0.5065, **H=0.2525** |

Retrieval is the clear bottleneck — classification accuracy drops from 0.71 (given correct evidence) to 0.51 (given retrieved evidence), showing the harmonic mean is limited almost entirely by retrieval quality rather than classifier quality. Full per-class breakdowns, ablations, and error analysis are in the report.

## Repository Contents

- `COMP90042_A3.ipynb` — full pipeline: data loading, BM25/dense/TF-IDF/query-expansion retrieval, RRF fusion, cross-encoder reranking, BiLSTM/BERT/RoBERTa training, evaluation, and error analysis
- `report.pdf` — full written report with methodology, results tables, ablations, and comparison against an alternative DeBERTa+LoRA system

## Data

**Not included in this repo.** The notebook expects the following files (from the COMP90042 project dataset) inside `DATA_DIR`:
- `train-claims.json` (1,228 claims)
- `dev-claims.json` (154 claims)
- `test-claims-unlabelled.json` (153 claims)
- `evidence.json` (1,208,827 evidence passages)

These are University of Melbourne / subject-provided files and are not redistributed here. To run the notebook, place these files in the path set by `DATA_DIR` in the config cell, or update that variable to point to your own copy.

## Requirements

```
torch
transformers
sentence-transformers
faiss-gpu (or faiss-cpu)
rank_bm25
scikit-learn
numpy
```

The notebook was developed on Google Colab and mounts Google Drive for data and checkpoint storage (`FOLDER_NAME = 'COMP90042'`); adjust `DATA_DIR` / `CKPT_DIR` if running elsewhere.

## Limitations

- Retrieval F-score (0.168–0.185) is the main bottleneck on the full pipeline; a comparison system using BM25 with Porter stemming and margin-gated dynamic top-K evidence selection reached a higher harmonic mean (0.3186) primarily via better retrieval, not a stronger classifier
- DISPUTED is the weakest class (F1=0.40) due to limited training examples (124) and inherent label ambiguity
- Small training set (1,228 claims) limits generalization, especially for minority classes

## Author

Manav Gupta — COMP90042, University of Melbourne
