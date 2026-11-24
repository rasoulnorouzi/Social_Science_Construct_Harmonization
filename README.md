# Social Science Construct Harmonization

This project benchmarks machine learning models for concept harmonization: merging heterogeneous social science terms (e.g., "Insomnia" and "Sleeping Disorders") into unified constructs. A factorial design crosses four embedding models with three harmonization strategies (UMAP + HDBSCAN clustering, pairwise cosine thresholding, seeded clustering), next to a character 3-gram baseline without a language model. Each pipeline is evaluated on two expert thesauri and five diagnostic tests (embedding stability, lexical trap, rare-word bias, semantic decay, structural alignment).

## Versions

| Tag | Commit | Corresponds to |
|-----|--------|----------------|
| `v1.0-submission` | `36c33ab` | Results reported in the initial submission to *Communications Psychology* (archived on Zenodo, doi:10.5281/zenodo.24355576) |
| `main` after the revision | to be tagged | Revised pair construction, seed sampler, baseline and diagnostics (see below) |

Check out a tag to obtain the exact code, settings and results behind a manuscript version:

```bash
git checkout v1.0-submission
```

## Data

| Dataset | Role | Licence / access |
|---------|------|------------------|
| ELSST (European Language Social Science Thesaurus), release used: `ELSST_R5.rdf` | Calibration (train split) and held-out test (test split) | CC BY-SA 4.0, included in `datasets/raw_datasets/`; also at https://elsst.cessda.eu |
| APA Thesaurus of Psychological Index Terms (Zthes XML) | Out-of-distribution evaluation only; never used for calibration | Rights belong to the American Psychological Association. Not redistributed here. Request access from psycinfo@apa.org |

To reproduce the APA analyses, place the APA Zthes export at `datasets/raw_datasets/APA.xml`.

## Reproducing the results

1. **Environment:** Python 3.13. Install dependencies with `pip install -r requirements.txt`. The file is UTF-16 encoded; if your tooling rejects it, convert it first with `iconv -f utf-16 -t utf-8 requirements.txt > requirements-utf8.txt`. On Google Colab, torch, sentence-transformers, scikit-learn, pandas and pyarrow are preinstalled; install the rest with `pip install rdflib wordfreq umap-learn`.
2. **Build pair datasets:** `python scripts/process_dataset.py` (about one minute). It writes Parquet files to `datasets/processed_datasets/elsst/` (prefixes `train_` and `test_`) and `datasets/processed_datasets/apa/` (no prefix). Each split has `terms`, `concepts`, `positive_pairs`, `negative_pairs` and `hierarchy_pairs` files; pairs store integer term ids (`idx1`, `idx2`) that point into the `terms` file. `python scripts/process_dataset.py --count-only` prints the counts without writing; `python scripts/process_dataset.py elsst` (or `apa`) builds one data set only.
   - ELSST is split 70/30 by concept with seed 42. Concepts are grouped with union-find so that no term appears in both splits. APA is built once over the whole thesaurus (no split).
   - Positive pairs are label combinations within a concept.
   - Negative pairs are all label combinations (preferred labels and entry terms) of two concepts that are not on the same family line; each concept's ancestors and descendants, up to the root, are excluded.
   - Broader/narrower pairs (a concept with each of its ancestors) are written to a separate file and are not part of the benchmark.
3. **Calibrate (ELSST train split only):** run `analysis/pairwise_analysis.ipynb` (models and character 3-gram baseline), `analysis/clustering_analysis.ipynb` and `analysis/seed_clustering_analysis.ipynb`. Each notebook prints the selected values as a block to paste into `scripts/config.py` (`BEST_PAIRWISE_THRESHOLDS`, `BEST_NGRAM_THRESHOLD`, `BEST_CLUSTERING_PARAMS`, `BEST_SEEDED_PARAMS`).
4. **Evaluate and run diagnostics with the frozen settings:** run `analysis/elsst_diagnostic_analysis.ipynb` (ELSST test split) and `analysis/apa_diagnostic_analysis.ipynb` (full APA thesaurus). Both notebooks first check that `config.py` matches the calibration results. Outputs go to `analysis/results/{elsst,apa}/`. Set `FAST_DEBUG = True` in the configuration cell for a quick test run on a sample (not valid results).
5. **Tables:** `analysis/paper_tables.ipynb` reads the result CSVs of both diagnostic notebooks and writes every main-text and supplementary table of the paper to `analysis/results/paper/` (CSV and LaTeX).

**Device:** `scripts/config.py` uses the GPU when one is available (`HARMONIZATION_DEVICE=cpu` forces the CPU). The GPU only encodes the terms; similarity matrices, clustering, metrics and the bootstrap run on the CPU, so results do not depend on the device beyond the rounding of the encoder. The APA notebook holds about 60 million pairs in memory; use a runtime with at least 12 GB RAM (high-RAM on Colab).

### What each diagnostic computes

| Test | Pairs | Output |
|------|-------|--------|
| Embedding stability | all | Mean and SD of F1, mean change from the noise-free run, agreement of predicted matches (Jaccard overlap of predicted synonym pairs) under Gaussian embedding noise; seed variation of seeded clustering reported separately |
| Lexical trap | negatives with character 3-gram Jaccard > 0.5 vs = 0 | False-positive rates |
| Rare-word bias | positives in the bottom vs top decile of word frequency | Recall gap, overall and within three lexical-overlap bands |
| Semantic decay | positives with Jaccard > 0.5 vs = 0 | Recall gap |
| Structural alignment | negatives with a thesaurus path | (A) Spearman ρ between similarity and path length per model; (B) false-merge rate by path length per pipeline |

Further outputs: false-positive rate and F1 at the ELSST test prevalence, broader/narrower merge rates by depth, 95 % concept-level bootstrap intervals and paired F1 differences, top-k recall and correct matches among the top 100/500/1,000 ranked pairs.

Definitions of the techniques, metrics and diagnostic tests are in [`analysis/methodology.md`](analysis/methodology.md). The audit-set thresholds are explained in [`analysis/sampling_threshold_logic.md`](analysis/sampling_threshold_logic.md).

Result CSVs in `analysis/results/` are stored with Git LFS. Run `git lfs pull` after cloning.

## Interactive demo

`web-concept-clustering/` is a browser app, deployed to GitHub Pages, that clusters user-supplied terms against precomputed ELSST embeddings.

## Licence

GNU General Public License v3.0. See [`LICENSE`](LICENSE).
