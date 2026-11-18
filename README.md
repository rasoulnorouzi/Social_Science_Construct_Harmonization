# Social Science Construct Harmonization

This project benchmarks machine learning models for concept harmonization: merging heterogeneous social science terms (e.g., "Insomnia" and "Sleeping Disorders") into unified constructs. A factorial design crosses four embedding models with three harmonization strategies (UMAP + HDBSCAN clustering, pairwise cosine thresholding, seeded clustering). Each of the twelve pipelines is evaluated on two expert thesauri and five diagnostic tests (embedding stability, lexical trap, rare-word bias, semantic decay, structural alignment).

## Versions

| Tag | Commit | Corresponds to |
|-----|--------|----------------|
| `v1.0-submission` | `36c33ab` | Results reported in the initial submission to *Communications Psychology* |

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

1. **Environment:** Python 3.13. Install dependencies with `pip install -r requirements.txt`. The file is UTF-16 encoded; if your tooling rejects it, convert it first with `iconv -f utf-16 -t utf-8 requirements.txt > requirements-utf8.txt`.
2. **Build pair datasets:** `python scripts/process_dataset.py`. This writes `datasets/processed_datasets/{elsst,apa}/{train,test}_{positive,negative}_pairs.csv`.
   - The split is 70/30 by concept with seed 42. Concepts are grouped with union-find so that no term appears in both splits.
   - Positive pairs are label combinations within a concept.
   - Negative pairs exclude each concept's ancestors and descendants in the thesaurus hierarchy.
3. **Calibrate (ELSST train split only):** run `analysis/pairwise_analysis.ipynb`, `analysis/clustering_analysis.ipynb` and `analysis/seed_clustering_analysis.ipynb`. The selected hyperparameters are frozen in `scripts/config.py` (`BEST_PAIRWISE_THRESHOLDS`, `BEST_CLUSTERING_PARAMS`, `BEST_SEEDED_PARAMS`).
4. **Evaluate and run diagnostics with the frozen settings:** run `analysis/elsst_diagnostic_analysis.ipynb` (ELSST test split) and `analysis/apa_diagnostic_analysis.ipynb` (APA train and test splits merged). Outputs go to `analysis/results/{elsst,apa}/`.
5. **Figures:** `analysis/publication_figures.ipynb` writes `analysis/results/publication/figures/`.

Definitions of the techniques, metrics and diagnostic tests are in [`analysis/methodology.md`](analysis/methodology.md). The audit-set thresholds are explained in [`analysis/sampling_threshold_logic.md`](analysis/sampling_threshold_logic.md).

Result CSVs in `analysis/results/` are stored with Git LFS (about 2.1 GB). Run `git lfs pull` after cloning.

## Interactive demo

`web-concept-clustering/` is a browser app, deployed to GitHub Pages, that clusters user-supplied terms against precomputed ELSST embeddings.

## Licence

GNU General Public License v3.0. See [`LICENSE`](LICENSE).
