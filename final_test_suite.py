#!/usr/bin/env python3
"""
FINAL TEST SUITE — Social Science Concept Integration
======================================================

This is a thin CLI wrapper.  All evaluation logic lives in:
  • model_utils_shared.py   — shared utilities (data loading, embeddings, metrics,
                               correlations, graph helpers, ICC)
  • model_utils_clustering.py — UMAP + HDBSCAN clustering
  • model_utils_pairwise.py   — cosine-similarity thresholding
  • model_utils_seed.py       — seeded clustering

For the full evaluation with formatted tables and all five psychometric
audits, **run the notebook**:

    jupyter nbconvert --execute final_testing.ipynb --to html

Or open `final_testing.ipynb` interactively in VS Code / JupyterLab.

Usage (quick smoke-test from CLI):
    python final_test_suite.py
"""

import numpy as np
import torch

import model_utils_shared as shared
import model_utils_clustering as muc
import model_utils_seed as mus
from config import MODELS, TYPE_SENTENCE, BATCH_SIZE, SEED

# ── Best hyperparameters (from cross-validation) ────────────────────────────
MODELS_TO_EVAL = [
    "all-mpnet-base-v2",
    "dwulff/mpnet-personality",
    "allenai/scibert_scivocab_uncased",
    "bert-base-uncased",
]

BEST_PAIRWISE_THRESHOLDS = {
    "all-mpnet-base-v2": 0.69,
    "dwulff/mpnet-personality": 0.66,
    "allenai/scibert_scivocab_uncased": 0.89,
    "bert-base-uncased": 0.90,
}

BEST_CLUSTERING_PARAMS = {
    "all-mpnet-base-v2": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "dwulff/mpnet-personality": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "allenai/scibert_scivocab_uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "bert-base-uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
}

BEST_SEEDED_PARAMS = {
    "all-mpnet-base-v2": {"n_initial_seeds": 250, "threshold": 0.70},
    "dwulff/mpnet-personality": {"n_initial_seeds": 10, "threshold": 0.70},
    "allenai/scibert_scivocab_uncased": {"n_initial_seeds": 250, "threshold": 0.85},
    "bert-base-uncased": {"n_initial_seeds": 250, "threshold": 0.85},
}

TEST_POS = "datasets/processed_datasets/test_positive_pairs.csv"
TEST_NEG = "datasets/processed_datasets/test_negative_pairs.csv"


def main():
    """Run a quick CLI evaluation (see final_testing.ipynb for full audits)."""
    shared.setup_reproducibility(SEED)
    import pandas as pd
    from contextlib import redirect_stdout
    from io import StringIO

    pos_df, neg_df, full_df = shared.load_test_data(TEST_POS, TEST_NEG)
    print(f"Test set: {len(pos_df)} pos + {len(neg_df)} neg = {len(full_df)} total\n")

    terms1 = full_df["term1"].tolist()
    terms2 = full_df["term2"].tolist()
    labels = full_df["label"].values
    unique_terms = pd.unique(full_df[["term1", "term2"]].values.ravel("K")).tolist()
    term_to_idx = {t: i for i, t in enumerate(unique_terms)}
    idx1 = np.array([term_to_idx[t] for t in terms1], dtype=np.int32)
    idx2 = np.array([term_to_idx[t] for t in terms2], dtype=np.int32)
    idx1_t = torch.tensor(idx1, dtype=torch.long)
    idx2_t = torch.tensor(idx2, dtype=torch.long)

    for model_name in MODELS_TO_EVAL:
        cfg = MODELS[model_name]
        display_name = cfg.get("display_name", model_name)
        print(f"{'=' * 60}")
        print(f"  {display_name}")
        print(f"{'=' * 60}")

        model = shared.load_model(model_name, cfg.get("type", TYPE_SENTENCE))
        if model is None:
            continue
        emb = shared.build_embeddings(model, unique_terms, BATCH_SIZE)
        del model

        # Clustering
        params = BEST_CLUSTERING_PARAMS[model_name]
        cl = muc.run_hdbscan_clustering(emb["norm_np"], params["min_cluster_size"], params["min_samples"])
        preds_cl = ((cl[idx1] != -1) & (cl[idx1] == cl[idx2])).astype(int)
        p, r, f1, *_ = shared.compute_metrics(labels, preds_cl)
        print(f"  Clustering   — P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

        # Pairwise
        thr = BEST_PAIRWISE_THRESHOLDS[model_name]
        sims = (emb["norm_t"][idx1_t] * emb["norm_t"][idx2_t]).sum(dim=1).cpu().numpy()
        preds_pw = (sims > thr).astype(int)
        p, r, f1, *_ = shared.compute_metrics(labels, preds_pw)
        print(f"  Pairwise     — P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (θ={thr:.2f})")

        # Seeded
        sp = BEST_SEEDED_PARAMS[model_name]
        norm_t = emb["norm_t"].cpu().detach()
        t2e = {t: e for t, e in zip(unique_terms, norm_t)}
        seeds, rem = mus.sample_unique_terms(sp["n_initial_seeds"], full_df, np.array(unique_terms), SEED)
        with redirect_stdout(StringIO()):
            sc = mus.seed_clustering(seeds, rem, t2e, sp["threshold"])
        t2c = {}
        for cid, (sd, ms) in enumerate(sc.items()):
            for m in ms:
                t2c[m] = cid
        c1 = np.array([t2c.get(t, -1) for t in terms1])
        c2 = np.array([t2c.get(t, -1) for t in terms2])
        preds_sd = ((c1 == c2) & (c1 != -1)).astype(int)
        p, r, f1, *_ = shared.compute_metrics(labels, preds_sd)
        print(f"  Seeded       — P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (seeds={sp['n_initial_seeds']}, θ={sp['threshold']:.2f})")
        print()

    print("For full audits (Reliability, Discriminant Validity, Structural Validity,")
    print("DIF, Semantic Decay) run final_testing.ipynb")


if __name__ == "__main__":
    main()
