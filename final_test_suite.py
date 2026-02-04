#!/usr/bin/env python3
"""
FINAL TEST SUITE FOR SOCIAL SCIENCE CONCEPT INTEGRATION
========================================================

This script performs a comprehensive evaluation of ML models for concept harmonization:
1. Loads test data using the_loading function
2. Evaluates each model under 3 techniques (Clustering, Pairwise, Seeded Clustering)
3. Runs 5 audits: Reliability, Discriminant Validity, Structural Validity, DIF, Semantic Decay

Usage: python final_test_suite.py
"""

import os
import sys
import random
import re
from collections import defaultdict, deque
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize

# HDBSCAN import with fallback
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    from hdbscan import HDBSCAN

# Optional imports with graceful fallback
try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

# Local imports
import model_utils_shared
import model_utils_seed as mus
from config import MODELS, TYPE_TOKEN, TYPE_SENTENCE, BATCH_SIZE, SEED


# ==============================================================================
#                              FIXED BEST CONFIGS
# ==============================================================================
# These are pre-tuned hyperparameters from cross-validation. DO NOT MODIFY.

MODELS_TO_EVAL = [
    "all-mpnet-base-v2",
    "dwulff/mpnet-personality",
    "allenai/scibert_scivocab_uncased",
    "bert-base-uncased",
]

BEST_CLUSTERING_PARAMS = {
    "all-mpnet-base-v2": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "dwulff/mpnet-personality": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "allenai/scibert_scivocab_uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "bert-base-uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
}

BEST_PAIRWISE_THRESHOLDS = {
    "all-mpnet-base-v2": 0.69,
    "dwulff/mpnet-personality": 0.66,
    "allenai/scibert_scivocab_uncased": 0.89,
    "bert-base-uncased": 0.90,
}

BEST_SEEDED_PARAMS = {
    "all-mpnet-base-v2": {"n_initial_seeds": 250, "threshold": 0.70},
    "dwulff/mpnet-personality": {"n_initial_seeds": 10, "threshold": 0.70},
    "allenai/scibert_scivocab_uncased": {"n_initial_seeds": 250, "threshold": 0.85},
    "bert-base-uncased": {"n_initial_seeds": 250, "threshold": 0.85},
}


# ==============================================================================
#                              DATA LOADING
# ==============================================================================

def the_loading(pos_path, neg_path):
    """
    Required dataset loader for test datasets.
    Returns: (pos_df, neg_df, full_df)
    """
    print("Loading test datasets with the_loading...")
    pos_df = pd.read_csv(pos_path, on_bad_lines="skip")
    neg_df = pd.read_csv(neg_path, on_bad_lines="skip")

    # Ensure labels exist
    if "label" not in pos_df.columns:
        pos_df["label"] = 1
    if "label" not in neg_df.columns:
        neg_df["label"] = 0

    # Normalize term types
    pos_df["term1"] = pos_df["term1"].astype(str)
    pos_df["term2"] = pos_df["term2"].astype(str)
    neg_df["term1"] = neg_df["term1"].astype(str)
    neg_df["term2"] = neg_df["term2"].astype(str)

    full_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return pos_df, neg_df, full_df


# ==============================================================================
#                              UTILITIES
# ==============================================================================

TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokens(text):
    """Tokenize text into lowercase alphanumeric words."""
    return TOKEN_RE.findall(str(text).lower())


def letter_ngrams(text, n=3):
    """Generate character n-grams for letter-level similarity."""
    text = str(text).lower()
    # Keep only alphanumeric chars
    text = "".join(c for c in text if c.isalnum())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_sim(a_tokens, b_tokens):
    """Compute Jaccard similarity between two token sets."""
    if not a_tokens and not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union) if union else 0.0


def set_reproducibility(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confusion_counts(y_true, y_pred):
    """Compute TP, FP, TN, FN from predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_metrics(y_true, y_pred):
    """Compute Precision, Recall, F1 and confusion matrix counts."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    return p, r, f1, tp, fp, tn, fn


def recall_on_subset(y_true, y_pred, mask):
    """Compute recall on a subset defined by mask."""
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def spearman_corr(x, y):
    """Compute Spearman rank correlation."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if spearmanr is not None:
        corr = spearmanr(x, y, nan_policy="omit").correlation
        return float(corr) if corr is not None else np.nan
    # Fallback without scipy
    x_rank = pd.Series(x).rank(method="average").values
    y_rank = pd.Series(y).rank(method="average").values
    if x_rank.std() == 0 or y_rank.std() == 0:
        return np.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


# ==============================================================================
#                              PRINTING UTILITIES
# ==============================================================================

def print_header(title):
    """Print a visually distinct section header."""
    border = "=" * max(60, len(title) + 4)
    print(f"\n{border}")
    print(f"  {title}")
    print(border)


def print_subheader(title):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def print_table(headers, rows):
    """Print a formatted ASCII table."""
    if not rows:
        print("(No data)")
        return
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(val)))
    
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


# ==============================================================================
#                              MODEL LOADING & EMBEDDINGS
# ==============================================================================

def load_model_for_name(model_name):
    """Load a model by name using config-defined type."""
    model_cfg = MODELS.get(model_name)
    if not model_cfg:
        raise ValueError(f"Unknown model: {model_name}")
    model_type = model_cfg.get("type", TYPE_SENTENCE)
    return model_utils_shared.load_model(model_name=model_name, model_type=model_type)


def build_embeddings(model, terms, batch_size=BATCH_SIZE):
    """Build and cache embeddings for unique terms."""
    print(f"  Encoding {len(terms)} unique terms...")
    raw_t = model.encode(list(terms), convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
    raw_t = raw_t.cpu()
    raw_np = raw_t.numpy()

    # L2 normalize for cosine similarity
    norm_np = normalize(raw_np, norm="l2", axis=1)
    norm_np = np.ascontiguousarray(norm_np, dtype=np.float64)
    norm_t = F.normalize(raw_t, p=2, dim=1)

    # Noise scale: per-dimension standard deviation for reliability audit
    sigma = raw_np.std(axis=0, ddof=0)

    return {
        "raw_t": raw_t,
        "raw_np": raw_np,
        "norm_t": norm_t,
        "norm_np": norm_np,
        "sigma": sigma,
    }


# ==============================================================================
#                              TECHNIQUE IMPLEMENTATIONS
# ==============================================================================

def clustering_predict(emb_norm_np, idx1, idx2, params, random_state=SEED):
    """
    Run HDBSCAN clustering and predict pair relationships.
    Uses same logic as model_utils_clustering.py.
    """
    n_components = params.get("n_components")
    
    # Dimensionality reduction with UMAP if needed
    if n_components is None or n_components >= emb_norm_np.shape[1]:
        reduced = emb_norm_np
    else:
        if UMAP is None:
            raise ImportError("UMAP is required but not installed.")
        reducer = UMAP(
            n_neighbors=15,
            n_components=int(n_components),
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
        )
        reduced = reducer.fit_transform(emb_norm_np)

    reduced = np.ascontiguousarray(reduced, dtype=np.float64)
    
    clusterer = HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        metric="cosine",
        copy=True,
    )
    cluster_labels = clusterer.fit_predict(reduced)
    
    # Predict: same cluster if both have matching non-noise labels
    l1 = cluster_labels[idx1]
    l2 = cluster_labels[idx2]
    preds = ((l1 != -1) & (l1 == l2)).astype(int)
    
    return preds, cluster_labels


def pairwise_similarity(norm_t, idx1_t, idx2_t):
    """Compute pairwise cosine similarities."""
    sims = (norm_t[idx1_t] * norm_t[idx2_t]).sum(dim=1)
    return sims.cpu().numpy()


def pairwise_predict(norm_t, idx1_t, idx2_t, threshold):
    """Predict pair relationships based on similarity threshold."""
    sims = pairwise_similarity(norm_t, idx1_t, idx2_t)
    preds = (sims > threshold).astype(np.int8)
    return preds, sims


def seeded_predict(df_pairs, unique_terms, term_to_embedding, n_seeds, threshold, random_state=SEED):
    """
    Run seeded clustering and predict pair relationships.
    Uses same logic as model_utils_seed.py.
    """
    initial_seeds, remaining_terms = mus.sample_unique_terms(
        n_seeds, df_pairs, unique_terms, random_state=random_state
    )
    seeds_cluster = mus.seed_clustering(
        initial_seeds, remaining_terms, term_to_embedding, threshold=threshold
    )
    
    # Build term-to-cluster mapping
    term_to_cluster = {}
    for cluster_id, (seed, terms) in enumerate(seeds_cluster.items()):
        for term in terms:
            term_to_cluster[term] = cluster_id

    # Predict pair relationships
    cluster1 = np.array([term_to_cluster.get(t, -1) for t in df_pairs["term1"]])
    cluster2 = np.array([term_to_cluster.get(t, -1) for t in df_pairs["term2"]])
    preds = ((cluster1 == cluster2) & (cluster1 != -1)).astype(int)
    
    return preds, term_to_cluster


# ==============================================================================
#                              GRAPH UTILITIES (for Audit 3)
# ==============================================================================

def union_find(n):
    """Create a union-find data structure."""
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return parent, find, union


def model_graph_shortest_paths(n_nodes, edges_idx1, edges_idx2, query_idx1, query_idx2):
    """Compute shortest paths on a graph built from pairwise positive predictions."""
    # Build adjacency list
    adj = [set() for _ in range(n_nodes)]
    for a, b in zip(edges_idx1, edges_idx2):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    # Group queries by source for efficient BFS
    pairs_by_src = defaultdict(list)
    for i, src in enumerate(query_idx1):
        pairs_by_src[src].append(i)

    distances = np.full(len(query_idx1), -1, dtype=np.int32)

    for src, pair_indices in pairs_by_src.items():
        target_to_indices = defaultdict(list)
        for i in pair_indices:
            target_to_indices[query_idx2[i]].append(i)
        targets = set(target_to_indices.keys())
        
        if src in targets:
            for i in target_to_indices[src]:
                distances[i] = 0
            targets.discard(src)
        if not targets:
            continue

        # BFS from source
        visited = {src}
        q = deque([(src, 0)])
        while q and targets:
            node, dist = q.popleft()
            for nbr in adj[node]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                if nbr in targets:
                    for i in target_to_indices[nbr]:
                        distances[i] = dist + 1
                    targets.discard(nbr)
                q.append((nbr, dist + 1))
    
    return distances


# ==============================================================================
#                              AUDIT 1: RELIABILITY (STABILITY TEST)
# ==============================================================================

def compute_icc_oneway(sum_y, sum_y2, n_targets, k_raters):
    """
    Compute ICC(1,1): one-way random effects.
    This measures consistency across runs with stochastic variation.
    """
    if n_targets < 2 or k_raters < 2:
        return 0.0
    grand_mean = sum_y.sum() / (n_targets * k_raters)
    mean_i = sum_y / k_raters
    msr = k_raters * np.sum((mean_i - grand_mean) ** 2) / (n_targets - 1)
    ss_within = np.sum(sum_y2 - k_raters * mean_i ** 2)
    mse = ss_within / (n_targets * (k_raters - 1))
    denom = msr + (k_raters - 1) * mse
    if denom == 0:
        return 0.0
    return float((msr - mse) / denom)


def run_reliability_audit(technique, model_name, embedding_cache, unique_terms, full_df,
                          idx1, idx2, idx1_t, idx2_t, labels, noise_levels=None, n_runs=5):
    """
    Audit 1: Measure Standard Error of Model (SEM) under stochastic noise.
    
    Procedure:
    - Test multiple noise levels: noise_level * σ (per-dimension std of raw embeddings)
    - Re-run decision pipeline n_runs times per noise level
    - Compute mean F1, SD_F1, ICC, SEM = SD_F1 * sqrt(1 - ICC) for each noise level
    
    Returns dict with F1 results for each noise level and ICC/SEM metrics.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    emb_raw = embedding_cache["raw_np"]
    sigma = embedding_cache["sigma"]  # Per-dimension std
    
    params_cluster = BEST_CLUSTERING_PARAMS.get(model_name, {})
    threshold_pairwise = BEST_PAIRWISE_THRESHOLDS.get(model_name, 0.5)
    params_seeded = BEST_SEEDED_PARAMS.get(model_name, {})
    
    results_by_noise = {}
    
    for noise_level in noise_levels:
        f1_scores = []
        all_predictions = []  # For ICC computation
        
        for run_idx in range(n_runs):
            run_seed = SEED + run_idx + int(noise_level * 10000)
            rng = np.random.default_rng(run_seed)
            
            if noise_level == 0.0:
                # Baseline: no noise
                noisy_raw = emb_raw
            else:
                # Inject relative Gaussian noise (noise_level * per-dimension sigma)
                noise = rng.normal(0.0, 1.0, size=emb_raw.shape) * (noise_level * sigma)
                noisy_raw = emb_raw + noise
            
            # Re-normalize
            noisy_norm_np = normalize(noisy_raw, norm="l2", axis=1)
            noisy_norm_t = F.normalize(torch.tensor(noisy_raw, dtype=torch.float32), p=2, dim=1)
            
            # Run technique
            if technique == "clustering":
                preds, _ = clustering_predict(noisy_norm_np, idx1, idx2, params_cluster, random_state=run_seed)
            elif technique == "pairwise":
                preds, _ = pairwise_predict(noisy_norm_t, idx1_t, idx2_t, threshold_pairwise)
            else:  # seeded
                noisy_norm_t = noisy_norm_t.cpu().detach()
                term_to_embedding = {t: e for t, e in zip(unique_terms, noisy_norm_t)}
                with redirect_stdout(StringIO()):  # Suppress verbose output
                    preds, _ = seeded_predict(
                        full_df, np.array(unique_terms), term_to_embedding,
                        params_seeded.get("n_initial_seeds", 250),
                        params_seeded.get("threshold", 0.70),
                        random_state=run_seed
                    )
            
            f1 = f1_score(labels, preds, zero_division=0)
            f1_scores.append(f1)
            all_predictions.append(preds)
        
        f1_scores = np.array(f1_scores)
        mean_f1 = np.mean(f1_scores)
        sd_f1 = np.std(f1_scores, ddof=1)
        
        # Compute ICC across runs (for per-pair decisions)
        all_predictions = np.array(all_predictions)  # Shape: (n_runs, n_pairs)
        n_pairs = all_predictions.shape[1]
        sum_y = all_predictions.sum(axis=0)
        sum_y2 = (all_predictions ** 2).sum(axis=0)
        icc = compute_icc_oneway(sum_y, sum_y2, n_pairs, n_runs)
        
        # SEM = SD_F1 * sqrt(1 - ICC)
        sem = sd_f1 * np.sqrt(max(0, 1 - icc))
        
        # 95% CI
        ci_95 = 1.96 * sem
        
        results_by_noise[noise_level] = {
            "mean_f1": mean_f1,
            "sd_f1": sd_f1,
            "icc": icc,
            "sem": sem,
            "ci_95_lower": mean_f1 - ci_95,
            "ci_95_upper": mean_f1 + ci_95,
        }
    
    return results_by_noise


# ==============================================================================
#                              MAIN EVALUATION
# ==============================================================================

def main():
    set_reproducibility(SEED)

    # ========================
    # LOAD DATA
    # ========================
    print_header("LOADING TEST DATA")
    
    pos_path = "datasets/processed_datasets/test_positive_pairs.csv"
    neg_path = "datasets/processed_datasets/test_negative_pairs.csv"
    
    pos_df, neg_df, full_df = the_loading(pos_path, neg_path)
    
    print(f"  Positive pairs: {len(pos_df)}")
    print(f"  Negative pairs: {len(neg_df)}")
    print(f"  Total samples:  {len(full_df)}")
    print("  Dataset includes shortest_path distances from ELSST graph.")

    # Prepare indices and labels
    terms1 = full_df["term1"].astype(str).tolist()
    terms2 = full_df["term2"].astype(str).tolist()
    labels = full_df["label"].astype(int).values
    shortest_path = full_df.get("shortest_path", pd.Series([-1] * len(full_df))).values

    unique_terms = pd.unique(full_df[["term1", "term2"]].values.ravel("K")).tolist()
    term_to_idx = {t: i for i, t in enumerate(unique_terms)}

    idx1 = np.array([term_to_idx[t] for t in terms1], dtype=np.int32)
    idx2 = np.array([term_to_idx[t] for t in terms2], dtype=np.int32)
    idx1_t = torch.tensor(idx1, dtype=torch.long)
    idx2_t = torch.tensor(idx2, dtype=torch.long)

    pos_len = len(pos_df)
    pos_mask = np.zeros(len(full_df), dtype=bool)
    pos_mask[:pos_len] = True

    # Token cache for lexical audits
    # Changed to letter_ngrams per user request (Audit 2 & 5)
    token_cache = {t: letter_ngrams(t, n=3) for t in unique_terms}

    # ========================
    # PRECOMPUTE AUDIT MASKS
    # ========================
    print_subheader("Precomputing Audit Masks")

    # Audit 2: Lexical trap (Hard Negatives)
    # Definition: Jaccard > 0.5 (N=65). "Discriminant Validity (Traps)"
    # Unrelated concepts with high orthographic overlap.
    lex_mask_neg = []
    for t1, t2 in zip(neg_df["term1"], neg_df["term2"]):
        s1 = token_cache.get(t1, set())
        s2 = token_cache.get(t2, set())
        lex_mask_neg.append(jaccard_sim(s1, s2) > 0.5)
    lex_mask = np.zeros(len(full_df), dtype=bool)
    lex_mask[pos_len:] = np.array(lex_mask_neg, dtype=bool)
    print(f"  Lexical trap subset (Hard Negatives, Jaccard > 0.5): {lex_mask.sum()}")

    # Audit 5: Semantic decay (positive pairs)
    # Hard Positives: Jaccard == 0.0 (Semantic Gap Test)
    # Easy Positives: Jaccard > 0.5 (Lexical Anchors)
    jacc_pos = []
    for t1, t2 in zip(pos_df["term1"], pos_df["term2"]):
        s1 = token_cache.get(t1, set())
        s2 = token_cache.get(t2, set())
        jacc_pos.append(jaccard_sim(s1, s2))
    jacc_pos = np.array(jacc_pos)
    easy_mask = np.zeros(len(full_df), dtype=bool)
    hard_mask = np.zeros(len(full_df), dtype=bool)
    easy_mask[:pos_len] = jacc_pos > 0.5
    hard_mask[:pos_len] = jacc_pos == 0.0
    print(f"  Easy Potives (Lexical Anchors, Jaccard > 0.5): {easy_mask.sum()}")
    print(f"  Hard Positives (Semantic Gap, Jaccard == 0): {hard_mask.sum()}")

    # Audit 4: Rare word bins
    if zipf_frequency is None:
        print("  WARNING: wordfreq not installed. Audit 4 will be skipped.")
        rare_mask = np.zeros(len(full_df), dtype=bool)
        common_mask = np.zeros(len(full_df), dtype=bool)
    else:
        term_freq = {}
        for t in unique_terms:
            toks = simple_tokens(t)
            if not toks:
                term_freq[t] = 0.0
            else:
                term_freq[t] = float(np.mean([zipf_frequency(tok, "en") for tok in toks]))

        pos_pair_freq = []
        for t1, t2 in zip(pos_df["term1"], pos_df["term2"]):
            f1 = term_freq.get(t1, 0.0)
            f2 = term_freq.get(t2, 0.0)
            pos_pair_freq.append((f1 + f2) / 2.0)
        pos_pair_freq = np.array(pos_pair_freq)

        low_thr = np.quantile(pos_pair_freq, 0.10) if len(pos_pair_freq) > 0 else 0.0
        high_thr = np.quantile(pos_pair_freq, 0.90) if len(pos_pair_freq) > 0 else 0.0

        rare_mask = np.zeros(len(full_df), dtype=bool)
        common_mask = np.zeros(len(full_df), dtype=bool)
        rare_mask[:pos_len] = pos_pair_freq <= low_thr
        common_mask[:pos_len] = pos_pair_freq >= high_thr
        print(f"  Common pairs (top 10% freq): {common_mask.sum()}")
        print(f"  Rare pairs (bottom 10% freq): {rare_mask.sum()}")

    # ========================
    # LOAD EMBEDDINGS
    # ========================
    print_header("LOADING MODEL EMBEDDINGS")
    print("Loading embeddings once for reuse across all techniques and audits...\n")

    embedding_cache = {}
    for model_name in MODELS_TO_EVAL:
        print(f"  Loading {model_name}... ", end="", flush=True)
        model = load_model_for_name(model_name)
        if model is None:
            print("FAILED")
            continue
        embedding_cache[model_name] = build_embeddings(model, unique_terms)
        print("Done")

    # Store predictions for audits
    preds_by_technique = {"clustering": {}, "pairwise": {}, "seeded": {}}

    # ==========================================================================
    #                      FINAL EVALUATION - CLUSTERING
    # ==========================================================================
    print_header("FINAL EVALUATION — CLUSTERING")
    clustering_rows = []

    for model_name in MODELS_TO_EVAL:
        if model_name not in embedding_cache:
            continue
        
        norm_np = embedding_cache[model_name]["norm_np"]
        params = BEST_CLUSTERING_PARAMS[model_name]
        preds, cluster_labels = clustering_predict(norm_np, idx1, idx2, params)
        p, r, f1, tp, fp, tn, fn = compute_metrics(labels, preds)

        clustering_rows.append([model_name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", tp, fp, tn, fn])
        preds_by_technique["clustering"][model_name] = {
            "preds": preds, "cluster_labels": cluster_labels, "idx1": idx1, "idx2": idx2
        }

    print_table(["Model", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"], clustering_rows)

    # ==========================================================================
    #                      FINAL EVALUATION - PAIRWISE
    # ==========================================================================
    print_header("FINAL EVALUATION — PAIRWISE")
    pairwise_rows = []

    for model_name in MODELS_TO_EVAL:
        if model_name not in embedding_cache:
            continue
        
        norm_t = embedding_cache[model_name]["norm_t"]
        threshold = BEST_PAIRWISE_THRESHOLDS[model_name]
        preds, sims = pairwise_predict(norm_t, idx1_t, idx2_t, threshold)
        p, r, f1, tp, fp, tn, fn = compute_metrics(labels, preds)

        pairwise_rows.append([model_name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", tp, fp, tn, fn])
        preds_by_technique["pairwise"][model_name] = {"preds": preds, "similarities": sims}

    print_table(["Model", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"], pairwise_rows)

    # ==========================================================================
    #                      FINAL EVALUATION - SEEDED CLUSTERING
    # ==========================================================================
    print_header("FINAL EVALUATION — SEEDED CLUSTERING")
    seeded_rows = []

    for model_name in MODELS_TO_EVAL:
        if model_name not in embedding_cache:
            continue
        
        norm_t = embedding_cache[model_name]["norm_t"].cpu().detach()
        term_to_embedding = {t: e for t, e in zip(unique_terms, norm_t)}
        params = BEST_SEEDED_PARAMS[model_name]

        preds, term_to_cluster = seeded_predict(
            full_df, np.array(unique_terms), term_to_embedding,
            params["n_initial_seeds"], params["threshold"]
        )
        p, r, f1, tp, fp, tn, fn = compute_metrics(labels, preds)

        seeded_rows.append([model_name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", tp, fp, tn, fn])
        preds_by_technique["seeded"][model_name] = {"preds": preds, "term_to_cluster": term_to_cluster}

    print_table(["Model", "Precision", "Recall", "F1", "TP", "FP", "TN", "FN"], seeded_rows)

    # ==========================================================================
    #                      AUDIT 1 — RELIABILITY (STABILITY TEST)
    # ==========================================================================
    print_header("AUDIT 1 — RELIABILITY (STABILITY TEST)")
    print("\nWhat this means:")
    print("  Measures Standard Error of Model (SEM) under stochastic embedding noise.")
    print("  Tests robustness across noise levels: 0.0 (baseline), 0.05, 0.1, 0.2, 0.3")
    print("  SEM = SD_F1 * sqrt(1 - ICC), where ICC = Intraclass Correlation Coefficient.")
    print("  Lower SEM = more stable model. 5 runs per noise level.\n")

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    for technique in ["clustering", "pairwise", "seeded"]:
        print_subheader(f"Technique: {technique.upper()}")
        audit1_rows = []
        
        for model_name in MODELS_TO_EVAL:
            if model_name not in embedding_cache:
                continue
            
            print(f"  Testing {model_name}...", end="", flush=True)
            results = run_reliability_audit(
                technique, model_name, embedding_cache[model_name],
                unique_terms, full_df, idx1, idx2, idx1_t, idx2_t, labels,
                noise_levels=noise_levels, n_runs=5
            )
            print(" Done")
            
            # Build row with F1 scores for each noise level
            row = [model_name]
            best_f1 = -1
            best_noise = None
            
            for noise_level in noise_levels:
                mean_f1 = results[noise_level]["mean_f1"]
                row.append(f"{mean_f1:.4f}")
                
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_noise = noise_level
            
            # Add best performance indicator
            if best_noise == 0.0:
                best_label = "Baseline"
            else:
                best_label = f"Noise={best_noise}"
            row.append(best_label)
            
            audit1_rows.append(row)
        
        # Print table for this technique
        headers = ["Model", "F1@0.0", "F1@0.05", "F1@0.1", "F1@0.2", "F1@0.3", "Best"]
        print_table(headers, audit1_rows)
        print()

    # ==========================================================================
    #                      AUDIT 2 — DISCRIMINANT VALIDITY (LEXICAL TRAP)
    # ==========================================================================
    print_header("AUDIT 2 — DISCRIMINANT VALIDITY (LEXICAL TRAP)")
    print("\nWhat this means:")
    print("  Measures False Positive Rate on negative pairs with high lexical similarity.")
    print("  FPR = FP / (FP + TN). Lower is better (avoids 'lexical traps').")
    print(f"  Subset: negative pairs with Jaccard > 0.5 (N = {lex_mask.sum()})\n")

    audit2_rows = []
    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in MODELS_TO_EVAL:
            preds = preds_by_technique[technique].get(model_name, {}).get("preds")
            if preds is None:
                continue
            
            subset_preds = preds[lex_mask]
            subset_true = labels[lex_mask]
            tp, fp, tn, fn = confusion_counts(subset_true, subset_preds)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            audit2_rows.append([technique, model_name, int(lex_mask.sum()), fp, tn, f"{fpr:.4f}"])

    print_table(["Technique", "Model", "Subset_N", "FP", "TN", "FPR"], audit2_rows)

    # ==========================================================================
    #                      AUDIT 3 — STRUCTURAL VALIDITY (MAP MATCH)
    # ==========================================================================
    print_header("AUDIT 3 — STRUCTURAL VALIDITY (MAP MATCH)")
    print("\nWhat this means:")
    print("  Correlates model distances with expert ELSST graph shortest paths.")
    print("  Clustering/Seeded: binary distance (0=same cluster, 1=different).")
    print("  Pairwise: binary component, model-graph shortest path, and cosine distance.")
    print("  Uses Spearman rank correlation (ρ). Higher = better structural alignment.\n")

    expert_mask = shortest_path >= 0
    d_expert = shortest_path[expert_mask]
    print(f"  Pairs with valid expert distances: {expert_mask.sum()}\n")

    audit3_rows = []

    # Clustering + Seeded: binary model distance
    for technique in ["clustering", "seeded"]:
        for model_name in MODELS_TO_EVAL:
            preds_info = preds_by_technique[technique].get(model_name)
            if not preds_info:
                continue

            if technique == "clustering":
                cluster_labels = preds_info["cluster_labels"]
                l1 = cluster_labels[preds_info["idx1"]]
                l2 = cluster_labels[preds_info["idx2"]]
                d_model = np.where((l1 != -1) & (l1 == l2), 0, 1)
            else:  # seeded
                term_to_cluster = preds_info["term_to_cluster"]
                c1 = np.array([term_to_cluster.get(t, -1) for t in terms1])
                c2 = np.array([term_to_cluster.get(t, -1) for t in terms2])
                d_model = np.where((c1 != -1) & (c1 == c2), 0, 1)

            rho = spearman_corr(d_expert, d_model[expert_mask])
            audit3_rows.append([technique, model_name, f"binary_dist={rho:.4f}"])

    # Pairwise: multiple correlation metrics
    for model_name in MODELS_TO_EVAL:
        preds_info = preds_by_technique["pairwise"].get(model_name)
        if not preds_info:
            continue
        preds = preds_info["preds"]

        # Binary same-component (union-find)
        edges_mask = preds == 1
        edges_i = idx1[edges_mask]
        edges_j = idx2[edges_mask]
        parent, find, union = union_find(len(unique_terms))
        for a, b in zip(edges_i, edges_j):
            union(a, b)
        roots1 = np.array([find(i) for i in idx1])
        roots2 = np.array([find(i) for i in idx2])
        same_component = (roots1 == roots2).astype(np.int8)
        rho_bin = spearman_corr(d_expert, same_component[expert_mask])

        # Model-graph shortest path
        model_sp = model_graph_shortest_paths(len(unique_terms), edges_i, edges_j, idx1, idx2)
        valid_sp_mask = expert_mask & (model_sp >= 0)
        rho_sp = spearman_corr(shortest_path[valid_sp_mask], model_sp[valid_sp_mask])

        # Cosine distance correlation
        sims = preds_info["similarities"]
        cosine_dist = 1.0 - sims
        rho_cos = spearman_corr(d_expert, cosine_dist[expert_mask])

        audit3_rows.append([
            "pairwise", model_name,
            f"bin_comp={rho_bin:.4f} | graph_sp={rho_sp:.4f} | cos_dist={rho_cos:.4f}"
        ])

    print_table(["Technique", "Model", "Spearman ρ"], audit3_rows)

    # ==========================================================================
    #                      AUDIT 4 — DIFFERENTIAL ITEM FUNCTIONING (RARE WORD)
    # ==========================================================================
    print_header("AUDIT 4 — DIFFERENTIAL ITEM FUNCTIONING (RARE WORD TEST)")
    print("\nWhat this means:")
    print("  Detects bias against technical/rare terminology.")
    print("  ΔRecall = Recall_Common - Recall_Rare. Higher gap = more bias against rare terms.")
    print(f"  Common = top 10% freq ({common_mask.sum()}), Rare = bottom 10% freq ({rare_mask.sum()})\n")

    if zipf_frequency is None:
        print("  SKIPPED: wordfreq library not installed.\n")
    else:
        audit4_rows = []
        for technique in ["clustering", "pairwise", "seeded"]:
            for model_name in MODELS_TO_EVAL:
                preds = preds_by_technique[technique].get(model_name, {}).get("preds")
                if preds is None:
                    continue
                
                recall_common = recall_on_subset(labels, preds, common_mask)
                recall_rare = recall_on_subset(labels, preds, rare_mask)
                gap = recall_common - recall_rare
                
                audit4_rows.append([
                    technique, model_name,
                    f"{recall_common:.4f}", f"{recall_rare:.4f}", f"{gap:.4f}",
                    int(common_mask.sum()), int(rare_mask.sum())
                ])

        print_table(["Technique", "Model", "Recall_Common", "Recall_Rare", "ΔRecall", "N_Common", "N_Rare"], audit4_rows)

    # ==========================================================================
    #                      AUDIT 5 — SEMANTIC DECAY (SEMANTIC GAP TEST)
    # ==========================================================================
    print_header("AUDIT 5 — SEMANTIC DECAY (SEMANTIC GAP TEST)")
    print("\nWhat this means:")
    print("  Tests robustness as keyword overlap vanishes.")
    print("  Slope = (Recall_Hard - Recall_Easy) / ΔDifficulty.")
    print("  Negative slope = performance degrades on harder (no-overlap) pairs.")
    print(f"  Easy = Jaccard > 0.5 ({easy_mask.sum()}), Hard = Jaccard == 0 ({hard_mask.sum()})\n")

    audit5_rows = []
    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in MODELS_TO_EVAL:
            preds = preds_by_technique[technique].get(model_name, {}).get("preds")
            if preds is None:
                continue
            
            recall_easy = recall_on_subset(labels, preds, easy_mask)
            recall_hard = recall_on_subset(labels, preds, hard_mask)
            # ΔDifficulty = 1 (easy=0, hard=1)
            slope = recall_hard - recall_easy
            
            audit5_rows.append([
                technique, model_name,
                f"{recall_easy:.4f}", f"{recall_hard:.4f}", f"{slope:.4f}",
                int(easy_mask.sum()), int(hard_mask.sum())
            ])

    print_table(["Technique", "Model", "Recall_Easy", "Recall_Hard", "Slope", "N_Easy", "N_Hard"], audit5_rows)

    # ==========================================================================
    #                      SUMMARY
    # ==========================================================================
    print_header("EVALUATION COMPLETE")
    print("\nAll models evaluated across 3 techniques with 5 audits.")
    print("Results are printed above in formatted tables.")
    print("No graph loading required — shortest_path data is included in test CSVs.\n")


if __name__ == "__main__":
    main()
