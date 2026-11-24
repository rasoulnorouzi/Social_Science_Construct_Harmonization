"""
model_utils_shared.py
=====================
Shared utilities for Social Science Concept Integration benchmarking.

Provides:
- Reproducibility setup
- Data loading (pair files with integer term ids)
- Model loading (SentenceTransformer / token-level with mean pooling)
- Embedding computation, cosine and character n-gram similarity matrices
- Evaluation metrics (precision, recall, F1, FPR) from confusion counts
- Concept-level bootstrap, top-k recall, correct matches per reviewed candidates

Pairs are handled as integer arrays (idx1, idx2 = row numbers in the term list),
because APA has about 60 million negative pairs. Metrics are computed from
confusion counts with numpy (same values as sklearn).
"""

import re
import random

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer, models
from sklearn.preprocessing import normalize

# Optional imports with graceful fallback
try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

try:
    from wordfreq import zipf_frequency
except ImportError:
    zipf_frequency = None


# ==========================================================================
#  Reproducibility
# ==========================================================================

def setup_reproducibility(seed=42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ==========================================================================
#  Data Loading
# ==========================================================================

PAIR_COLUMNS = ["idx1", "idx2", "concept1", "concept2", "label", "shortest_path", "pair_type"]
PAIR_TYPE_NAMES = ["pref_pref", "pref_alt", "alt_alt"]


def _read_column(path, column):
    """One column of a Parquet file as a numpy array (read column by column to save memory)."""
    return pq.read_table(path, columns=[column]).column(column).to_numpy()


def load_pair_data(folder, prefix):
    """
    Load one data set written by scripts/process_dataset.py.

    Returns a dict with:
        terms      – list of term strings; idx1 / idx2 point into this list
        idx1, idx2 – int32 arrays, the two terms of every pair
        concept1, concept2 – int32 arrays, the concepts of the two terms
        label      – int8 array, 1 = synonyms (positive pair), 0 = negative pair
        shortest_path – int16 array, concept distance in the thesaurus graph (-1 = not connected)
        pair_type  – int8 array, 0 = pref-pref, 1 = pref-entry term, 2 = entry-entry term
        n_pos      – number of positive pairs; positives come first, then negatives
        hierarchy  – dict with idx1, idx2, concept1, concept2, depth, pair_type of the
                     broader/narrower pairs (not part of the benchmark)
    """
    folder = str(folder)
    print(f"Loading pair data from {folder} (prefix '{prefix}')...")
    pos_path = f"{folder}/{prefix}positive_pairs.parquet"
    neg_path = f"{folder}/{prefix}negative_pairs.parquet"
    hier_path = f"{folder}/{prefix}hierarchy_pairs.parquet"

    terms = pd.read_parquet(f"{folder}/{prefix}terms.parquet")["term"].tolist()
    data = {"terms": terms}
    for col in PAIR_COLUMNS:
        data[col] = np.concatenate([_read_column(pos_path, col), _read_column(neg_path, col)])
    data["n_pos"] = int(pq.read_metadata(pos_path).num_rows)

    hier_columns = pq.read_schema(hier_path).names
    data["hierarchy"] = {col: _read_column(hier_path, col) for col in hier_columns}
    pa.default_memory_pool().release_unused()

    print(f"  terms: {len(terms):,} | positive pairs: {data['n_pos']:,} | "
          f"negative pairs: {len(data['label']) - data['n_pos']:,} | "
          f"broader/narrower pairs: {len(data['hierarchy']['idx1']):,}")
    return data


def subsample_negatives(data, n_negatives, seed=42):
    """
    Quick test only (FAST_DEBUG switch in the notebooks): keep all
    positives and a random sample of negatives. Never used for reported results.
    """
    rng = np.random.default_rng(seed)
    n_pos = data["n_pos"]
    n_neg = len(data["label"]) - n_pos
    keep_neg = np.sort(rng.choice(n_neg, size=min(n_negatives, n_neg), replace=False)) + n_pos
    keep = np.concatenate([np.arange(n_pos), keep_neg])
    for col in PAIR_COLUMNS:
        data[col] = data[col][keep]
    print(f"  FAST_DEBUG: kept {n_pos:,} positives and {len(keep_neg):,} negatives")
    return data


# ==========================================================================
#  Model Loading
# ==========================================================================

def load_model(model_name, model_type='sentence_transformer', device='cpu'):
    """
    Load a Sentence Transformer model based on its type.

    Args:
        model_name:  HuggingFace model name or local path.
        model_type:  'sentence_transformer' or 'token_embedding_mean_pool'.
        device:      'cuda' or 'cpu' (config.DEVICE).

    Returns:
        SentenceTransformer instance, or None on failure.
    """
    print(f"Loading Model ({model_name}) on {device}...")
    try:
        if model_type == 'token_embedding_mean_pool':
            word_embedding_model = models.Transformer(
                model_name, model_args={'local_files_only': False}
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode='mean',
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], device=device
            )
        else:
            model = SentenceTransformer(
                model_name, model_kwargs={'local_files_only': False}, device=device
            )
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None


# ==========================================================================
#  Embedding Computation
# ==========================================================================

def build_embeddings(model, terms, batch_size=16):
    """
    Encode a list of terms and return a dict with multiple representations.
    All returned arrays live on the CPU.

    Returns dict with keys:
        raw_t     – torch.Tensor (unnormalised)
        raw_np    – np.ndarray   (unnormalised)
        norm_t    – torch.Tensor (L2-normalised)
        norm_np   – np.ndarray   (L2-normalised, float64, C-contiguous)
        sigma     – np.ndarray   (per-dimension std of raw embeddings)
    """
    print(f"  Encoding {len(terms)} unique terms...")
    with torch.no_grad():
        raw_t = model.encode(
            list(terms), convert_to_tensor=True,
            show_progress_bar=True, batch_size=batch_size,
        )
    raw_t = raw_t.float().cpu()
    raw_np = raw_t.numpy()

    norm_np = normalize(raw_np, norm="l2", axis=1)
    norm_np = np.ascontiguousarray(norm_np, dtype=np.float64)
    norm_t = torch.nn.functional.normalize(raw_t, p=2, dim=1)

    sigma = raw_np.std(axis=0, ddof=0)

    return {
        "raw_t": raw_t,
        "raw_np": raw_np,
        "norm_t": norm_t,
        "norm_np": norm_np,
        "sigma": sigma,
    }


def cosine_matrix(norm_np):
    """
    Cosine similarity of every term with every term (CPU, float32).
    The input must be L2-normalised. For APA (11,014 terms) this is about 0.5 GB.
    """
    x = np.ascontiguousarray(norm_np, dtype=np.float32)
    return x @ x.T


def pair_values(matrix, idx1, idx2, chunk=5_000_000):
    """Look up one value per pair in a term x term matrix (in chunks to save memory)."""
    out = np.empty(len(idx1), dtype=matrix.dtype)
    for start in range(0, len(idx1), chunk):
        end = start + chunk
        out[start:end] = matrix[idx1[start:end], idx2[start:end]]
    return out


def above_threshold(scores, threshold):
    """
    Prediction = score > threshold, with the threshold in the same float type
    as the scores. Character n-gram Jaccard values are exact fractions (for example 11/20 = 0.55);
    comparing a float32 0.55 with the Python float 0.55 would wrongly count it as larger.
    """
    scores = np.asarray(scores)
    return scores > np.asarray(threshold, dtype=scores.dtype)


def same_cluster(cluster_labels, idx1, idx2, chunk=5_000_000):
    """1 when both terms are in the same cluster (noise label -1 never matches)."""
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)
    out = np.empty(len(idx1), dtype=np.int8)
    for start in range(0, len(idx1), chunk):
        end = start + chunk
        c1 = cluster_labels[idx1[start:end]]
        c2 = cluster_labels[idx2[start:end]]
        out[start:end] = (c1 == c2) & (c1 != -1)
    return out


# ==========================================================================
#  Evaluation Metrics
# ==========================================================================

def confusion_counts(y_true, y_pred):
    """Return (TP, FP, TN, FN)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    tn = int(len(y_true) - tp - fp - fn)
    return tp, fp, tn, fn


def metrics_from_counts(tp, fp, tn, fn):
    """
    Precision, recall, F1, false-positive rate and MCC from confusion
    counts. Same definitions as sklearn (0 when a denominator is 0).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "mcc": mcc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def binary_metrics(y_true, y_pred):
    """All metrics of one prediction vector (dict)."""
    return metrics_from_counts(*confusion_counts(y_true, y_pred))


def compute_metrics(y_true, y_pred):
    """Return (precision, recall, F1, TP, FP, TN, FN)."""
    m = binary_metrics(y_true, y_pred)
    return m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["tn"], m["fn"]


def recall_on_subset(y_true, y_pred, mask):
    """Compute recall on a boolean-masked subset."""
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return 0.0
    return binary_metrics(y_true, y_pred)["recall"]


def compute_subset_metrics(y_true, y_pred, mask):
    """
    Compute metrics on a boolean-masked subset.

    Returns:
        dict with keys: precision, recall, f1, tp, fp, tn, fn, fpr, n_subset
    """
    y_true_sub = np.asarray(y_true)[mask]
    y_pred_sub = np.asarray(y_pred)[mask]
    if len(y_true_sub) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'fpr': 0.0, 'n_subset': 0}
    m = binary_metrics(y_true_sub, y_pred_sub)
    return {'precision': float(m['precision']), 'recall': float(m['recall']), 'f1': float(m['f1']),
            'tp': m['tp'], 'fp': m['fp'], 'tn': m['tn'], 'fn': m['fn'],
            'fpr': float(m['fpr']), 'n_subset': int(np.sum(mask))}


def f1_at_prevalence(recall, fpr, n_pos_ref, n_neg_ref):
    """
    F1 that a pipeline would reach on a data set
    with n_pos_ref positives and n_neg_ref negatives, keeping its recall and its
    false-positive rate fixed:
        TP = recall * n_pos_ref,  FP = fpr * n_neg_ref,  precision = TP / (TP + FP)
    """
    tp = recall * n_pos_ref
    fp = fpr * n_neg_ref
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if precision + recall == 0:
        return 0.0, precision
    return 2 * precision * recall / (precision + recall), precision


def threshold_curve(scores, labels, thresholds):
    """
    Precision and recall at a grid of thresholds (prediction = score > t).
    Replaces the full precision-recall curve, which would have one row per pair.
    """
    pos = np.sort(scores[labels == 1])
    neg = np.sort(scores[labels == 0])
    tp = len(pos) - np.searchsorted(pos, thresholds, side="right")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="right")
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    recall = tp / max(len(pos), 1)
    return pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall,
                         "tp": tp, "fp": fp})


# ==========================================================================
#  Correlation Helper
# ==========================================================================

def spearman_corr(x, y, max_n=None, seed=42):
    """
    Spearman rank correlation, ignoring NaN values.
    With max_n, a fixed random sample of max_n pairs is used (saves memory on very large inputs).
    """
    x, y = np.asarray(x), np.asarray(y)
    if max_n is not None and len(x) > max_n:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(x), size=max_n, replace=False))
        x, y = x[pick], y[pick]
    x, y = x.astype(float), y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    if spearmanr is not None:
        corr = spearmanr(x[mask], y[mask]).correlation
        return float(corr) if corr is not None else np.nan
    # Fallback without scipy
    xr = pd.Series(x[mask]).rank(method="average").values
    yr = pd.Series(y[mask]).rank(method="average").values
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


# ==========================================================================
#  Lexical / Token Utilities
# ==========================================================================

TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokens(text):
    """Tokenise text into lowercase alphanumeric words."""
    return TOKEN_RE.findall(str(text).lower())


def letter_ngrams(text, n=3):
    """Generate character n-grams for letter-level similarity."""
    text = "".join(c for c in str(text).lower() if c.isalnum())
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_sim(a_tokens, b_tokens):
    """Jaccard similarity between two token/n-gram sets."""
    if not a_tokens and not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union) if union else 0.0


def jaccard_matrix(terms, n=3):
    """
    Character n-gram Jaccard similarity of every term with every term
    (dense float32). Same definition as jaccard_sim(letter_ngrams(a), letter_ngrams(b)),
    computed with one sparse matrix product instead of a Python loop over all pairs.
    Used for the lexical masks and for the character 3-gram baseline.
    """
    vocab = {}
    rows, cols = [], []
    for i, t in enumerate(terms):
        for gram in letter_ngrams(t, n):
            if gram not in vocab:
                vocab[gram] = len(vocab)
            rows.append(i)
            cols.append(vocab[gram])
    x = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(terms), len(vocab)))
    sizes = np.asarray(x.sum(axis=1), dtype=np.float32).ravel()
    jac = (x @ x.T).toarray().astype(np.float32)          # intersection sizes
    union = sizes[:, None] + sizes[None, :] - jac
    # Jaccard = intersection / union, written into the same array to save memory
    np.divide(jac, union, out=jac, where=union > 0)
    jac[union <= 0] = 0.0
    return jac


# ==========================================================================
#  Concept-level bootstrap
#  Pairs are not independent: one concept appears in thousands of pairs. We therefore
#  resample CONCEPTS with replacement. A concept drawn k times counts k times, so a
#  pair inside one concept (positive) gets weight k, and a pair of two different
#  concepts gets weight k1 * k2. Precision, recall and F1 are recomputed from the
#  weighted counts. Only true positives, false positives and false negatives are
#  needed, so the 60 million correct rejections never enter the computation.
#  The pipelines are kept fixed: the interval shows the uncertainty from which
#  concepts happen to be in the test set, not from re-running the clustering.
# ==========================================================================

def bootstrap_weights(concept1, concept2, n_boot=1000, seed=42):
    """
    Draw the bootstrap samples once. Returns (concepts, W): the sorted concept ids of
    the evaluation set and a (n_boot x n_concepts) matrix with how often each concept
    was drawn in each bootstrap sample.
    """
    concepts = np.unique(np.concatenate([concept1, concept2]))
    rng = np.random.default_rng(seed)
    w = np.zeros((n_boot, len(concepts)), dtype=np.float32)
    for b in range(n_boot):
        draw = rng.integers(0, len(concepts), size=len(concepts))
        w[b] = np.bincount(draw, minlength=len(concepts))
    return concepts, w


def bootstrap_weight_sum(w, col1, col2, chunk=5000):
    """
    Sum of the bootstrap weights of a set of pairs, one value per bootstrap sample.
    col1 / col2 are column numbers in W (np.searchsorted(concepts, concept_ids)).
    """
    total = np.zeros(w.shape[0], dtype=np.float64)
    for start in range(0, len(col1), chunk):
        a = col1[start:start + chunk]
        b = col2[start:start + chunk]
        wa = w[:, a]
        weight = np.where(a == b, wa, wa * w[:, b])
        total += weight.sum(axis=1)
    return total


def bootstrap_prf(w, col1, col2, labels, preds):
    """Precision, recall and F1 in every bootstrap sample (arrays of length n_boot)."""
    labels = labels.astype(bool)
    preds = preds.astype(bool)
    tp_rows = np.flatnonzero(labels & preds)
    fp_rows = np.flatnonzero(~labels & preds)
    fn_rows = np.flatnonzero(labels & ~preds)
    tp = bootstrap_weight_sum(w, col1[tp_rows], col2[tp_rows])
    fp = bootstrap_weight_sum(w, col1[fp_rows], col2[fp_rows])
    fn = bootstrap_weight_sum(w, col1[fn_rows], col2[fn_rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def bootstrap_rate(w, col1, col2, subset_mask, hits):
    """
    Share of 'hits' inside a subset of pairs in every bootstrap sample, e.g. the
    false-positive rate on the hard negatives or the recall on the hard positives.
    """
    rows = np.flatnonzero(subset_mask)
    hit_rows = rows[hits[rows].astype(bool)]
    all_w = bootstrap_weight_sum(w, col1[rows], col2[rows])
    hit_w = bootstrap_weight_sum(w, col1[hit_rows], col2[hit_rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(all_w > 0, hit_w / all_w, np.nan)


def percentile_interval(samples, level=0.95):
    """Percentile confidence interval of bootstrap samples."""
    low = (1 - level) / 2 * 100
    return float(np.nanpercentile(samples, low)), float(np.nanpercentile(samples, 100 - low))


# ==========================================================================
#  Top-k recall and correct matches per reviewed candidates
# ==========================================================================

def topk_recall(sim, query_idx, target_idx, k_list, chunk=1000):
    """
    For every ordered synonym pair (query term -> synonym): is the synonym among the k nearest
    neighbours of the query (the query itself excluded)? With ties, the hit counts as the chance
    that a random order of the tied terms puts the synonym inside the top k.
    Returns {k: mean hit over all synonym pairs}.
    Each positive pair is used in both directions (term1 -> term2 and term2 -> term1).
    """
    n_greater = np.empty(len(query_idx), dtype=np.int64)
    n_ties = np.empty(len(query_idx), dtype=np.int64)
    for start in range(0, len(query_idx), chunk):
        q = query_idx[start:start + chunk]
        t = target_idx[start:start + chunk]
        rows = np.array(sim[q], dtype=np.float32)          # copy: (chunk, n_terms)
        rows[np.arange(len(q)), q] = -np.inf               # the query itself is not a candidate
        target_score = rows[np.arange(len(q)), t][:, None]
        greater = (rows > target_score).sum(axis=1)
        ties = (rows == target_score).sum(axis=1) - 1      # minus the synonym itself
        n_greater[start:start + chunk] = greater
        n_ties[start:start + chunk] = ties
    # The synonym takes one of the places greater+1 ... greater+ties+1, each equally likely.
    return {k: float(np.mean(np.clip((k - n_greater) / (n_ties + 1), 0, 1))) for k in k_list}


def correct_in_top(scores, labels, budgets, seed=42):
    """
    Rank all pairs by score (highest first) and count the synonym pairs among the first
    N pairs, for every N in budgets. Ties at the cut-off are broken at random (fixed seed).
    Returns {N: number of correct matches among the top N}.
    """
    n_top = max(budgets)
    kth_score = np.partition(scores, len(scores) - n_top)[len(scores) - n_top]
    candidates = np.flatnonzero(scores >= kth_score)
    rng = np.random.default_rng(seed)
    order = candidates[np.lexsort((rng.random(len(candidates)), -scores[candidates]))]
    return {n: int(labels[order[:n]].sum()) for n in budgets}
