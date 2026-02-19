"""
model_utils_shared.py
=====================
Shared utilities for Social Science Concept Integration benchmarking.

Provides:
- Reproducibility setup
- Data loading and preparation
- Model loading (SentenceTransformer / token-level with mean pooling)
- Embedding computation and caching
- Evaluation metrics (precision, recall, F1, confusion matrix)
- Correlation helpers (Spearman, Pearson, Point-Biserial)
- Graph / Union-Find utilities for structural validity
"""

import re
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize

# Optional imports with graceful fallback
try:
    from scipy.stats import spearmanr, pearsonr, pointbiserialr
except ImportError:
    spearmanr = pearsonr = pointbiserialr = None

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

def load_and_prepare_data(pos_path, neg_path, seed=42, balance=False):
    """
    Load positive and negative pair CSVs and return a shuffled DataFrame.

    Args:
        pos_path:  Path to positive pairs CSV.
        neg_path:  Path to negative pairs CSV.
        seed:      Random seed for reproducible shuffling.
        balance:   If True, down-sample negatives to match positives.

    Returns:
        pd.DataFrame with columns [term1, term2, label, ...].
    """
    print("Loading datasets...")
    try:
        pos_df = pd.read_csv(pos_path, on_bad_lines='skip')
        neg_df = pd.read_csv(neg_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading files: {e}")
        raise e

    print(f"Positive samples: {len(pos_df)}")
    print(f"Negative samples (total): {len(neg_df)}")

    if balance:
        n_pos = len(pos_df)
        if len(neg_df) > n_pos:
            neg_df_sampled = neg_df.sample(n=n_pos, random_state=seed)
        else:
            neg_df_sampled = neg_df
        print(f"Negative samples (sampled): {len(neg_df_sampled)}")
    else:
        neg_df_sampled = neg_df
        print(f"Negative samples (used as is): {len(neg_df_sampled)}")

    df = pd.concat([pos_df, neg_df_sampled])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df['term1'] = df['term1'].astype(str)
    df['term2'] = df['term2'].astype(str)
    return df


def load_test_data(pos_path, neg_path):
    """
    Load test datasets keeping positives and negatives separable.

    Returns:
        (pos_df, neg_df, full_df)  — full_df = concat(pos_df, neg_df).
    """
    print("Loading test datasets...")
    pos_df = pd.read_csv(pos_path, on_bad_lines="skip")
    neg_df = pd.read_csv(neg_path, on_bad_lines="skip")

    if "label" not in pos_df.columns:
        pos_df["label"] = 1
    if "label" not in neg_df.columns:
        neg_df["label"] = 0

    for col in ("term1", "term2"):
        pos_df[col] = pos_df[col].astype(str)
        neg_df[col] = neg_df[col].astype(str)

    full_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return pos_df, neg_df, full_df


# ==========================================================================
#  Model Loading
# ==========================================================================

def load_model(model_name, model_type='sentence_transformer'):
    """
    Load a Sentence Transformer model based on its type.

    Args:
        model_name:  HuggingFace model name or local path.
        model_type:  'sentence_transformer' or 'token_embedding_mean_pool'.

    Returns:
        SentenceTransformer instance, or None on failure.
    """
    print(f"Loading Model ({model_name})...")
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
                modules=[word_embedding_model, pooling_model]
            )
        else:
            model = SentenceTransformer(
                model_name, model_kwargs={'local_files_only': False}
            )
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

    Returns dict with keys:
        raw_t     – torch.Tensor (unnormalised)
        raw_np    – np.ndarray   (unnormalised)
        norm_t    – torch.Tensor (L2-normalised)
        norm_np   – np.ndarray   (L2-normalised, float64, C-contiguous)
        sigma     – np.ndarray   (per-dimension std of raw embeddings)
    """
    print(f"  Encoding {len(terms)} unique terms...")
    raw_t = model.encode(
        list(terms), convert_to_tensor=True,
        show_progress_bar=True, batch_size=batch_size,
    )
    raw_t = raw_t.cpu()
    raw_np = raw_t.numpy()

    norm_np = normalize(raw_np, norm="l2", axis=1)
    norm_np = np.ascontiguousarray(norm_np, dtype=np.float64)
    norm_t = F.normalize(raw_t, p=2, dim=1)

    sigma = raw_np.std(axis=0, ddof=0)

    return {
        "raw_t": raw_t,
        "raw_np": raw_np,
        "norm_t": norm_t,
        "norm_np": norm_np,
        "sigma": sigma,
    }


# ==========================================================================
#  Evaluation Metrics
# ==========================================================================

def confusion_counts(y_true, y_pred):
    """Return (TP, FP, TN, FN)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_metrics(y_true, y_pred):
    """Return (precision, recall, F1, TP, FP, TN, FN)."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    return p, r, f, tp, fp, tn, fn


def recall_on_subset(y_true, y_pred, mask):
    """Compute recall on a boolean-masked subset."""
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def compute_subset_metrics(y_true, y_pred, mask):
    """
    Compute comprehensive metrics on a boolean-masked subset using sklearn.
    
    Args:
        y_true: Ground truth labels (full array).
        y_pred: Predicted labels (full array).
        mask: Boolean mask indicating which samples to include.
    
    Returns:
        dict with keys: precision, recall, f1, tp, fp, tn, fn, fpr, n_subset
    """
    y_true_sub = np.asarray(y_true)[mask]
    y_pred_sub = np.asarray(y_pred)[mask]
    
    if len(y_true_sub) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'fpr': 0.0, 'n_subset': 0
        }
    
    # Compute sklearn metrics
    precision = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    recall = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)
    
    # Compute confusion matrix elements
    tp, fp, tn, fn = confusion_counts(y_true_sub, y_pred_sub)
    
    # Compute FPR (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'fpr': float(fpr),
        'n_subset': int(mask.sum())
    }


# ==========================================================================
#  Correlation Helpers
# ==========================================================================

def spearman_corr(x, y):
    """Spearman rank correlation with NaN-safe fallback."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
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


def pearson_corr(x, y):
    """Pearson correlation with NaN-safe handling."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if pearsonr is not None:
        r, p = pearsonr(x[mask], y[mask])
        return float(r), float(p)
    # Fallback
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan, np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), np.nan


def biserial_corr(binary, continuous):
    """
    Biserial correlation between a binary variable and a continuous one.
    Handles imbalanced binary data better than point-biserial.

    Args:
        binary:      array-like of 0/1 values (e.g. same-cluster indicator).
        continuous:  array-like of continuous values (e.g. expert shortest path).

    Returns:
        r_b  or  NaN on failure.
    """
    import scipy.stats as stats
    binary = np.asarray(binary, dtype=float)
    continuous = np.asarray(continuous, dtype=float)
    mask = np.isfinite(binary) & np.isfinite(continuous)
    if mask.sum() < 3:
        return np.nan
    
    b_valid = binary[mask]
    c_valid = continuous[mask]
    
    # Ensure binary is 0 and 1
    unique_vals = np.unique(b_valid)
    if len(unique_vals) != 2:
        return np.nan
        
    val0, val1 = unique_vals[0], unique_vals[1]
    
    group1 = c_valid[b_valid == val1]
    group0 = c_valid[b_valid == val0]
    
    p = len(group1) / len(c_valid)
    q = 1.0 - p
    
    if p == 0 or q == 0:
        return np.nan
        
    mean1 = np.mean(group1)
    mean0 = np.mean(group0)
    sy = np.std(c_valid, ddof=1)
    
    if sy == 0:
        return np.nan
        
    # Find the z-score that splits the normal distribution into p and q
    # stats.norm.ppf(q) gives the z-value where the CDF is q
    z = stats.norm.ppf(q)
    
    # Ordinate (height) of the standard normal distribution at z
    y = stats.norm.pdf(z)
    
    if y == 0:
        return np.nan
        
    rb = ((mean1 - mean0) / sy) * (p * q / y)
    
    # Biserial correlation can sometimes exceed 1 or -1 in extreme cases,
    # but we return the calculated value.
    return float(rb)


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


# ==========================================================================
#  Graph / Union-Find Utilities  (Structural Validity)
# ==========================================================================

def union_find(n):
    """Create a union-find (disjoint set) data structure of size n."""
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


def model_graph_shortest_paths(n_nodes, edges_idx1, edges_idx2,
                               query_idx1, query_idx2):
    """BFS shortest paths on a graph built from predicted positive edges."""
    adj = [set() for _ in range(n_nodes)]
    for a, b in zip(edges_idx1, edges_idx2):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

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


# ==========================================================================
#  ICC / Reliability Helpers
# ==========================================================================

def compute_icc_oneway(sum_y, sum_y2, n_targets, k_raters):
    """
    ICC(1,1): one-way random effects intraclass correlation.
    Measures consistency of per-pair decisions across repeated runs.
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
