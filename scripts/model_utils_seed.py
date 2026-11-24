"""
model_utils_seed.py
===================
Seeded clustering: start from a random set of seed terms and grow clusters by
assigning every other term to the most similar cluster centre, or starting a new
cluster when no centre is similar enough.

Seeds are drawn uniformly at random from all terms; no gold information is used.
The clustering works on term indices and returns one cluster label per term, so
it is evaluated in the same way as HDBSCAN.
"""

import numpy as np
import pandas as pd

from . import model_utils_shared as shared


def sample_initial_seeds(n_terms, n_seeds, random_state=42):
    """
    Draw n_seeds distinct term indices uniformly at random (no gold information).
    Returns (seed_indices, remaining_indices). The remaining terms are processed in a random
    order (same random state), not in the term-list order, which groups synonyms together.
    """
    rng = np.random.default_rng(random_state)
    seed_idx = rng.choice(n_terms, size=min(n_seeds, n_terms), replace=False)
    is_seed = np.zeros(n_terms, dtype=bool)
    is_seed[seed_idx] = True
    remaining_idx = rng.permutation(np.flatnonzero(~is_seed))
    return seed_idx, remaining_idx


def _unit(v):
    """Vector scaled to length 1."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def seed_clustering(norm_emb, seed_idx, remaining_idx, threshold=0.7):
    """
    Grow clusters from the seed terms.

    Args:
        norm_emb:      L2-normalised embeddings (n_terms x dim), numpy array or CPU tensor.
        seed_idx:      term indices used as initial seeds (one cluster each).
        remaining_idx: term indices assigned one by one, in this order.
        threshold:     a term joins the most similar cluster when the cosine similarity
                       with its centre is > threshold; otherwise it starts a new cluster.

    The centre of a cluster is the mean of its members. Its cosine similarity with a term
    only depends on the direction of the mean, so we keep the running sum of the members
    and a unit-length copy of it. Scoring a term is then one matrix-vector product.

    Returns:
        cluster_labels: np.ndarray, one cluster id per term.
    """
    x = norm_emb.numpy() if hasattr(norm_emb, "numpy") else np.asarray(norm_emb)
    x = np.asarray(x, dtype=np.float32)
    n_terms, dim = x.shape
    cluster_labels = np.full(n_terms, -1, dtype=np.int64)

    sums = np.zeros((n_terms, dim), dtype=np.float32)      # running sum of members per cluster
    centres = np.zeros((n_terms, dim), dtype=np.float32)   # unit-length direction of each sum
    n_clusters = 0

    # One cluster per seed
    for i in seed_idx:
        sums[n_clusters] = x[i]
        centres[n_clusters] = _unit(x[i])
        cluster_labels[i] = n_clusters
        n_clusters += 1

    for i in remaining_idx:
        term = _unit(x[i])
        scores = centres[:n_clusters] @ term
        best = int(np.argmax(scores))

        if scores[best] > threshold:
            # Join the best cluster and move its centre
            cluster_labels[i] = best
            sums[best] += x[i]
            centres[best] = _unit(sums[best])
        else:
            # Start a new cluster
            sums[n_clusters] = x[i]
            centres[n_clusters] = term
            cluster_labels[i] = n_clusters
            n_clusters += 1

    return cluster_labels


def evaluate_seed_clustering(cluster_labels, idx1, idx2, labels):
    """
    Evaluate seeded clustering against ground-truth pairs.

    Returns:
        precision, recall, f1, pos_acc, neg_acc
    """
    preds = shared.same_cluster(cluster_labels, idx1, idx2)
    m = shared.binary_metrics(labels, preds)
    return m["precision"], m["recall"], m["f1"], m["recall"], 1.0 - m["fpr"]


def run_seed_clustering_optimization(norm_t, idx1, idx2, labels,
                                     n_initial_seeds=(10, 25, 50, 100, 250, 500),
                                     similarity_threshold=np.arange(0.1, 0.9, 0.05),
                                     random_state=42, num_random_trials=1):
    """
    Grid search over the number of seeds and the similarity threshold. Every setting is
    repeated with num_random_trials different seed draws; the best setting maximises the
    mean F1 over the draws.

    Returns:
        df_results, grouped, best_n_initial, best_threshold, best_f1_mean
    """
    n_terms = norm_t.shape[0]
    print(f"Total unique terms: {n_terms}")

    results = []
    for n_initial in n_initial_seeds:
        for threshold in similarity_threshold:
            threshold_rounded = round(float(threshold), 2)
            for trial in range(num_random_trials):
                trial_random_state = random_state + trial
                seed_idx, remaining_idx = sample_initial_seeds(n_terms, n_initial, trial_random_state)
                cluster_labels = seed_clustering(norm_t, seed_idx, remaining_idx, threshold_rounded)
                precision, recall, f1, pos_acc, neg_acc = evaluate_seed_clustering(
                    cluster_labels, idx1, idx2, labels)
                n_clusters = int(cluster_labels.max()) + 1
                results.append({
                    'threshold': threshold_rounded,
                    'trial': trial + 1,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'pos_acc': pos_acc,
                    'neg_acc': neg_acc,
                    'num_initial_seeds': n_initial,
                    'n_clusters': n_clusters,
                })
                print(f"Seeds: {n_initial:>4}|Threshold: {threshold_rounded:>6.2f}|Trial: {trial+1:>2}|"
                      f"F1: {f1:>7.4f}|Precision: {precision:>7.4f}|Recall: {recall:>7.4f}|"
                      f"Clusters:{n_clusters:>5}")

    df_results = pd.DataFrame(results)

    # Mean over the seed draws of every (number of seeds, threshold) setting
    grouped = df_results.groupby(['num_initial_seeds', 'threshold'])[
        ['precision', 'recall', 'f1', 'pos_acc', 'neg_acc']].mean().reset_index()
    grouped.columns = ['num_initial_seeds', 'threshold', 'precision_mean', 'recall_mean',
                       'f1_mean', 'pos_acc_mean', 'neg_acc_mean']

    best_row = grouped.loc[grouped['f1_mean'].idxmax()]
    return (df_results, grouped, int(best_row['num_initial_seeds']),
            float(best_row['threshold']), float(best_row['f1_mean']))
