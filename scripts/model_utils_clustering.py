import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from umap import UMAP


# ---------------------------------------------------------------------------
# 1. Embedding Preparation
# ---------------------------------------------------------------------------

def get_unique_terms_embeddings(model, df):
    """
    Extracts unique terms and computes their L2-normalised embeddings.
    """
    unique_terms = pd.unique(df[['term1', 'term2']].values.ravel('K'))
    print(f"Computed embeddings for {len(unique_terms)} unique terms.")
    embeddings = model.encode(unique_terms, convert_to_tensor=False, show_progress_bar=False)
    # L2 normalize for cosine similarity simulation with Euclidean distance
    embeddings = normalize(embeddings, norm='l2', axis=1)
    # Ensure standard double precision and contiguous memory for Scikit-Learn/Cython compatibility
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)
    return unique_terms, embeddings


# ---------------------------------------------------------------------------
# 2. Single-Run Building Blocks (mirror model_utils_seed.py style)
# ---------------------------------------------------------------------------

def reduce_embeddings_with_umap(embeddings, n_components=None, random_state=42):
    """
    Reduce embedding dimensionality with UMAP.
    
    Args:
        embeddings: np.ndarray of shape (n_terms, dim).
        n_components: Target dimensionality.  None = skip reduction and
                      return the original embeddings unchanged.
        random_state: Seed for UMAP reproducibility.
    
    Returns:
        reduced_embeddings: np.ndarray of shape (n_terms, display_dim).
        display_dim: The actual dimensionality used (for logging).
    """
    display_dim = n_components if n_components is not None else embeddings.shape[1]

    if n_components is None:
        print(f"Using original embeddings ({display_dim} dims)...")
        reduced_embeddings = embeddings
    else:
        print(f"Reducing to {n_components} dims with UMAP...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
            reducer = UMAP(
                n_neighbors=15,
                n_components=n_components,
                min_dist=0.0,
                metric='cosine',
                random_state=random_state
            )
            reduced_embeddings = reducer.fit_transform(embeddings)

    # Cast for Cython safety – HDBSCAN prefers float64
    reduced_embeddings = np.ascontiguousarray(reduced_embeddings, dtype=np.float64)
    return reduced_embeddings, display_dim


def run_hdbscan_clustering(reduced_embeddings, min_cluster_size=5, min_samples=5):
    """
    Run a single HDBSCAN clustering on (optionally UMAP-reduced) embeddings.
    
    Args:
        reduced_embeddings: np.ndarray, the (reduced) embedding matrix.
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        min_samples: HDBSCAN min_samples parameter.
    
    Returns:
        cluster_labels: np.ndarray of cluster labels (-1 = noise).
    """
    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric='cosine',
        copy=True
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    return cluster_labels


def evaluate_clustering(cluster_labels, df_pairs, unique_terms):
    """
    Evaluate clustering results against ground-truth pairs.
    
    Mirrors evaluate_seed_clustering() from model_utils_seed.py:
    accepts already-computed cluster labels and returns scalar metrics.
    
    Args:
        cluster_labels: np.ndarray of cluster assignments (one per unique term).
        df_pairs: DataFrame with columns 'term1', 'term2', 'label'.
        unique_terms: array-like of unique terms (same order as cluster_labels).
    
    Returns:
        precision, recall, f1, pos_acc, neg_acc, n_clusters, n_noise
    """
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}

    # Filter to pairs where both terms are known
    valid_pairs_mask = df_pairs['term1'].isin(term_to_idx) & df_pairs['term2'].isin(term_to_idx)
    valid_df = df_pairs[valid_pairs_mask]

    idx1_arr = np.array([term_to_idx[t] for t in valid_df['term1']])
    idx2_arr = np.array([term_to_idx[t] for t in valid_df['term2']])
    labels_true = valid_df['label'].values

    # Vectorised prediction
    l1 = cluster_labels[idx1_arr]
    l2 = cluster_labels[idx2_arr]
    preds = ((l1 != -1) & (l1 == l2)).astype(int)

    precision = precision_score(labels_true, preds, zero_division=0)
    recall = recall_score(labels_true, preds, zero_division=0)
    f1 = f1_score(labels_true, preds, zero_division=0)

    pos_mask = labels_true == 1
    neg_mask = labels_true == 0
    pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
    neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))

    return precision, recall, f1, pos_acc, neg_acc, n_clusters, n_noise


# ---------------------------------------------------------------------------
# 3. Optimisation Pipeline (sweeps over parameter grid)
# ---------------------------------------------------------------------------

def run_clustering_optimization(model_display_name, unique_terms, embeddings, df_pairs,
                                min_cluster_sizes, min_samples_list, umap_n_components):
    """
    Sweeps UMAP components, HDBSCAN min_cluster_size and min_samples to
    maximise F1.  Internally calls the single-run building blocks above.
    """
    print(f"Optimizing Clustering for {model_display_name}...")

    best_f1 = -1
    best_params = {}
    results = []

    print(f"{'Dim (UMAP)':<10} | {'Min Size':<10} | {'Min Samp':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10}")
    print("-" * 110)

    for n_components in umap_n_components:
        try:
            reduced_embeddings, display_dim = reduce_embeddings_with_umap(
                embeddings, n_components
            )
        except Exception as e:
            print(f"UMAP FAILED for n_components={n_components}: {e}")
            continue

        for min_size in min_cluster_sizes:
            for min_samp in min_samples_list:
                try:
                    cluster_labels = run_hdbscan_clustering(
                        reduced_embeddings, min_size, min_samp
                    )
                    p, r, f1, pos_acc, neg_acc, n_clust, n_noise = evaluate_clustering(
                        cluster_labels, df_pairs, unique_terms
                    )

                    results.append({
                        'model': model_display_name,
                        'n_components': display_dim,
                        'min_cluster_size': min_size,
                        'min_samples': min_samp,
                        'f1': f1,
                        'precision': p,
                        'recall': r,
                        'pos_acc': pos_acc,
                        'neg_acc': neg_acc,
                        'n_clusters': n_clust,
                        'n_noise': n_noise
                    })

                    print(f"{display_dim:<10} | {min_size:<10} | {min_samp:<10} | {f1:.4f}     | {p:.4f}     | {r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}")

                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            'n_components': display_dim,
                            'min_cluster_size': min_size,
                            'min_samples': min_samp
                        }

                except Exception as e:
                    display_dim_fallback = n_components if n_components is not None else embeddings.shape[1]
                    print(f"{display_dim_fallback:<10} | {min_size:<10} | {min_samp:<10} | HDBSCAN FAILED ({e})")

    print(f"Best F1: {best_f1:.4f} with params {best_params}")
    return results, best_params

def plot_clustering_heatmap(results_df, model_name, plots_dir=None):
    """
    Plots a heatmap of F1 scores for hyperparameter sweep.
    Axes: Min Cluster Size vs. UMAP Dimensions
    """
    print(f"Generating heatmap for {model_name}...")
    # Pivot data for heatmap
    if 'min_samples' in results_df.columns:
        # If min_samples is present, aggregate by max F1 over min_samples
        pivot_df = results_df.pivot_table(index='min_cluster_size', columns='n_components', values='f1', aggfunc='max')
        title_suffix = " (Max F1 over min_samples)"
    else:
        pivot_df = results_df.pivot(index='min_cluster_size', columns='n_components', values='f1')
        title_suffix = ""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': 'F1 Score'})
    plt.title(f'Clustering Performance (UMAP + HDBSCAN)\n{model_name}{title_suffix}', fontsize=14)
    plt.xlabel('UMAP Dimensions')
    plt.ylabel('Min Cluster Size')
    plt.tight_layout()
    
    plots_dir = plots_dir or 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    safe_name = model_name.replace('/', '_').replace('-', '_').replace(' ', '_')
    save_path = os.path.join(plots_dir, f'clustering_heatmap_{safe_name}.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show() # Inline display

def print_best_clustering_settings(results_df):
    """
    Prints summary of best configurations.
    """
    print("\n" + "="*40)
    print(" BEST CLUSTERING SETTINGS PER MODEL ")
    print("="*40)
    
    best_configs = []
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        if model_df.empty:
            continue
            
        best_row_idx = model_df['f1'].idxmax()
        best_row = model_df.loc[best_row_idx]
        
        print(f"\nModel: {model}")
        print(f"  Max F1: {best_row['f1']:.4f}")
        print(f"  Precision: {best_row['precision']:.4f}")
        print(f"  Recall: {best_row['recall']:.4f}")
        print(f"  Pos Acc: {best_row['pos_acc']:.4f}")
        print(f"  Neg Acc: {best_row['neg_acc']:.4f}")
        print(f"  UMAP Dims: {best_row['n_components']}")
        print(f"  Min Cluster Size: {best_row['min_cluster_size']}")
        print(f"  Clusters Found: {best_row['n_clusters']}")
        print(f"  Noise Points: {best_row['n_noise']}")
