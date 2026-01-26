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

def get_unique_terms_embeddings(model, df):
    """
    Extracts unique terms and computes their embeddings.
    """
    unique_terms = pd.unique(df[['term1', 'term2']].values.ravel('K'))
    print(f"Computed embeddings for {len(unique_terms)} unique terms.")
    embeddings = model.encode(unique_terms, convert_to_tensor=False, show_progress_bar=False)
    # L2 normalize for cosine similarity simulation with Euclidean distance
    embeddings = normalize(embeddings, norm='l2', axis=1)
    # Ensure standard double precision and contiguous memory for Scikit-Learn/Cython compatibility
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)
    return unique_terms, embeddings

def run_clustering_optimization(model_display_name, unique_terms, embeddings, df_pairs, min_cluster_sizes, min_samples, umap_n_components):
    """
    Sweeps UMAP components and HDBSCAN min_cluster_size to maximize F1.
    """
    print(f"Optimizing Clustering for {model_display_name}...")
    
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    
    # Pre-calculate indices for pairs to speed up evaluation
    valid_pairs_mask = df_pairs['term1'].isin(term_to_idx) & df_pairs['term2'].isin(term_to_idx)
    valid_df = df_pairs[valid_pairs_mask]
    
    # Vectorization setup: Create arrays of indices for term1 and term2
    idx1_arr = np.array([term_to_idx[t] for t in valid_df['term1']])
    idx2_arr = np.array([term_to_idx[t] for t in valid_df['term2']])
    
    labels_true = valid_df['label'].values
    
    best_f1 = -1
    best_params = {}
    results = []

    print(f"{'Dim (UMAP)':<10} | {'Min Size':<10} | {'Min Samp':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10}")
    print("-" * 110)

    # Cache UMAP reductions to avoid re-computing if we were looping differently, 
    # but here we loop dim -> size, so we compute UMAP once per dim.
    
    for n_components in umap_n_components:
        # Determine actual running dimension for display/logging
        display_dim = n_components if n_components is not None else embeddings.shape[1]

        try:
            if n_components is None:
                # Use original embeddings without reduction
                print(f"Using Original Embeddings ({display_dim} dims)...")
                reduced_embeddings = embeddings
            else:
                # Reduce dimensionality with UMAP
                # metric='cosine' is often preferred for semantic embeddings
                # n_neighbors=15 is standard
                print(f"Reducing to {n_components} dims with UMAP...")
                
                # Suppress UMAP warning about n_jobs and random_state
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")
                    reducer = UMAP(
                        n_neighbors=15, 
                        n_components=n_components, 
                        min_dist=0.0, 
                        metric='cosine', 
                        random_state=42
                    )
                    reduced_embeddings = reducer.fit_transform(embeddings)

            # Cast for Cython safety (though float32 is usually fine for UMAP output, HDBSCAN likes float64)
            reduced_embeddings = np.ascontiguousarray(reduced_embeddings, dtype=np.float64)

            for min_size in min_cluster_sizes:
                for min_samp in min_samples:
                    try:
                        # Run HDBSCAN on reduced data
                        # metric='cosine' to match embedding space
                        # cluster_selection_epsilon=0.0 (default)
                        clusterer = HDBSCAN(
                            min_cluster_size=int(min_size), 
                            min_samples=int(min_samp),
                            metric='cosine', 
                            copy=True
                        )
                        cluster_labels = clusterer.fit_predict(reduced_embeddings)
                        
                        # Predict labels for pairs (Vectorized)
                        l1 = cluster_labels[idx1_arr]
                        l2 = cluster_labels[idx2_arr]
                        
                        # Use vectorized comparison
                        # HDBSCAN noise is -1. Points are "in the same cluster" if labels match and != -1
                        preds = ((l1 != -1) & (l1 == l2)).astype(int)
                        
                        p = precision_score(labels_true, preds, zero_division=0)
                        r = recall_score(labels_true, preds, zero_division=0)
                        f1 = f1_score(labels_true, preds, zero_division=0)
                        
                        pos_mask = (labels_true == 1)
                        neg_mask = (labels_true == 0)
                        
                        pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
                        neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0
                        
                        # Note: Storing epsilon as 'Default' for clarity in logs/csv
                        results.append({
                            'model': model_display_name,
                            'n_components': display_dim, # store actual int dim
                            'min_cluster_size': min_size,
                            'min_samples': min_samp,
                            'f1': f1,
                            'precision': p,
                            'recall': r,
                            'pos_acc': pos_acc,
                            'neg_acc': neg_acc,
                            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                            'n_noise': np.sum(cluster_labels == -1)
                        })
                        
                        print(f"{display_dim:<10} | {min_size:<10} | {min_samp:<10} | {f1:.4f}     | {p:.4f}     | {r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}")
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = {'n_components': display_dim, 'min_cluster_size': min_size, 'min_samples': min_samp}
                    
                    except Exception as e:
                        print(f"{display_dim:<10} | {min_size:<10} | {min_samp:<10} | HDBSCAN FAILED ({e})")

        except Exception as e:
            print(f"UMAP FAILED for n_components={n_components}: {e}")
    
    print(f"Best F1: {best_f1:.4f} with params {best_params}")
    return results, best_params

def plot_clustering_heatmap(results_df, model_name):
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
    
    plots_dir = 'results/plots'
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
