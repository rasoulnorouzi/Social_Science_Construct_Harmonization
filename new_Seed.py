import pandas as pd
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import util

# --- 1. Seed Selection Logic ---

def sample_diverse_seeds(df, n_seeds=10, random_state=42):
    """
    Selects n unique terms from df such that no two have the same concept_uri.
    This ensures the initial seeds are NOT synonyms of each other.
    """
    sampled_terms = []
    sampled_concept_uris = set()
    
    # Shuffle dataframe to get random start points
    shuffled_df = df.sample(frac=1, random_state=random_state)
    
    for _, row in shuffled_df.iterrows():
        term = row['term1']
        # Use concept_uri if available to ensure distinct concepts
        # If 'concept_uri' column is missing, we fall back to the term itself
        concept_uri = row.get('concept_uri', term) 
        
        if concept_uri not in sampled_concept_uris:
            sampled_terms.append(term)
            sampled_concept_uris.add(concept_uri)
        
        if len(sampled_terms) >= n_seeds:
            break
            
    return sampled_terms

# --- 2. The Core Clustering Algorithm ---

def fast_seeded_clustering(term_embeddings, terms, threshold=0.75, initial_seeds_count=10, device='cpu'):
    """
    Core implementation of the Seeded/Leader Algorithm using PyTorch.
    
    CRITICAL ASSUMPTION: The input `term_embeddings` and `terms` MUST be sorted 
    such that the first `initial_seeds_count` items are the pre-selected diverse seeds.
    
    Args:
        term_embeddings (np.ndarray or torch.Tensor): Array of shape (n_terms, n_dim).
        terms (np.ndarray): Array of term strings corresponding to embeddings.
        threshold (float): Cosine similarity threshold to merge into a cluster.
        initial_seeds_count (int): Number of seeds at the start of the array.
        device (str): 'cuda' or 'cpu'.

    Returns:
        dict: mapping of {term: cluster_id}
    """
    # Convert inputs to PyTorch tensors
    if isinstance(term_embeddings, np.ndarray):
        all_embeddings = torch.tensor(term_embeddings, dtype=torch.float32).to(device)
    else:
        all_embeddings = term_embeddings.to(device)
    
    n_terms = len(terms)
    
    # Indices of current seeds (The first N items)
    # We clone to allow updating centroids without affecting original data
    seed_embeddings = all_embeddings[:initial_seeds_count].clone()
    
    # Keep track of which cluster each term belongs to. Initialize with -1
    term_cluster_ids = torch.full((n_terms,), -1, dtype=torch.long, device=device)
    
    # Assign initial seeds to their own clusters (0 to N-1)
    for i in range(initial_seeds_count):
        term_cluster_ids[i] = i

    # Track count of members per cluster for mean calculation (dynamic centroid update)
    cluster_counts = torch.ones(initial_seeds_count, device=device)

    # Iterate through remaining terms (The Rest)
    for i in range(initial_seeds_count, n_terms):
        current_emb = all_embeddings[i]
        
        # Calculate similarity against ALL current seeds
        scores = util.cos_sim(current_emb, seed_embeddings)[0]
        max_score, max_idx = torch.max(scores, dim=0)
        
        if max_score > threshold:
            # Case A: Found a matching cluster
            best_cluster_idx = max_idx.item()
            term_cluster_ids[i] = best_cluster_idx
            
            # --- Dynamic Update of Centroid ---
            # Update the seed embedding to be the moving average
            n = cluster_counts[best_cluster_idx]
            seed_embeddings[best_cluster_idx] = (seed_embeddings[best_cluster_idx] * n + current_emb) / (n + 1)
            cluster_counts[best_cluster_idx] += 1
            
        else:
            # Case B: No match found, create NEW cluster
            new_cluster_id = len(seed_embeddings)
            term_cluster_ids[i] = new_cluster_id
            
            # Append new seed
            seed_embeddings = torch.cat([seed_embeddings, current_emb.unsqueeze(0)], dim=0)
            cluster_counts = torch.cat([cluster_counts, torch.tensor([1.0], device=device)])

    # Convert to dictionary for easy mapping
    cluster_ids_cpu = term_cluster_ids.cpu().numpy()
    term_to_cluster_id = {term: int(cid) for term, cid in zip(terms, cluster_ids_cpu)}
    
    return term_to_cluster_id

# --- 3. Evaluation Logic ---

def run_seeded_clustering_optimization(model_display_name, unique_terms, embeddings, df_pairs, initial_seeds, thresholds):
    """
    Sweeps Similarity Thresholds to maximize F1, ensuring initial seeds are processed first.
    """
    print(f"Optimizing Seeded Clustering for {model_display_name}...")
    
    # 1. Reorder Data: Seeds First, Rest Follow
    term_to_emb_idx = {term: i for i, term in enumerate(unique_terms)}
    
    # Identify indices of seeds and non-seeds
    seed_indices = [term_to_emb_idx[s] for s in initial_seeds if s in term_to_emb_idx]
    
    # Create mask for valid seeds (in case some seeds didn't have embeddings computed)
    valid_seeds = [unique_terms[i] for i in seed_indices]
    
    # Identify remaining terms (Set difference)
    seed_set = set(valid_seeds)
    rest_indices = [i for i, t in enumerate(unique_terms) if t not in seed_set]
    
    # Construct ordered arrays
    ordered_indices = seed_indices + rest_indices
    ordered_terms = unique_terms[ordered_indices]
    ordered_embeddings = embeddings[ordered_indices]
    
    num_seeds = len(valid_seeds)
    print(f"Data Reordered: {num_seeds} Diverse Seeds + {len(rest_indices)} Candidates.")
    
    # 2. Setup Validation Data
    # Map original terms to dataframe indices for validation
    term_to_idx = {term: i for i, term in enumerate(unique_terms)} # Use original mapping for validation lookup or create new?
    # Actually, the mapping returned by clustering will use the strings, so we can map dataframe columns directly.
    
    valid_pairs_mask = df_pairs['term1'].isin(term_to_idx) & df_pairs['term2'].isin(term_to_idx)
    valid_df = df_pairs[valid_pairs_mask].copy()
    labels_true = valid_df['label'].values
    
    best_f1 = -1
    best_params = {}
    results = []

    print(f"{'Threshold':<10} | {'F1':<10} | {'Precision':<10} | {'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10} | {'Clusters':<10}")
    print("-" * 95)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Threshold Sweep ---
    for thresh in thresholds:
        try:
            # Run Clustering on ORDERED data
            term_to_cid = fast_seeded_clustering(
                ordered_embeddings, 
                ordered_terms, 
                threshold=thresh, 
                initial_seeds_count=num_seeds,
                device=device
            )
            
            # Generate Predictions
            t1_ids = valid_df['term1'].map(term_to_cid).values
            t2_ids = valid_df['term2'].map(term_to_cid).values
            
            preds = (t1_ids == t2_ids).astype(int)
            
            # Metrics
            p = precision_score(labels_true, preds, zero_division=0)
            r = recall_score(labels_true, preds, zero_division=0)
            f1 = f1_score(labels_true, preds, zero_division=0)
            
            pos_mask = (labels_true == 1)
            neg_mask = (labels_true == 0)
            pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
            neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0
            
            n_clusters = len(set(term_to_cid.values()))

            results.append({
                'model': model_display_name,
                'threshold': thresh,
                'f1': f1,
                'precision': p,
                'recall': r,
                'pos_acc': pos_acc,
                'neg_acc': neg_acc,
                'n_clusters': n_clusters
            })
            
            print(f"{thresh:<10.2f} | {f1:.4f}     | {p:.4f}     | {r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}     | {n_clusters}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'threshold': thresh}
        
        except Exception as e:
            print(f"Clustering Error at thresh={thresh}: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results)
    return results_df, best_params

# --- 4. Visualization ---

def plot_clustering_metrics(results_df, model_name):
    """
    Plots Line Chart: Threshold vs F1/Precision/Recall
    """
    if results_df.empty:
        print("No results to plot.")
        return

    print(f"Generating performance plot for {model_name}...")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x='threshold', y='f1', label='F1 Score', marker='o', linewidth=2.5)
    sns.lineplot(data=results_df, x='threshold', y='precision', label='Precision', marker='s', linestyle='--')
    sns.lineplot(data=results_df, x='threshold', y='recall', label='Recall', marker='^', linestyle='--')
    
    plt.title(f'Seeded Clustering Performance by Threshold\n{model_name}', fontsize=14)
    plt.xlabel('Cosine Similarity Threshold')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    try:
        import model_utils_shared
        
        # 1. Load Data
        neg_path = "datasets/processed_datasets/train_negative_pairs.csv"
        pos_path = "datasets/processed_datasets/train_positive_pairs.csv"
        # Load df (Assuming it returns columns: term1, term2, label, concept_uri)
        df_pairs = model_utils_shared.load_and_prepare_data(pos_path, neg_path, balance=False)
        
        # 2. Select Diverse Seeds (Logic: Distinct concept_uri)
        print("Selecting diverse initial seeds...")
        diverse_seeds = sample_diverse_seeds(df_pairs, n_seeds=20)
        print(f"Selected Seeds: {diverse_seeds}")
        
        # 3. Load Model & Compute Embeddings
        checkpoint = "sentence-transformers/all-mpnet-base-v2"
        model = model_utils_shared.load_model(model_name=checkpoint, model_type='sentence_transformer')
        
        unique_terms = pd.unique(df_pairs[['term1', 'term2']].values.ravel('K'))
        print(f"Encoding {len(unique_terms)} unique terms...")
        embeddings = model.encode(unique_terms, convert_to_tensor=False, show_progress_bar=True)
        
        # 4. Run Optimization with Enforced Seed Order
        thresholds_to_test = np.arange(0.1, 0.96, 0.01)

        results, best = run_seeded_clustering_optimization(
            model_display_name="MPNet + Diverse Seeds",
            unique_terms=unique_terms,
            embeddings=embeddings,
            df_pairs=df_pairs,
            initial_seeds=diverse_seeds,
            thresholds=thresholds_to_test
        )
        
        plot_clustering_metrics(results, "MPNet-Base Seeded Clustering")
        
        print(f"\nOptimization Complete. Best F1: {results['f1'].max():.4f} at Threshold {best['threshold']}")
        
    except ImportError:
        print("Note: 'model_utils_shared' not found. This script requires your local utils to run.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    