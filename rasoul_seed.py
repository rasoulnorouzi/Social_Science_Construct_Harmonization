# %%
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import model_utils_shared  
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
# %%

def sample_unique_terms(n, df, unique_terms, random_state=42):
    """
    Sample n unique terms from df such that no two have the same concept_uri1.
    """
    sampled_terms = []
    sampled_concept_uris = set()
    

    # Shuffle and iterate
    for _, row in df.sample(frac=1, random_state=random_state).iterrows():
        term1 = row['term1']
        concept_uri1 = row.get('concept_uri', None) # Handle missing column safely
        
        # If concept_uri is missing, we treat term itself as unique identifier
        check_val = concept_uri1 if concept_uri1 is not None else term1
        
        if check_val not in sampled_concept_uris:
            sampled_terms.append(term1)
            sampled_concept_uris.add(check_val)
        
        if len(sampled_terms) >= n:
            break
    remaining_terms = unique_terms[~np.isin(unique_terms, sampled_terms)]

    return sampled_terms, remaining_terms
# %%


def seed_clustering(initial_seeds, remaining_terms, term_to_embedding, threshold=0.7):
    current_seeds = list(initial_seeds)
    seeds_cluster = {seed: [seed] for seed in current_seeds}

    seed_embeddings = torch.stack([term_to_embedding[seed] for seed in current_seeds])

    for i, term in enumerate(remaining_terms):
        term_emb = term_to_embedding[term]
        # print(f"term is : {term}")
        # print(f"term_emb.shape: {term_emb.shape}")
        
        # Calculate cosine similarity against ALL current seeds
        scores = util.cos_sim(term_emb, seed_embeddings)[0] 
        # print(f"scores: {scores}")
        # print(f"scores.shape: {scores.shape}")
        
        max_score, max_idx = torch.max(scores, dim=0)
        # print(f"Max score: {max_score}, Max idx: {max_idx}")
        
        if max_score > threshold:
            # Case A: Found a matching cluster
            best_seed_name = current_seeds[max_idx.item()]
            # print(f"Best seed name: {best_seed_name}")
            seeds_cluster[best_seed_name].append(term)
            # print(f"seeds_cluster after adding term: {seeds_cluster}")
            
            # --- DYNAMIC UPDATE ---
            # Re-calculate the centroid (mean) of this cluster
            cluster_terms = seeds_cluster[best_seed_name]
            # print(f"cluster_terms: {cluster_terms}")
            cluster_embs = torch.stack([term_to_embedding[t] for t in cluster_terms])
            # print(f"cluster_embs.shape: {cluster_embs.shape}")
            new_mean_emb = torch.mean(cluster_embs, dim=0)
            # print(f"new_mean_emb.shape: {new_mean_emb.shape}")
            
            # Update the seed_embeddings tensor row
            seed_embeddings[max_idx] = new_mean_emb
            # print(f"Updated seed_embeddings at index {max_idx}")
            
        else:
            # Case B: No match found, create NEW cluster
            current_seeds.append(term)
            seeds_cluster[term] = [term]
            # print(f"seeds_cluster after adding new seed: {seeds_cluster}")
            
            # Append the new seed embedding to our comparison tensor
            seed_embeddings = torch.cat([seed_embeddings, term_emb.unsqueeze(0)], dim=0)
    

    return seeds_cluster


def evaluate_seed_clustering(seeds_cluster, df_pairs, unique_terms):
    """
    Evaluate the clustering results against the ground truth in df.
    df should contain columns 'term1', 'term2', and 'label' (1 for positive pair, 0 for negative).
    """
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    
    # Pre-calculate indices for pairs to speed up evaluation
    valid_pairs_mask = df_pairs['term1'].isin(term_to_idx) & df_pairs['term2'].isin(term_to_idx)
    valid_df = df_pairs[valid_pairs_mask]

    labels_true = valid_df['label'].values

    # Create a mapping from term to cluster ID
    term_to_cluster = {}
    for cluster_id, (seed, terms) in enumerate(seeds_cluster.items()):
        for term in terms:
            term_to_cluster[term] = cluster_id

    # Predict cluster IDs for each term in the pairs
    cluster1_arr = np.array([term_to_cluster.get(term, -1) for term in valid_df['term1']])
    cluster2_arr = np.array([term_to_cluster.get(term, -1) for term in valid_df['term2']])

    # Predict same cluster if both terms have the same cluster ID and it's not -1
    labels_pred = (cluster1_arr == cluster2_arr) & (cluster1_arr != -1) 

    # Calculate metrics
    precision = precision_score(labels_true, labels_pred)
    recall = recall_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred)

    pos_mask = labels_true == 1
    neg_mask = labels_true == 0
    pos_acc = np.sum(labels_pred[pos_mask]) / np.sum(pos_mask) if np.sum(pos_mask) > 0 else 0.0
    neg_acc = np.sum(~labels_pred[neg_mask]) / np.sum(neg_mask) if np.sum(neg_mask) > 0 else 0.0

    
    return precision, recall, f1, pos_acc, neg_acc

def plot_seed_clusters(seeds_cluster, term_to_embedding):
    """
    Plot the distribution of cluster sizes, also scatter plot of clusters.
    """
    cluster_sizes = [len(terms) for terms in seeds_cluster.values()]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cluster_sizes, bins=30, kde=False)
    plt.title("Distribution of Seed Cluster Sizes")
    plt.xlabel("Cluster Size")
    plt.ylabel("Number of Clusters")
    plt.yscale('log')  # Log scale for better visibility
    plt.show()

    # Scatter plot of clusters with umap reduction to 2D
    from umap import UMAP
    all_terms = []
    all_embeddings = []
    for seed, terms in seeds_cluster.items():
        for term in terms:
            all_terms.append(term)
            all_embeddings.append(term_to_embedding[term].cpu().numpy())
    all_embeddings = np.array(all_embeddings)
    reducer = UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(all_embeddings)
    cluster_ids = []
    for seed, terms in seeds_cluster.items():
        cluster_id = list(seeds_cluster.keys()).index(seed)
        cluster_ids.extend([cluster_id] * len(terms))
    cluster_ids = np.array(cluster_ids)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_ids, cmap='tab20', s=10)
    plt.title("Seed Clusters Visualization (UMAP 2D Projection)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.colorbar(scatter, label='Cluster ID')
    plt.show()



# %% Main Execution
def run_seed_clustering_pipeline(df, model, n_initial_seeds=10, similarity_threshold=0.7, random_state=42, num_random_trials=1):
    
    # 3. Prepare Terms and Embeddings
    # Make a unique list of all terms
    terms = pd.unique(df[['term1', 'term2']].values.ravel('K'))

    print(f"Total unique terms: {len(terms)}")

    # Encode all terms at once (Move to GPU if available automatically)
    term_embeddings = model.encode(terms, convert_to_tensor=True, show_progress_bar=True)

    # Create a fast lookup dictionary
    term_to_embedding = {term: embedding for term, embedding in zip(terms, term_embeddings)}
    
    all_results = []
    for trial in range(num_random_trials):
        print(f"\n--- Trial {trial+1}/{num_random_trials} ---")
        #  4. Select Initial Seeds
        initial_seeds, remaining_terms = sample_unique_terms(n_initial_seeds, df, terms, random_state=random_state + trial)
        print(f"Initial seeds: {initial_seeds}")

        #  5. Seed Clustering
        seeds_cluster = seed_clustering(initial_seeds, remaining_terms, term_to_embedding, threshold=similarity_threshold)
        print(f"Number of clusters formed: {len(seeds_cluster)}")

        #  6. Evaluation
        precision, recall, f1, pos_acc, neg_acc = evaluate_seed_clustering(seeds_cluster, df, terms)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, PosAcc: {pos_acc:.4f}, NegAcc: {neg_acc:.4f}")
        
        all_results.append({
            'trial': trial + 1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pos_acc': pos_acc,
            'neg_acc': neg_acc,
            'num_clusters': len(seeds_cluster)
        })
    
    return all_results



# %% 1. Load Data
neg_path = "datasets/processed_datasets/train_negative_pairs.csv"
pos_path = "datasets/processed_datasets/train_positive_pairs.csv"

# Load dataframe
df = model_utils_shared.load_and_prepare_data(pos_path, neg_path, balance=False)

checkpoint = "sentence-transformers/all-mpnet-base-v2"
model = model_utils_shared.load_model(model_name=checkpoint, model_type='sentence_transformer')
# %%
# Run the full pipeline
results = run_seed_clustering_pipeline(df, model, n_initial_seeds=10, similarity_threshold=0.67, random_state=42, num_random_trials=5)
# %%
