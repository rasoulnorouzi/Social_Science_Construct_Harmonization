# %%
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import model_utils_shared  # Assuming this exists in your environment
# %% 1. Load Data
neg_path = "datasets/processed_datasets/train_negative_pairs.csv"
pos_path = "datasets/processed_datasets/train_positive_pairs.csv"

# Load dataframe
df = model_utils_shared.load_and_prepare_data(pos_path, neg_path, balance=False)

# %% 2. Load Model
checkpoint = "sentence-transformers/all-mpnet-base-v2"
model = model_utils_shared.load_model(model_name=checkpoint, model_type='sentence_transformer')

# %% 3. Prepare Terms and Embeddings
# Make a unique list of all terms
terms = pd.unique(df[['term1', 'term2']].values.ravel('K'))

print(f"Total unique terms: {len(terms)}")

# Encode all terms at once (Move to GPU if available automatically)
term_embeddings = model.encode(terms, convert_to_tensor=True, show_progress_bar=True)

# Create a fast lookup dictionary
term_to_embedding = {term: embedding for term, embedding in zip(terms, term_embeddings)}

# %% 4. Select Initial Seeds
def sample_unique_terms(n, df):
    """
    Sample n unique terms from df such that no two have the same concept_uri1.
    """
    sampled_terms = []
    sampled_concept_uris = set()
    
    # Shuffle and iterate
    for _, row in df.sample(frac=1, random_state=42).iterrows():
        term1 = row['term1']
        concept_uri1 = row.get('concept_uri', None) # Handle missing column safely
        
        # If concept_uri is missing, we treat term itself as unique identifier
        check_val = concept_uri1 if concept_uri1 is not None else term1
        
        if check_val not in sampled_concept_uris:
            sampled_terms.append(term1)
            sampled_concept_uris.add(check_val)
        
        if len(sampled_terms) >= n:
            break
    return sampled_terms

# %%

# Get initial seeds
initial_seeds = sample_unique_terms(10, df)
print(f"Initial seeds: {initial_seeds}")

# %%

# Identify remaining terms
remaining_terms = terms[~np.isin(terms, initial_seeds)]

current_seeds = list(initial_seeds)
seeds_cluster = {seed: [seed] for seed in current_seeds}

seed_embeddings = torch.stack([term_to_embedding[seed] for seed in current_seeds])
for i, term in enumerate(remaining_terms):
    term_emb = term_to_embedding[term]
    print(f"term is : {term}")
    print(f"term_emb.shape: {term_emb.shape}")
    
    # Calculate cosine similarity against ALL current seeds
    scores = util.cos_sim(term_emb, seed_embeddings)[0] 
    print(f"scores: {scores}")
    print(f"scores.shape: {scores.shape}")
    
    max_score, max_idx = torch.max(scores, dim=0)
    print(f"Max score: {max_score}, Max idx: {max_idx}")
    
    if max_score > 0.990:
        # Case A: Found a matching cluster
        best_seed_name = current_seeds[max_idx.item()]
        print(f"Best seed name: {best_seed_name}")
        seeds_cluster[best_seed_name].append(term)
        print(f"seeds_cluster after adding term: {seeds_cluster}")
        
        # --- DYNAMIC UPDATE ---
        # Re-calculate the centroid (mean) of this cluster
        cluster_terms = seeds_cluster[best_seed_name]
        print(f"cluster_terms: {cluster_terms}")
        cluster_embs = torch.stack([term_to_embedding[t] for t in cluster_terms])
        print(f"cluster_embs.shape: {cluster_embs.shape}")
        new_mean_emb = torch.mean(cluster_embs, dim=0)
        print(f"new_mean_emb.shape: {new_mean_emb.shape}")
        
        # Update the seed_embeddings tensor row
        seed_embeddings[max_idx] = new_mean_emb
        print(f"Updated seed_embeddings at index {max_idx}")
        
    else:
        # Case B: No match found, create NEW cluster
        current_seeds.append(term)
        seeds_cluster[term] = [term]
        print(f"seeds_cluster after adding new seed: {seeds_cluster}")
        
        # Append the new seed embedding to our comparison tensor
        seed_embeddings = torch.cat([seed_embeddings, term_emb.unsqueeze(0)], dim=0)
        
    print(f"Total clusters so far: {len(seeds_cluster)}")

