import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

def setup_reproducibility(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def load_and_prepare_data(pos_path, neg_path, seed=42):
    print("Loading datasets...")
    try:
        pos_df = pd.read_csv(pos_path, on_bad_lines='skip')
        neg_df = pd.read_csv(neg_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading files: {e}")
        raise e

    print(f"Positive samples: {len(pos_df)}")
    print(f"Negative samples (total): {len(neg_df)}")

    n_pos = len(pos_df)
    if len(neg_df) > n_pos:
        neg_df_sampled = neg_df.sample(n=n_pos, random_state=seed)
    else:
        neg_df_sampled = neg_df
    print(f"Negative samples (sampled): {len(neg_df_sampled)}")

    df = pd.concat([pos_df, neg_df_sampled])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df['term1'] = df['term1'].astype(str)
    df['term2'] = df['term2'].astype(str)
    
    return df

def load_model(model_name, model_type='sentence_transformer'):
    """
    Load a Sentence Transformer model based on its type.
    
    Args:
        model_name (str): The name/path of the model to load.
        model_type (str): The type of model ('sentence_transformer' or 'token_embedding_mean_pool').
    """
    print(f"Loading Model ({model_name})...")
    try:
        if model_type == 'token_embedding_mean_pool':
            word_embedding_model = models.Transformer(model_name, model_args={'local_files_only': True})
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            model = SentenceTransformer(model_name, model_kwargs={'local_files_only': True})
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None
