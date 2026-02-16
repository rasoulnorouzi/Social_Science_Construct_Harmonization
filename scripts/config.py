import numpy as np

# Random Seed
SEED = 42

# Data Paths for ELSST data and also for APA data
DATA_PATHS = {
    'elsst': 
    {
        'pos_pairs': 'datasets/processed_datasets/elsst/train_positive_pairs.csv',
        'neg_pairs': 'datasets/processed_datasets/elsst/train_negative_pairs.csv',
        'results_dir': 'elsst/results',
        'plots_dir': 'elsst/results/plots',
        'results_csv': 'elsst/results/cv_results.csv'
    }
    ,
    'apa': 
    {
        'pos_pairs': 'datasets/processed_datasets/apa/train_positive_pairs.csv',
        'neg_pairs': 'datasets/processed_datasets/apa/train_negative_pairs.csv',
        'results_dir': 'apa/results',
        'plots_dir': 'apa/results/plots',
        'results_csv': 'apa/results/cv_results.csv'
    }
}

# Model Types Enum-like constants
TYPE_SENTENCE = 'sentence_transformer'
TYPE_TOKEN = 'token_embedding_mean_pool'

# Models Configuration
# You can add more models here.
# 'type': Defines how the model is loaded/pooled.
# 'display_name': Used for plotting legends instead of the raw huggingface ID.
MODELS = {
    'all-mpnet-base-v2': {
        'type': TYPE_SENTENCE,
        'display_name': 'All-MPNet-Base-v2'
    },
    'dwulff/mpnet-personality': {
        'type': TYPE_SENTENCE,
        'display_name': 'MPNet-Personality'
    },
    'allenai/scibert_scivocab_uncased': {
        'type': TYPE_TOKEN,
        'display_name': 'SciBERT (SciVocab)'
    },
    'bert-base-uncased': {
        'type': TYPE_TOKEN,
        'display_name': 'BERT Base'
    }
}

# Analysis Parameters
# Thresholds: Range of cosine similarity thresholds to test
THRESHOLDS = np.arange(0.10, 1.00, 0.01)

# Clustering Parameters (HDBSCAN & UMAP)
CLUSTERING_PARAMS = {
    'min_cluster_sizes': range(2, 12),
    'min_samples': range(2, 12),
    'umap_n_components': [None, 2, 5, 10, 20, 50] # Dimensionality reduction target (None = Original)
}

# Batch size for encoding
BATCH_SIZE = 16

SEED_CLUSTER_PARAMS = {
    'n_initial_seeds_list': [10, 25, 50, 100, 250, 500],
    'n_trials': 10
}


##################### Auditing Parameters #####################

# ── Best HDBSCAN clustering parameters (from training grid search) on ELSST Training set ───────────
BEST_CLUSTERING_PARAMS = {
    "all-mpnet-base-v2":              {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "dwulff/mpnet-personality":       {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "allenai/scibert_scivocab_uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "bert-base-uncased":              {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
}

# ── Best pairwise cosine similarity thresholds (from training grid search) on ELSST Training set ───────────────────────────────
BEST_PAIRWISE_THRESHOLDS = {
    "all-mpnet-base-v2":              0.69,
    "dwulff/mpnet-personality":       0.66,
    "allenai/scibert_scivocab_uncased": 0.89,
    "bert-base-uncased":              0.90,
}

# ── Best seeded clustering parameters (from training grid search) on ELSST Training set ────────────────────────────────────────
BEST_SEEDED_PARAMS = {
    "all-mpnet-base-v2":              {"n_initial_seeds": 250, "threshold": 0.70},
    "dwulff/mpnet-personality":       {"n_initial_seeds": 10,  "threshold": 0.70},
    "allenai/scibert_scivocab_uncased": {"n_initial_seeds": 250, "threshold": 0.85},
    "bert-base-uncased":              {"n_initial_seeds": 250, "threshold": 0.85},
}

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
N_RELIABILITY_RUNS = 10

