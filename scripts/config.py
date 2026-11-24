import os
import numpy as np
import torch
from pathlib import Path

# Project Root - Works on any machine/OS
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Random Seed
SEED = 42

# Device for every notebook: "cuda" when a GPU is available,
# otherwise "cpu". Set the environment variable HARMONIZATION_DEVICE=cpu to force the CPU.
# The GPU is only used to encode terms with the language models. All similarity, clustering
# and metric computations run on the CPU, so the results do not depend on the device
# (up to the small rounding differences of the encoder itself).
DEVICE = os.environ.get("HARMONIZATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Pair files are Parquet files with integer term ids (see scripts/process_dataset.py).
# A data set is a folder plus a file prefix.
PROCESSED_DIR = PROJECT_ROOT / "datasets/processed_datasets"
DATASETS = {
    "elsst_train": {"folder": PROCESSED_DIR / "elsst", "prefix": "train_"},  # calibration only
    "elsst_test":  {"folder": PROCESSED_DIR / "elsst", "prefix": "test_"},   # held-out evaluation
    "apa":         {"folder": PROCESSED_DIR / "apa",   "prefix": ""},        # unseen thesaurus, full set
}

# Output folders
RESULTS_DIRS = {
    "elsst": PROJECT_ROOT / "analysis/results/elsst",
    "apa": PROJECT_ROOT / "analysis/results/apa",
}
CALIBRATION_DIR = PROJECT_ROOT / "analysis/results/elsst/callibrations_results"
CALIBRATION_PLOTS_DIR = PROJECT_ROOT / "analysis/results/elsst/plots"

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

# Character n-gram baseline: no language model,
# only the overlap of letter trigrams (Jaccard). It is used as a pairwise method and its
# threshold is calibrated on the ELSST train split like the thresholds of the models.
BASELINE_KEY = 'char-3gram'
BASELINE_DISPLAY_NAME = 'Character 3-gram (baseline)'
NGRAM_N = 3

# Batch size for encoding
BATCH_SIZE = 16

# Analysis Parameters
# Thresholds: Range of cosine similarity thresholds to test
PAIRWISE_PARAMS = {
    'thresholds': np.arange(0.10, 1.00, 0.01)
}

# Thresholds tested for the character 3-gram baseline (Jaccard scale 0 to 1)
NGRAM_PARAMS = {
    'thresholds': np.arange(0.05, 1.00, 0.01)
}

# Clustering Parameters (HDBSCAN & UMAP)
CLUSTERING_PARAMS = {
    'min_cluster_sizes': range(2, 12),
    'min_samples': range(2, 12),
    'umap_n_components': [None, 2, 5, 10, 20, 50] # Dimensionality reduction target (None = Original)
}



SEED_CLUSTER_PARAMS = {
    'n_initial_seeds_list': [10, 25, 50, 100, 250, 500],
    'n_trials': 10,
    'thresholds': np.arange(0.10, 1.00, 0.05)
}


##################### Auditing Parameters #####################

# The BEST_* values below are printed at the end of the three calibration notebooks
# (pairwise_analysis, clustering_analysis, seed_clustering_analysis) and copied here.
# The diagnostic notebooks compare them with the calibration CSV files and stop when they differ.

# ── Best HDBSCAN clustering parameters (from training grid search) on ELSST Training set ───────────
BEST_CLUSTERING_PARAMS = {
    "all-mpnet-base-v2":              {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "dwulff/mpnet-personality":       {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "allenai/scibert_scivocab_uncased": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
    "bert-base-uncased":              {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
}

# ── Best pairwise cosine similarity thresholds (from training grid search) on ELSST Training set ───────────────────────────────
BEST_PAIRWISE_THRESHOLDS = {
    "all-mpnet-base-v2":              0.74,
    "dwulff/mpnet-personality":       0.71,
    "allenai/scibert_scivocab_uncased": 0.90,
    "bert-base-uncased":              0.91,
}

# Best Jaccard threshold of the character 3-gram baseline (from pairwise_analysis.ipynb)
BEST_NGRAM_THRESHOLD = 0.45

# ── Best seeded clustering parameters (from training grid search) on ELSST Training set ────────────────────────────────────────
BEST_SEEDED_PARAMS = {
    "all-mpnet-base-v2":              {"n_initial_seeds": 250, "threshold": 0.75},
    "dwulff/mpnet-personality":       {"n_initial_seeds": 10,  "threshold": 0.75},
    "allenai/scibert_scivocab_uncased": {"n_initial_seeds": 500, "threshold": 0.90},
    "bert-base-uncased":              {"n_initial_seeds": 500, "threshold": 0.90},
}

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
N_RELIABILITY_RUNS = 10

# Seeded clustering depends on
# which terms are drawn as seeds. The final evaluation repeats it with this many seed draws
# and reports the mean and SD; the diagnostics use the draw with the median F1.
N_SEED_DRAWS = 10

# Number of concept-level bootstrap resamples for the confidence intervals.
N_BOOTSTRAP = 1000

# Reference prevalence for "F1 at a common
# prevalence": the ELSST test split, 1,557 positive and 1,754,671 negative pairs.
# The ELSST diagnostic notebook checks that these numbers match its data.
REFERENCE_N_POS = 1557
REFERENCE_N_NEG = 1754671

# Top-k recall and correct matches per reviewed candidates.
TOPK_LIST = [1, 5, 10]
REVIEW_BUDGETS = [100, 500, 1000]
