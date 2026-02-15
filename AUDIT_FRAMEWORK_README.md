# Psychometric Audit Framework for Concept Harmonisation

This framework provides a **clean, reusable, and abstracted** system for evaluating machine learning models on concept harmonisation tasks. The code has been refactored from the original `final_testing.ipynb` notebook into modular, reusable components.

---

## 📁 File Structure

```
├── audit_utility.py              # Core audit functions (5 psychometric audits)
├── abstracted_test_suite.ipynb   # Clean, abstracted notebook template
├── final_testing.ipynb           # Original notebook (for reference)
├── model_utils_shared.py         # Shared utilities
├── model_utils_clustering.py     # Clustering technique
├── model_utils_pairwise.py       # Pairwise technique
├── model_utils_seed.py           # Seeded clustering technique
└── config.py                     # Model configurations
```

---

## 🚀 Quick Start

### 1. Prepare Your Dataset

Your dataset should consist of two CSV files:
- **Positive pairs**: `test_positive_pairs.csv` with columns `term1`, `term2`, `label` (=1)
- **Negative pairs**: `test_negative_pairs.csv` with columns `term1`, `term2`, `label` (=0)

Optional columns:
- `shortest_path`: Expert graph distance (for Audit 3 - Structural Validity)

### 2. Configure the Notebook

Open `abstracted_test_suite.ipynb` and edit **Section 1 - Configuration**:

```python
# Dataset paths
TEST_POS_PATH = "path/to/your/test_positive_pairs.csv"
TEST_NEG_PATH = "path/to/your/test_negative_pairs.csv"

# Models to evaluate
MODELS_TO_EVAL = [
    "all-mpnet-base-v2",
    "your-model-name",
]

# Best hyperparameters (from your cross-validation)
BEST_CLUSTERING_PARAMS = {
    "your-model-name": {"n_components": 768, "min_cluster_size": 2, "min_samples": 2},
}

BEST_PAIRWISE_THRESHOLDS = {
    "your-model-name": 0.70,
}

BEST_SEEDED_PARAMS = {
    "your-model-name": {"n_initial_seeds": 250, "threshold": 0.70},
}
```

### 3. Run the Notebook

Execute all cells in order. The notebook will:
1. Load your data
2. Build embeddings for each model
3. Run three harmonisation techniques (Clustering, Pairwise, Seeded)
4. Execute all five psychometric audits
5. Generate comprehensive reports
6. Save results to CSV files

---

## 📊 The Five Psychometric Audits

| # | Audit | What it Measures | Key Metric |
|---|-------|------------------|------------|
| **1** | **Reliability** | Model stability under stochastic noise | SEM (Standard Error of Model) |
| **2** | **Discriminant Validity** | False-positive rate on lexically similar negatives | FPR on Jaccard > 0.5 pairs |
| **3** | **Structural Validity** | Correlation with expert graph distances | Spearman ρ, Pearson r, Point-Biserial r |
| **4** | **DIF (Rare-Word Bias)** | Performance gap on rare vs. common terms | ΔRecall (Common - Rare) |
| **5** | **Semantic Decay** | Recall degradation on purely semantic pairs | Slope (Hard - Easy) |

---

## 🔧 Using the Audit Utility Module

You can also use the audit functions programmatically in your own scripts:

```python
import audit_utility as audit

# Precompute audit masks
masks = audit.precompute_audit_masks(full_df, pos_df, neg_df, unique_terms)

# Run individual audits
audit1_df = audit.run_reliability_audit(
    models_to_eval, embedding_cache, model_configs,
    idx1, idx2, idx1_t, idx2_t, labels,
    unique_terms, full_df, terms1, terms2,
    best_clustering_params, best_pairwise_thresholds, best_seeded_params
)

# Or run all audits at once
audit_results = audit.run_all_audits(
    models_to_eval=models_to_eval,
    embedding_cache=embedding_cache,
    preds_store=preds_store,
    model_configs=model_configs,
    # ... (see abstracted_test_suite.ipynb for full parameter list)
)
```

---

## 📦 Output Files

After running the notebook, you'll find these files in the `results/` directory:

```
results/
├── test_summary.csv                      # Overall F1 scores
├── audit1_reliability.csv                # Reliability audit results
├── audit2_discriminant_validity.csv      # Discriminant validity results
├── audit3_structural_validity.csv        # Structural validity correlations
├── audit4_dif.csv                        # DIF (rare-word bias) results
└── audit5_semantic_decay.csv             # Semantic decay results
```

---

## 🔄 Migrating from `final_testing.ipynb`

If you're using the original `final_testing.ipynb` notebook:

### Before (Hard-coded, dataset-specific)
```python
# Lots of code embedded in cells
# Hard to reuse for different datasets
# Difficult to maintain
```

### After (Abstracted, reusable)
```python
# Configure once
TEST_POS_PATH = "your/dataset/positive_pairs.csv"
MODELS_TO_EVAL = ["your-model"]
BEST_PAIRWISE_THRESHOLDS = {"your-model": 0.70}

# Run audits in one line
audit_results = audit.run_all_audits(...)
```

### Key Benefits
✅ **Reusable**: Works with any dataset with minimal configuration
✅ **Maintainable**: Core logic in modules, not scattered across cells
✅ **Extensible**: Easy to add new audits or techniques
✅ **Clean**: Notebook focuses on workflow, not implementation details

---

## 🧪 Testing the Framework

To verify the framework works correctly with your existing dataset:

```bash
# Run the abstracted notebook
jupyter notebook abstracted_test_suite.ipynb

# Compare results with final_testing.ipynb
# (Results should be identical for the same configuration)
```

---

## 🛠️ Advanced Customization

### Adding Custom Audits

1. Add your audit function to `audit_utility.py`:

```python
def run_custom_audit(models_to_eval, preds_store, ...):
    """Your custom audit logic."""
    audit_records = []
    # ... implement your audit
    return pd.DataFrame(audit_records)
```

2. Call it in the notebook:

```python
custom_results = audit.run_custom_audit(
    models_to_eval, preds_store, ...
)
display(custom_results)
```

### Changing Audit Parameters

Edit the audit parameters in Section 1 of the notebook:

```python
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]  # For Audit 1
N_RELIABILITY_RUNS = 5                      # Number of runs per noise level
```

---

## 📝 Function Reference

### `audit_utility.py`

#### Core Functions

- **`precompute_audit_masks()`**: Build boolean masks for audit subsets
- **`run_reliability_audit()`**: Audit 1 - SEM under stochastic noise
- **`run_discriminant_validity_audit()`**: Audit 2 - Lexical trap FPR
- **`run_structural_validity_audit()`**: Audit 3 - Expert graph correlation
- **`run_dif_audit()`**: Audit 4 - Rare-word bias
- **`run_semantic_decay_audit()`**: Audit 5 - Semantic gap degradation
- **`run_all_audits()`**: Execute all five audits in sequence

#### Parameters

All audit functions share similar parameter structures:

```python
models_to_eval       # List of model names to evaluate
embedding_cache      # Dict: model_name → embedding dict
preds_store          # Dict: technique → model_name → predictions
model_configs        # Dict: model_name → config (with "display_name")
labels               # Ground truth labels (numpy array)
idx1, idx2           # Pair indices (numpy arrays)
idx1_t, idx2_t       # Pair indices (torch tensors)
unique_terms         # List of unique terms
terms1, terms2       # Lists of term strings
full_df, pos_df, neg_df  # DataFrames
shortest_path        # Expert graph distances (for Audit 3)
best_*_params        # Hyperparameters for each technique
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `wordfreq` not installed (Audit 4 skipped)
```bash
pip install wordfreq
```

**Issue**: CUDA out of memory
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU-only mode (automatically detected)

**Issue**: Results differ from `final_testing.ipynb`
- Verify random seed is the same
- Check hyperparameters match exactly
- Ensure dataset paths are correct

---

## 📄 License & Citation

If you use this framework in your research, please cite:

```bibtex
@software{concept_harmonisation_audit_framework,
  title={Psychometric Audit Framework for Concept Harmonisation},
  year={2026},
  version={1.0}
}
```

---

## 🤝 Contributing

To extend or improve this framework:

1. Add new audit functions to `audit_utility.py`
2. Update `abstracted_test_suite.ipynb` to call them
3. Document your changes in this README

---

## 📞 Support

For questions or issues:
- Check the original `final_testing.ipynb` for implementation details
- Review function docstrings in `audit_utility.py`
- Examine the model utilities in `model_utils_*.py`

---

**Happy Auditing! 🎯**
