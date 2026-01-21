# Copilot Instructions for SocialScience-ConceptIntegration

## Project Overview
This project benchmarks machine learning models for concept harmonization in social science, merging heterogeneous terms into unified constructs. It uses a factorial design to compare multiple vector representation models and harmonization strategies, with rigorous evaluation and bias analysis.

## Architecture & Data Flow
- **Data**: Processed datasets are in `datasets/processed_datasets/` (positive/negative train/test pairs). Raw data is in `datasets/raw_datasets/`.
- **Core Scripts**:
  - `analysis.py`: Main benchmarking pipeline. Loads data, samples, encodes terms, evaluates models, optimizes thresholds, and generates performance plots.
  - `model_utils_clustering.py`: Clustering-based harmonization using UMAP and HDBSCAN. Sweeps parameters for optimal F1, outputs results and plots.
  - `model_utils_pairwise.py`: Pairwise similarity evaluation, cosine similarity, and threshold optimization.
  - `config.py`: Centralized configuration for paths, model definitions, and conventions.
  - `debug_hdbscan.py`: Synthetic tests for HDBSCAN clustering/debugging.
- **Results**: Outputs are saved in `results/` (CSV summaries, plots in `results/plots/`).

## Developer Workflows
- **Run Main Analysis**: Execute `analysis.py` to benchmark models and generate plots. Results are saved to `results/cv_results.csv` and `results/plots/`.
- **Clustering Experiments**: Use `model_utils_clustering.py` for UMAP/HDBSCAN sweeps. Adjust parameters in script or via config.
- **Debugging**: Use `debug_hdbscan.py` for isolated clustering tests.
- **Testing**: No formal test suite; use `test.py` and Jupyter notebooks for ad-hoc validation and exploration.
- **Environment**: Python 3.13, dependencies managed via `myenv/`. Models loaded from HuggingFace (`sentence_transformers`, `transformers`).

## Project-Specific Conventions
- **Model Definitions**: All model configs are in `config.py` (`MODELS` dict). Add new models here for consistent usage.
- **Embeddings**: Always L2-normalize embeddings before clustering. Use mean pooling for token models.
- **Threshold Optimization**: Evaluate metrics across thresholds (0.10–0.99) for each model; best threshold is reported and plotted.
- **Reproducibility**: Set `SEED = 42` for all random operations.
- **Plots**: Save with model names sanitized (slashes/dashes replaced by underscores).

## Integration Points & Dependencies
- **External Models**: HuggingFace models (`sentence_transformers`, `transformers`).
- **Clustering**: UMAP for dimensionality reduction, HDBSCAN for clustering.
- **Metrics**: Precision, recall, F1, and class-specific accuracy.

## Examples & Patterns
- To add a new model: update `MODELS` in `config.py`, then reference in analysis scripts.
- To run clustering: call `run_clustering_optimization()` in `model_utils_clustering.py`.
- To debug clustering: run `debug_hdbscan.py` directly.

## Key Files & Directories
- `analysis.py`, `model_utils_clustering.py`, `model_utils_pairwise.py`, `config.py`
- `datasets/processed_datasets/`, `results/`, `results/plots/`

---
**If any section is unclear or missing, please provide feedback for further refinement.**
