import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef


# ---------------------------------------------------------------------------
# 1. Embedding Preparation
# ---------------------------------------------------------------------------

def compute_similarities(term_embeddings, term_to_idx, terms1, terms2, sim_batch_size=32768):
    """
    Compute cosine similarities for pairs using pre-computed, L2-normalised
    embeddings and an index lookup map.

    Args:
        term_embeddings: torch.Tensor of shape (n_unique_terms, dim), L2-normalised.
        term_to_idx:     dict mapping each unique term to its row index in
                         *term_embeddings*.
        terms1:          list of first terms in each pair.
        terms2:          list of second terms in each pair.
        sim_batch_size:  Number of pairs to process per batch to avoid OOM
                         errors (default 32 768).

    Returns:
        np.ndarray of cosine similarities, one per pair.
    """
    idx1 = torch.tensor([term_to_idx[t] for t in terms1], dtype=torch.long)
    idx2 = torch.tensor([term_to_idx[t] for t in terms2], dtype=torch.long)

    n = len(terms1)
    chunks = []
    for start in range(0, n, sim_batch_size):
        end = min(start + sim_batch_size, n)
        chunk = torch.nn.functional.cosine_similarity(
            term_embeddings[idx1[start:end]], term_embeddings[idx2[start:end]], dim=1
        ).cpu().numpy()
        chunks.append(chunk)

    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# 2. Single-Run Evaluation (mirror model_utils_seed.py style)
# ---------------------------------------------------------------------------

def evaluate_pairwise(similarities, labels, threshold):
    """
    Evaluate pairwise predictions at a **single** cosine-similarity threshold.

    Args:
        similarities: np.ndarray of cosine similarities (one per pair).
        labels:       np.ndarray of ground-truth labels (1 / 0).
        threshold:    float, cosine similarity cut-off.

    Returns:
        precision, recall, f1, pos_acc, neg_acc, mcc
    """
    preds = (similarities > threshold).astype(int)

    p = precision_score(labels, preds, zero_division=0)
    r = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds) if len(set(preds)) > 1 else 0.0

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_acc = np.mean(preds[pos_mask] == 1) if np.any(pos_mask) else 0.0
    neg_acc = np.mean(preds[neg_mask] == 0) if np.any(neg_mask) else 0.0

    return p, r, f1, pos_acc, neg_acc, mcc


# ---------------------------------------------------------------------------
# 3. Optimisation Pipeline (sweep thresholds)
# ---------------------------------------------------------------------------

def run_pairwise_optimization(model_name, model, df, batch_size=16,
                              thresholds=np.arange(0.10, 1.00, 0.01)):
    """
    Sweep cosine-similarity thresholds and collect metrics for each.
    Internally calls :func:`evaluate_pairwise` per threshold.

    Returns:
        results:             list[dict] – one entry per threshold.
        best_threshold:      float – threshold that maximised F1.
    """
    terms1 = df['term1'].tolist()
    terms2 = df['term2'].tolist()
    unique_terms = list(set(terms1) | set(terms2))
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}

    print(f"\n--- Running pairwise optimisation for {model_name} ---")
    print(f"Encoding {len(unique_terms)} unique concepts...")
    embeddings_raw = model.encode(
        unique_terms, convert_to_tensor=True,
        show_progress_bar=True, batch_size=batch_size
    )
    term_embeddings = torch.nn.functional.normalize(embeddings_raw, p=2, dim=1)

    similarities = compute_similarities(term_embeddings, term_to_idx, terms1, terms2)
    labels = df['label'].values

    print(f"\n--- Results per Threshold ({model_name}) ---")
    print(f"{'Threshold':<10} | {'F1':<10} | {'Precision':<10} | "
          f"{'Recall':<10} | {'PosAcc':<10} | {'NegAcc':<10} | {'MCC':<10}")
    print("-" * 95)

    results = []
    best_f1 = -1
    best_threshold = -1

    for t in thresholds:
        p, r, f1, pos_acc, neg_acc, mcc = evaluate_pairwise(
            similarities, labels, t
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        results.append({
            'model': model_name,
            'threshold': t,
            'precision_mean': p, 'precision_var': 0.0,
            'recall_mean': r, 'recall_var': 0.0,
            'f1_mean': f1, 'f1_var': 0.0,
            'pos_acc_mean': pos_acc,
            'neg_acc_mean': neg_acc,
            'mcc': mcc
        })

        print(f"{t:.2f}       | {f1:.4f}     | {p:.4f}     | "
              f"{r:.4f}     | {pos_acc:.4f}     | {neg_acc:.4f}     | {mcc:.4f}")

    return results, best_threshold

def plot_individual_performance(model_results, model_name, best_threshold_model, plots_dir, display_name=None):
    if display_name is None:
        display_name = model_name

    print(f"Generating plot for {display_name}...")
    current_df = pd.DataFrame(model_results)
    
    # Set a nicer theme
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=current_df, x='threshold', y='f1_mean', label='F1 Score', linewidth=2.5, color='#e74c3c')
    sns.lineplot(data=current_df, x='threshold', y='precision_mean', label='Precision', linestyle='--', color='#2980b9')
    sns.lineplot(data=current_df, x='threshold', y='recall_mean', label='Recall', linestyle='--', color='#2ecc71')
    
    # Annotate best threshold without crowding the legend
    plt.axvline(x=best_threshold_model, color='#34495e', linestyle=':', alpha=0.9)
    # Find max F1 for annotation position
    max_f1 = current_df.loc[current_df['threshold'] == best_threshold_model, 'f1_mean'].values[0]
    
    plt.annotate(f'Best F1: {max_f1:.3f}\nThresh: {best_threshold_model:.2f}', 
                 xy=(best_threshold_model, max_f1), 
                 xytext=(10, 20), textcoords='offset points',
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.title(f'Performance Analysis: {display_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Adjusted legend to be smaller and outside if needed, or just smaller inside
    plt.legend(loc='upper right', frameon=True, fontsize='small')
    plt.tight_layout()
    
    # Save results
    
    os.makedirs(plots_dir, exist_ok=True)
    
    safe_name = model_name.replace('/', '_').replace('-', '_')
    save_path = os.path.join(plots_dir, f'performance_graph_{safe_name}.svg')
    
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    plt.show() # Inline display

def plot_comparison(summary_df, model_display_names=None, title=None, plots_dir=None):
    print("Generating comparison plot...")
    
    # Enhance visualization style
    sns.set_theme(style="whitegrid", context="talk", palette="turbo")
    
    plt.figure(figsize=(14, 9))
    
    ax = sns.lineplot(data=summary_df, x='threshold', y='f1_mean', hue='model', linewidth=2.5,  style='model', dashes=False)

    # Annotate best points directly on the graph
    colors = sns.color_palette("turbo", n_colors=len(summary_df['model'].unique()))
    
    for i, model_name in enumerate(summary_df['model'].unique()):
        model_data = summary_df[summary_df['model'] == model_name]
        
        # Use friendly name if available
        display_label = model_display_names.get(model_name, model_name) if model_display_names else model_name
        
        if not model_data.empty:
            best_idx = model_data['f1_mean'].idxmax()
            best_row = model_data.loc[best_idx]
            
            # Simple marker for the max point
            plt.plot(best_row['threshold'], best_row['f1_mean'], 'o', markersize=8, color='black', alpha=0.5)
            
            # Smart annotation placement
            y_offset = 10 if i % 2 == 0 else -25  # Stagger labels up/down
            plt.annotate(
                f"{display_label}\n(F1={best_row['f1_mean']:.2f} @ {best_row['threshold']:.2f})",
                xy=(best_row['threshold'], best_row['f1_mean']),
                xytext=(0, y_offset),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
            )

    plt.title(title or 'F1 Score Comparison Across Models', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Cosine Similarity Threshold', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    
    # Improve legend - Move to bottom to avoid squeezing width
    handles, labels = ax.get_legend_handles_labels()
    # Map raw model names to display names for legend
    if model_display_names:
        labels = [model_display_names.get(l, l) for l in labels]
        
    plt.legend(handles, labels, title="Models", loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    
    plt.tight_layout()
    
    plots_dir = plots_dir or 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, 'performance_comparison.svg')
    
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Graph saved to '{save_path}'")
    plt.show() # Inline display

def print_best_settings(summary_df):
    print("\n" + "="*40)
    print(" BEST SETTINGS PER MODEL ")
    print("="*40)
    for model_name in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model_name]
        if not model_data.empty:
            best_idx = model_data['f1_mean'].idxmax()
            best_row = model_data.loc[best_idx]
            print(f"Model: {model_name}")
            print(f"  Optimal Threshold: {best_row['threshold']:.2f}")
            print(f"  Max F1 Score:      {best_row['f1_mean']:.4f}")
            print(f"  Precision:         {best_row['precision_mean']:.4f}")
            print(f"  Recall:            {best_row['recall_mean']:.4f}")
            print("-" * 40)
