"""
audit_utility.py
================
Unified psychometric audit suite for concept harmonisation benchmarking.

This module provides reusable functions for five psychometric audits:
1. Reliability (SEM under stochastic noise)
2. Discriminant Validity (lexical trap false-positive rate)
3. Structural Validity (correlation with expert graph)
4. Differential Item Functioning (rare-word bias)
5. Semantic Decay (semantic gap degradation)

Each audit function takes prepared data and model predictions, returning
structured results suitable for display or export.
"""

import warnings
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score as sk_f1
from sklearn.preprocessing import normalize

import model_utils_shared as shared
import model_utils_clustering as muc
import model_utils_pairwise as mup
import model_utils_seed as mus


# ==========================================================================
#  0. PRECOMPUTE AUDIT MASKS
# ==========================================================================

def precompute_audit_masks(full_df, pos_df, neg_df, unique_terms):
    """
    Build boolean masks for lexical, frequency, and difficulty subsets.

    Args:
        full_df:       Combined DataFrame (positive + negative pairs).
        pos_df:        Positive pairs DataFrame.
        neg_df:        Negative pairs DataFrame.
        unique_terms:  List of all unique terms.

    Returns:
        dict with keys:
            - lex_mask:    Hard negatives with Jaccard > 0.5 (Audit 2)
            - easy_mask:   Easy positives with Jaccard > 0.5 (Audit 5)
            - hard_mask:   Hard positives with Jaccard = 0.0 (Audit 5)
            - rare_mask:   Bottom 10% frequency pairs (Audit 4)
            - common_mask: Top 10% frequency pairs (Audit 4)
            - has_wordfreq: bool indicating if wordfreq is available
            - summary_df:  DataFrame summarizing mask counts
    """
    pos_len = len(pos_df)

    # Letter n-gram cache (3-grams) for lexical similarity
    token_cache = {t: shared.letter_ngrams(t, n=3) for t in unique_terms}

    # ── Audit 2: Lexical trap mask (hard negatives with Jaccard > 0.5) ───
    lex_mask_neg = np.array([
        shared.jaccard_sim(token_cache.get(t1, set()), token_cache.get(t2, set())) > 0.5
        for t1, t2 in zip(neg_df["term1"], neg_df["term2"])
    ], dtype=bool)
    lex_mask = np.zeros(len(full_df), dtype=bool)
    lex_mask[pos_len:] = lex_mask_neg

    # ── Audit 5: Easy / Hard positive masks ──────────────────────────────
    jacc_pos = np.array([
        shared.jaccard_sim(token_cache.get(t1, set()), token_cache.get(t2, set()))
        for t1, t2 in zip(pos_df["term1"], pos_df["term2"])
    ])
    easy_mask = np.zeros(len(full_df), dtype=bool)
    hard_mask = np.zeros(len(full_df), dtype=bool)
    easy_mask[:pos_len] = jacc_pos > 0.5
    hard_mask[:pos_len] = jacc_pos == 0.0

    # ── Audit 4: Rare / Common word frequency masks ──────────────────────
    try:
        from wordfreq import zipf_frequency

        term_freq = {}
        for t in unique_terms:
            toks = shared.simple_tokens(t)
            term_freq[t] = float(np.mean([zipf_frequency(tok, "en") for tok in toks])) if toks else 0.0

        pos_pair_freq = np.array([
            (term_freq.get(t1, 0.0) + term_freq.get(t2, 0.0)) / 2.0
            for t1, t2 in zip(pos_df["term1"], pos_df["term2"])
        ])
        low_thr = np.quantile(pos_pair_freq, 0.10)
        high_thr = np.quantile(pos_pair_freq, 0.90)

        rare_mask = np.zeros(len(full_df), dtype=bool)
        common_mask = np.zeros(len(full_df), dtype=bool)
        rare_mask[:pos_len] = pos_pair_freq <= low_thr
        common_mask[:pos_len] = pos_pair_freq >= high_thr
        has_wordfreq = True
    except ImportError:
        rare_mask = common_mask = np.zeros(len(full_df), dtype=bool)
        has_wordfreq = False
        warnings.warn("wordfreq not installed — Audit 4 will be skipped.", UserWarning)

    # Summary DataFrame
    summary_df = pd.DataFrame({
        "Audit Mask": [
            "Lexical trap (neg, Jaccard > 0.5)",
            "Easy positives (Jaccard > 0.5)",
            "Hard positives (Jaccard = 0.0)",
            "Common pairs (top 10% freq)",
            "Rare pairs (bottom 10% freq)",
        ],
        "N": [
            int(lex_mask.sum()),
            int(easy_mask.sum()),
            int(hard_mask.sum()),
            int(common_mask.sum()),
            int(rare_mask.sum()),
        ],
    })

    return {
        "lex_mask": lex_mask,
        "easy_mask": easy_mask,
        "hard_mask": hard_mask,
        "rare_mask": rare_mask,
        "common_mask": common_mask,
        "has_wordfreq": has_wordfreq,
        "summary_df": summary_df,
    }


# ==========================================================================
#  1. RELIABILITY AUDIT (SEM under stochastic noise)
# ==========================================================================

def _run_reliability_for_technique(
    technique, model_name, emb_cache,
    noise_levels, n_runs,
    idx1, idx2, idx1_t, idx2_t, labels,
    unique_terms, full_df, terms1, terms2,
    best_clustering_params, best_pairwise_thresholds, best_seeded_params,
    seed=42
):
    """
    Run reliability audit for one model × one technique.

    Returns:
        dict mapping noise_level → {mean_f1, sd_f1, icc, sem}
    """
    emb_raw = emb_cache["raw_np"]
    sigma = emb_cache["sigma"]

    params_c = best_clustering_params.get(model_name, {})
    thr_pw = best_pairwise_thresholds.get(model_name, 0.5)
    params_s = best_seeded_params.get(model_name, {})

    results = {}
    for nl in noise_levels:
        f1_runs, all_preds = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(seed + run + int(nl * 10000))
            noisy = emb_raw + (rng.normal(0, 1, emb_raw.shape) * (nl * sigma)) if nl > 0 else emb_raw
            noisy_norm_np = normalize(noisy, norm="l2", axis=1)
            noisy_norm_np = np.ascontiguousarray(noisy_norm_np, dtype=np.float64)
            noisy_norm_t = F.normalize(torch.tensor(noisy, dtype=torch.float32), p=2, dim=1)

            if technique == "clustering":
                n_comp = params_c.get("n_components")
                red = noisy_norm_np if (n_comp is None or n_comp >= noisy_norm_np.shape[1]) else \
                      muc.reduce_embeddings_with_umap(noisy_norm_np, n_comp, random_state=seed + run)[0]
                cl = muc.run_hdbscan_clustering(red, params_c["min_cluster_size"], params_c["min_samples"])
                p_ = ((cl[idx1] != -1) & (cl[idx1] == cl[idx2])).astype(int)

            elif technique == "pairwise":
                sims = (noisy_norm_t[idx1_t] * noisy_norm_t[idx2_t]).sum(dim=1).cpu().numpy()
                p_ = (sims > thr_pw).astype(np.int8)

            else:  # seeded
                nte = noisy_norm_t.cpu().detach()
                t2e = {t: e for t, e in zip(unique_terms, nte)}
                seeds, rem = mus.sample_unique_terms(
                    params_s.get("n_initial_seeds", 250), full_df,
                    np.array(unique_terms), random_state=seed + run)
                with redirect_stdout(StringIO()):
                    sc = mus.seed_clustering(seeds, rem, t2e, threshold=params_s.get("threshold", 0.7))
                t2c = {}
                for cid, (sd, ms) in enumerate(sc.items()):
                    for m in ms:
                        t2c[m] = cid
                c1 = np.array([t2c.get(t, -1) for t in terms1])
                c2 = np.array([t2c.get(t, -1) for t in terms2])
                p_ = ((c1 == c2) & (c1 != -1)).astype(int)

            f1_runs.append(sk_f1(labels, p_, zero_division=0))
            all_preds.append(p_)

        f1_arr = np.array(f1_runs)
        preds_mat = np.array(all_preds)
        icc = shared.compute_icc_oneway(
            preds_mat.sum(axis=0), (preds_mat ** 2).sum(axis=0),
            preds_mat.shape[1], n_runs)
        sd = np.std(f1_arr, ddof=1)
        sem = sd * np.sqrt(max(0, 1 - icc))
        results[nl] = {"mean_f1": f1_arr.mean(), "sd_f1": sd, "icc": icc, "sem": sem}
    return results


def run_reliability_audit(
    models_to_eval, embedding_cache, model_configs,
    idx1, idx2, idx1_t, idx2_t, labels,
    unique_terms, full_df, terms1, terms2,
    best_clustering_params, best_pairwise_thresholds, best_seeded_params,
    noise_levels=None, n_runs=5, seed=42
):
    """
    Run Audit 1 (Reliability) across all models and techniques.

    Args:
        models_to_eval:             List of model names to evaluate.
        embedding_cache:            Dict mapping model_name → embedding dict.
        model_configs:              Dict mapping model_name → config dict (with "display_name").
        idx1, idx2:                 Pair indices (numpy arrays).
        idx1_t, idx2_t:             Pair indices (torch tensors).
        labels:                     Ground truth labels.
        unique_terms:               List of unique terms.
        full_df:                    Full DataFrame (pos + neg).
        terms1, terms2:             Lists of term1 and term2 strings.
        best_clustering_params:     Dict mapping model_name → clustering params.
        best_pairwise_thresholds:   Dict mapping model_name → pairwise threshold.
        best_seeded_params:         Dict mapping model_name → seeded params.
        noise_levels:               List of noise levels (default [0.0, 0.05, 0.1, 0.2, 0.3]).
        n_runs:                     Number of runs per noise level (default 5).
        seed:                       Random seed.

    Returns:
        pd.DataFrame with columns [Technique, Model, Noise, Mean F1, SD(F1), ICC, SEM].
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

    audit1_records = []
    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in models_to_eval:
            if model_name not in embedding_cache:
                continue
            print(f"  Reliability | {technique:10s} | {model_name} ...", end="", flush=True)
            res = _run_reliability_for_technique(
                technique, model_name, embedding_cache[model_name],
                noise_levels, n_runs,
                idx1, idx2, idx1_t, idx2_t, labels,
                unique_terms, full_df, terms1, terms2,
                best_clustering_params, best_pairwise_thresholds, best_seeded_params,
                seed=seed
            )
            print(" ✓")
            for nl, vals in res.items():
                audit1_records.append({
                    "Technique": technique,
                    "Model": model_configs[model_name]["display_name"],
                    "Noise": nl,
                    "Mean F1": f"{vals['mean_f1']:.4f}",
                    "SD(F1)": f"{vals['sd_f1']:.4f}",
                    "ICC": f"{vals['icc']:.4f}",
                    "SEM": f"{vals['sem']:.4f}",
                })

    return pd.DataFrame(audit1_records)


# ==========================================================================
#  2. DISCRIMINANT VALIDITY AUDIT (Lexical Trap FPR)
# ==========================================================================

def run_discriminant_validity_audit(
    models_to_eval, preds_store, model_configs, labels, lex_mask
):
    """
    Run Audit 2 (Discriminant Validity) — False Positive Rate on lexical traps.

    Args:
        models_to_eval:  List of model names.
        preds_store:     Dict mapping technique → model_name → {"preds": ...}.
        model_configs:   Dict mapping model_name → config dict.
        labels:          Ground truth labels.
        lex_mask:        Boolean mask for lexical trap pairs.

    Returns:
        pd.DataFrame with columns [Technique, Model, Subset N, FP, TN, FPR].
    """
    audit2_records = []

    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in models_to_eval:
            info = preds_store[technique].get(model_name)
            if info is None:
                continue
            preds = info["preds"]
            sub_preds = preds[lex_mask]
            sub_true = labels[lex_mask]
            tp, fp, tn, fn = shared.confusion_counts(sub_true, sub_preds)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            audit2_records.append({
                "Technique": technique,
                "Model": model_configs[model_name]["display_name"],
                "Subset N": int(lex_mask.sum()),
                "FP": fp,
                "TN": tn,
                "FPR": f"{fpr:.4f}",
            })

    return pd.DataFrame(audit2_records)


# ==========================================================================
#  3. STRUCTURAL VALIDITY AUDIT (Expert Graph Correlation)
# ==========================================================================

def run_structural_validity_audit(
    models_to_eval, preds_store, model_configs,
    shortest_path, idx1, idx2, unique_terms, terms1, terms2
):
    """
    Run Audit 3 (Structural Validity) — Correlation with expert ELSST graph.

    Args:
        models_to_eval:  List of model names.
        preds_store:     Dict mapping technique → model_name → predictions.
        model_configs:   Dict mapping model_name → config dict.
        shortest_path:   Array of expert shortest path distances.
        idx1, idx2:      Pair indices (numpy arrays).
        unique_terms:    List of unique terms.
        terms1, terms2:  Lists of term strings.

    Returns:
        pd.DataFrame with correlation metrics.
    """
    expert_mask = shortest_path >= 0
    d_expert = shortest_path[expert_mask].astype(float)

    audit3_records = []

    # ── Clustering & Seeded: binary model distance ───────────────────────
    for technique in ["clustering", "seeded"]:
        for model_name in models_to_eval:
            info = preds_store[technique].get(model_name)
            if info is None:
                continue

            if technique == "clustering":
                cl = info["cluster_labels"]
                l1, l2 = cl[info["idx1"]], cl[info["idx2"]]
                d_model = np.where((l1 != -1) & (l1 == l2), 0, 1).astype(float)
            else:
                t2c = info["term_to_cluster"]
                c1 = np.array([t2c.get(t, -1) for t in terms1])
                c2 = np.array([t2c.get(t, -1) for t in terms2])
                d_model = np.where((c1 != -1) & (c1 == c2), 0, 1).astype(float)

            d_m = d_model[expert_mask]
            rho_s = shared.spearman_corr(d_expert, d_m)
            rho_p, p_val_p = shared.pearson_corr(d_expert, d_m)
            rpb, p_val_pb = shared.point_biserial_corr(d_m, d_expert)

            audit3_records.append({
                "Technique": technique,
                "Model": model_configs[model_name]["display_name"],
                "Metric": "binary_dist",
                "Spearman ρ": f"{rho_s:.4f}",
                "Pearson r": f"{rho_p:.4f}",
                "Point-Biserial r": f"{rpb:.4f}" if not np.isnan(rpb) else "N/A",
                "p (Pearson)": f"{p_val_p:.2e}" if not np.isnan(p_val_p) else "N/A",
                "p (PB)": f"{p_val_pb:.2e}" if not np.isnan(p_val_pb) else "N/A",
            })

    # ── Pairwise: multiple distance metrics ──────────────────────────────
    for model_name in models_to_eval:
        info = preds_store["pairwise"].get(model_name)
        if info is None:
            continue
        preds = info["preds"]
        sims = info["similarities"]

        # (a) Binary same-component via union-find
        edges_mask = preds == 1
        parent, find, union_fn = shared.union_find(len(unique_terms))
        for a, b in zip(idx1[edges_mask], idx2[edges_mask]):
            union_fn(a, b)
        roots1 = np.array([find(i) for i in idx1])
        roots2 = np.array([find(i) for i in idx2])
        same_comp = (roots1 == roots2).astype(float)

        rho_s = shared.spearman_corr(d_expert, same_comp[expert_mask])
        rho_p, pp = shared.pearson_corr(d_expert, same_comp[expert_mask])
        rpb, ppb = shared.point_biserial_corr(same_comp[expert_mask], d_expert)

        audit3_records.append({
            "Technique": "pairwise",
            "Model": model_configs[model_name]["display_name"],
            "Metric": "binary_component",
            "Spearman ρ": f"{rho_s:.4f}",
            "Pearson r": f"{rho_p:.4f}",
            "Point-Biserial r": f"{rpb:.4f}" if not np.isnan(rpb) else "N/A",
            "p (Pearson)": f"{pp:.2e}" if not np.isnan(pp) else "N/A",
            "p (PB)": f"{ppb:.2e}" if not np.isnan(ppb) else "N/A",
        })

        # (b) Model-graph shortest path
        model_sp = shared.model_graph_shortest_paths(
            len(unique_terms), idx1[edges_mask], idx2[edges_mask], idx1, idx2)
        valid_sp = expert_mask & (model_sp >= 0)
        rho_sp = shared.spearman_corr(
            shortest_path[valid_sp].astype(float), model_sp[valid_sp].astype(float)
        )
        rp_sp, pp_sp = shared.pearson_corr(
            shortest_path[valid_sp].astype(float), model_sp[valid_sp].astype(float)
        )

        audit3_records.append({
            "Technique": "pairwise",
            "Model": model_configs[model_name]["display_name"],
            "Metric": "graph_shortest_path",
            "Spearman ρ": f"{rho_sp:.4f}",
            "Pearson r": f"{rp_sp:.4f}",
            "Point-Biserial r": "N/A (continuous)",
            "p (Pearson)": f"{pp_sp:.2e}" if not np.isnan(pp_sp) else "N/A",
            "p (PB)": "—",
        })

        # (c) Cosine distance
        cos_dist = 1.0 - sims
        rho_cd = shared.spearman_corr(d_expert, cos_dist[expert_mask])
        rp_cd, pp_cd = shared.pearson_corr(d_expert, cos_dist[expert_mask])

        audit3_records.append({
            "Technique": "pairwise",
            "Model": model_configs[model_name]["display_name"],
            "Metric": "cosine_distance",
            "Spearman ρ": f"{rho_cd:.4f}",
            "Pearson r": f"{rp_cd:.4f}",
            "Point-Biserial r": "N/A (continuous)",
            "p (Pearson)": f"{pp_cd:.2e}" if not np.isnan(pp_cd) else "N/A",
            "p (PB)": "—",
        })

    return pd.DataFrame(audit3_records)


# ==========================================================================
#  4. DIFFERENTIAL ITEM FUNCTIONING AUDIT (Rare-Word Bias)
# ==========================================================================

def run_dif_audit(
    models_to_eval, preds_store, model_configs, labels,
    rare_mask, common_mask, has_wordfreq
):
    """
    Run Audit 4 (DIF) — Recall gap between common and rare terminology.

    Args:
        models_to_eval:  List of model names.
        preds_store:     Dict mapping technique → model_name → predictions.
        model_configs:   Dict mapping model_name → config dict.
        labels:          Ground truth labels.
        rare_mask:       Boolean mask for rare pairs.
        common_mask:     Boolean mask for common pairs.
        has_wordfreq:    Boolean indicating if wordfreq is available.

    Returns:
        pd.DataFrame with columns [Technique, Model, Recall (Common), Recall (Rare), ΔRecall].
        Returns None if wordfreq is not available.
    """
    if not has_wordfreq:
        return None

    audit4_records = []
    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in models_to_eval:
            preds = preds_store[technique].get(model_name, {}).get("preds")
            if preds is None:
                continue
            rec_common = shared.recall_on_subset(labels, preds, common_mask)
            rec_rare = shared.recall_on_subset(labels, preds, rare_mask)
            gap = rec_common - rec_rare
            audit4_records.append({
                "Technique": technique,
                "Model": model_configs[model_name]["display_name"],
                "Recall (Common)": f"{rec_common:.4f}",
                "Recall (Rare)": f"{rec_rare:.4f}",
                "ΔRecall": f"{gap:+.4f}",
                "N Common": int(common_mask.sum()),
                "N Rare": int(rare_mask.sum()),
            })

    return pd.DataFrame(audit4_records)


# ==========================================================================
#  5. SEMANTIC DECAY AUDIT (Semantic Gap Test)
# ==========================================================================

def run_semantic_decay_audit(
    models_to_eval, preds_store, model_configs, labels,
    easy_mask, hard_mask
):
    """
    Run Audit 5 (Semantic Decay) — Recall degradation on purely-semantic pairs.

    Args:
        models_to_eval:  List of model names.
        preds_store:     Dict mapping technique → model_name → predictions.
        model_configs:   Dict mapping model_name → config dict.
        labels:          Ground truth labels.
        easy_mask:       Boolean mask for easy positive pairs (Jaccard > 0.5).
        hard_mask:       Boolean mask for hard positive pairs (Jaccard = 0.0).

    Returns:
        pd.DataFrame with columns [Technique, Model, Recall (Easy), Recall (Hard), Slope].
    """
    audit5_records = []

    for technique in ["clustering", "pairwise", "seeded"]:
        for model_name in models_to_eval:
            preds = preds_store[technique].get(model_name, {}).get("preds")
            if preds is None:
                continue
            rec_easy = shared.recall_on_subset(labels, preds, easy_mask)
            rec_hard = shared.recall_on_subset(labels, preds, hard_mask)
            slope = rec_hard - rec_easy
            audit5_records.append({
                "Technique": technique,
                "Model": model_configs[model_name]["display_name"],
                "Recall (Easy)": f"{rec_easy:.4f}",
                "Recall (Hard)": f"{rec_hard:.4f}",
                "Slope": f"{slope:+.4f}",
                "N Easy": int(easy_mask.sum()),
                "N Hard": int(hard_mask.sum()),
            })

    return pd.DataFrame(audit5_records)


# ==========================================================================
#  CONSOLIDATED AUDIT RUNNER
# ==========================================================================

def run_all_audits(
    models_to_eval, embedding_cache, preds_store, model_configs,
    full_df, pos_df, neg_df,
    idx1, idx2, idx1_t, idx2_t, labels,
    unique_terms, terms1, terms2, shortest_path,
    best_clustering_params, best_pairwise_thresholds, best_seeded_params,
    noise_levels=None, n_reliability_runs=5, seed=42
):
    """
    Run all five psychometric audits in sequence.

    Args:
        models_to_eval:                List of model names.
        embedding_cache:               Dict mapping model_name → embeddings.
        preds_store:                   Dict mapping technique → model_name → predictions.
        model_configs:                 Dict mapping model_name → config.
        full_df, pos_df, neg_df:       DataFrames.
        idx1, idx2, idx1_t, idx2_t:    Pair indices.
        labels:                        Ground truth labels.
        unique_terms, terms1, terms2:  Term lists.
        shortest_path:                 Expert graph distances.
        best_clustering_params, etc.:  Hyperparameters.
        noise_levels:                  List of noise levels for Audit 1.
        n_reliability_runs:            Number of runs per noise level.
        seed:                          Random seed.

    Returns:
        dict with keys:
            - masks:   Precomputed audit masks.
            - audit1:  Reliability results.
            - audit2:  Discriminant validity results.
            - audit3:  Structural validity results.
            - audit4:  DIF results (or None if wordfreq unavailable).
            - audit5:  Semantic decay results.
    """
    print("=" * 60)
    print("  RUNNING PSYCHOMETRIC AUDIT SUITE")
    print("=" * 60)

    # Precompute masks
    print("\n[0/5] Precomputing audit masks...")
    masks = precompute_audit_masks(full_df, pos_df, neg_df, unique_terms)

    # Audit 1: Reliability
    print("\n[1/5] Audit 1 — Reliability (SEM under stochastic noise)")
    audit1_df = run_reliability_audit(
        models_to_eval, embedding_cache, model_configs,
        idx1, idx2, idx1_t, idx2_t, labels,
        unique_terms, full_df, terms1, terms2,
        best_clustering_params, best_pairwise_thresholds, best_seeded_params,
        noise_levels=noise_levels, n_runs=n_reliability_runs, seed=seed
    )

    # Audit 2: Discriminant Validity
    print("\n[2/5] Audit 2 — Discriminant Validity (Lexical Trap FPR)")
    audit2_df = run_discriminant_validity_audit(
        models_to_eval, preds_store, model_configs, labels, masks["lex_mask"]
    )

    # Audit 3: Structural Validity
    print("\n[3/5] Audit 3 — Structural Validity (Expert Graph Correlation)")
    audit3_df = run_structural_validity_audit(
        models_to_eval, preds_store, model_configs,
        shortest_path, idx1, idx2, unique_terms, terms1, terms2
    )

    # Audit 4: DIF
    print("\n[4/5] Audit 4 — Differential Item Functioning (Rare-Word Bias)")
    if masks["has_wordfreq"]:
        audit4_df = run_dif_audit(
            models_to_eval, preds_store, model_configs, labels,
            masks["rare_mask"], masks["common_mask"], masks["has_wordfreq"]
        )
    else:
        audit4_df = None
        print("  ⚠ Skipped (wordfreq not installed)")

    # Audit 5: Semantic Decay
    print("\n[5/5] Audit 5 — Semantic Decay (Semantic Gap Test)")
    audit5_df = run_semantic_decay_audit(
        models_to_eval, preds_store, model_configs, labels,
        masks["easy_mask"], masks["hard_mask"]
    )

    print("\n" + "=" * 60)
    print("  AUDIT SUITE COMPLETE")
    print("=" * 60)

    return {
        "masks": masks,
        "audit1": audit1_df,
        "audit2": audit2_df,
        "audit3": audit3_df,
        "audit4": audit4_df,
        "audit5": audit5_df,
    }
