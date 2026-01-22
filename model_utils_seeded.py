import os

import numpy as np
import pandas as pd

import config
from model_utils_shared import load_model


def _l2_normalize(matrix):
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = matrix / norms
    if normalized.dtype != matrix.dtype:
        normalized = normalized.astype(matrix.dtype, copy=False)
    return normalized


def _build_concept_groups(df, terms):
    term_set = set(terms)
    concept_to_terms = {}

    if 'concept_uri' in df.columns:
        for term1, term2, concept in zip(df['term1'], df['term2'], df['concept_uri']):
            if pd.isna(concept):
                continue
            if term1 in term_set:
                concept_to_terms.setdefault(concept, set()).add(term1)
            if term2 in term_set:
                concept_to_terms.setdefault(concept, set()).add(term2)

    if 'concept1_uri' in df.columns and 'concept2_uri' in df.columns:
        for term, concept in zip(df['term1'], df['concept1_uri']):
            if pd.isna(concept):
                continue
            if term in term_set:
                concept_to_terms.setdefault(concept, set()).add(term)
        for term, concept in zip(df['term2'], df['concept2_uri']):
            if pd.isna(concept):
                continue
            if term in term_set:
                concept_to_terms.setdefault(concept, set()).add(term)

    if not concept_to_terms:
        concept_to_terms = {term: {term} for term in terms}

    return concept_to_terms


def sample_distinct_seeds(data, n_seeds, terms=None, random_state=None):
    """
    Samples seeds from distinct concepts. If a DataFrame is provided, terms must
    be the full term list used for indexing. If a list/array is provided, it is
    treated as the term list directly.
    """
    if isinstance(data, pd.DataFrame):
        if terms is None:
            raise ValueError("terms must be provided when data is a DataFrame")
        terms_list = list(terms)
        concept_to_terms = _build_concept_groups(data, terms_list)
    else:
        terms_list = list(data)
        concept_to_terms = {term: {term} for term in terms_list}

    term_to_idx = {term: i for i, term in enumerate(terms_list)}
    concepts = list(concept_to_terms.keys())
    if not concepts:
        return []

    rng = np.random.default_rng(random_state)
    n_select = min(n_seeds, len(concepts))
    selected_concepts = rng.choice(concepts, size=n_select, replace=False)

    seed_indices = []
    for concept in selected_concepts:
        candidates = [t for t in concept_to_terms[concept] if t in term_to_idx]
        if not candidates:
            continue
        seed_term = rng.choice(candidates)
        seed_indices.append(term_to_idx[seed_term])

    return seed_indices


def _run_seeded_clustering_normalized(embeddings, seed_indices, threshold, order=None):
    n_terms = embeddings.shape[0]
    if n_terms == 0:
        return np.array([], dtype=int)

    dim = embeddings.shape[1]
    cluster_labels = np.full(n_terms, -1, dtype=int)
    seed_mask = np.zeros(n_terms, dtype=bool)
    # Pre-allocate to avoid repeated reallocations as clusters grow.
    cluster_sums = np.zeros((n_terms, dim), dtype=embeddings.dtype)
    centroid_matrix = np.zeros((n_terms, dim), dtype=embeddings.dtype)
    counts = np.zeros(n_terms, dtype=int)
    n_clusters = 0

    for idx in seed_indices:
        if idx < 0 or idx >= n_terms:
            raise IndexError("seed_indices contains out-of-range index")
        if seed_mask[idx]:
            continue
        seed_mask[idx] = True
        vec = embeddings[idx]
        cluster_labels[idx] = n_clusters
        cluster_sums[n_clusters] = vec
        centroid_matrix[n_clusters] = vec
        counts[n_clusters] = 1
        n_clusters += 1

    # Use a provided order to capture order effects without permuting arrays.
    if order is None:
        order = range(n_terms)

    for idx in order:
        if seed_mask[idx]:
            continue
        vec = embeddings[idx]
        if n_clusters == 0:
            cluster_labels[idx] = n_clusters
            cluster_sums[n_clusters] = vec
            centroid_matrix[n_clusters] = vec
            counts[n_clusters] = 1
            n_clusters += 1
            continue

        similarities = centroid_matrix[:n_clusters].dot(vec)
        best_cluster = int(np.argmax(similarities))
        best_sim = float(similarities[best_cluster])

        if best_sim > threshold:
            cluster_labels[idx] = best_cluster
            cluster_sums[best_cluster] += vec
            counts[best_cluster] += 1
            centroid = centroid_matrix[best_cluster]
            centroid[:] = cluster_sums[best_cluster] / counts[best_cluster]
            norm = np.sqrt(np.dot(centroid, centroid))
            if norm != 0.0:
                centroid /= norm
        else:
            cluster_labels[idx] = n_clusters
            cluster_sums[n_clusters] = vec
            centroid_matrix[n_clusters] = vec
            counts[n_clusters] = 1
            n_clusters += 1

    return cluster_labels


def run_seeded_clustering(embeddings, seed_indices, threshold, order=None):
    """
    Incremental seeded clustering using cosine similarity and dynamic centroids.
    """
    normalized = _l2_normalize(embeddings)
    return _run_seeded_clustering_normalized(normalized, seed_indices, threshold, order=order)


def prepare_pair_indices(df_pairs, term_to_idx):
    idx1 = df_pairs['term1'].map(term_to_idx).to_numpy()
    idx2 = df_pairs['term2'].map(term_to_idx).to_numpy()
    valid_mask = pd.notna(idx1) & pd.notna(idx2)
    if not valid_mask.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    return idx1[valid_mask].astype(int), idx2[valid_mask].astype(int)


def evaluate_clusters_from_indices(cluster_labels, pos_idx1, pos_idx2, neg_idx1, neg_idx2):
    if len(pos_idx1) == 0 and len(neg_idx1) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Manual metric computation avoids sklearn overhead per run.
    pos_preds = (cluster_labels[pos_idx1] == cluster_labels[pos_idx2]).astype(int)
    neg_preds = (cluster_labels[neg_idx1] == cluster_labels[neg_idx2]).astype(int)

    tp = int(pos_preds.sum())
    fn = len(pos_preds) - tp
    fp = int(neg_preds.sum())
    tn = len(neg_preds) - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    pos_acc = tp / len(pos_preds) if len(pos_preds) else 0.0
    neg_acc = tn / len(neg_preds) if len(neg_preds) else 0.0

    return precision, recall, f1, pos_acc, neg_acc


def _evaluate_clusters(cluster_labels, term_to_idx, df_pos, df_neg):
    pos_idx1, pos_idx2 = prepare_pair_indices(df_pos, term_to_idx)
    neg_idx1, neg_idx2 = prepare_pair_indices(df_neg, term_to_idx)
    return evaluate_clusters_from_indices(cluster_labels, pos_idx1, pos_idx2, neg_idx1, neg_idx2)


def calibrate_seeded_clustering():
    df_pos = pd.read_csv(config.DATA_PATHS['pos_pairs'])
    df_neg = pd.read_csv(config.DATA_PATHS['neg_pairs'])
    df_train = pd.concat([df_pos, df_neg], ignore_index=True)

    df_train['term1'] = df_train['term1'].astype(str)
    df_train['term2'] = df_train['term2'].astype(str)
    df_pos['term1'] = df_pos['term1'].astype(str)
    df_pos['term2'] = df_pos['term2'].astype(str)
    df_neg['term1'] = df_neg['term1'].astype(str)
    df_neg['term2'] = df_neg['term2'].astype(str)

    flat_terms = pd.unique(df_train[['term1', 'term2']].values.ravel('K'))
    unique_terms = [str(term) for term in flat_terms if pd.notna(term)]
    term_to_idx = {term: i for i, term in enumerate(unique_terms)}
    pos_idx1, pos_idx2 = prepare_pair_indices(df_pos, term_to_idx)
    neg_idx1, neg_idx2 = prepare_pair_indices(df_neg, term_to_idx)

    results = []
    seed_counts = [10, 25, 50, 100, 250, 500]
    run_states = [config.SEED + i for i in range(5)]

    results_dir = config.DATA_PATHS['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'seeded_calibration_results.csv')

    for model_name, model_info in config.MODELS.items():
        model_type = model_info.get('type', config.TYPE_SENTENCE)
        print(f"Loading model: {model_name}")
        model = load_model(model_name, model_type)
        if model is None:
            continue

        print(f"Encoding {len(unique_terms)} terms...")
        embeddings = model.encode(
            unique_terms,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=config.BATCH_SIZE
        )
        embeddings = _l2_normalize(embeddings)

        n_terms = len(unique_terms)
        for n_seeds in seed_counts:
            print(f"Calibrating seed count: {n_seeds}")
            for run_id, random_state in enumerate(run_states, start=1):
                rng = np.random.default_rng(random_state)
                order = rng.permutation(n_terms)
                seed_indices = sample_distinct_seeds(
                    df_train,
                    n_seeds,
                    terms=unique_terms,
                    random_state=random_state
                )

                for threshold in config.THRESHOLDS:
                    cluster_labels = _run_seeded_clustering_normalized(
                        embeddings,
                        seed_indices,
                        threshold,
                        order=order
                    )

                    precision, recall, f1, pos_acc, neg_acc = evaluate_clusters_from_indices(
                        cluster_labels,
                        pos_idx1,
                        pos_idx2,
                        neg_idx1,
                        neg_idx2
                    )

                    n_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
                    results.append({
                        'Model': model_name,
                        'n_seeds': n_seeds,
                        'Threshold': float(threshold),
                        'Run_ID': run_id,
                        'F1': f1,
                        'Precision': precision,
                        'Recall': recall,
                        'Pos_Acc': pos_acc,
                        'Neg_Acc': neg_acc,
                        'Num_Clusters': n_clusters
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Saved seeded calibration results to {output_path}")
    return results_df
