# Methodology: Evaluating NLP Models for Social Science Concept Harmonisation

## Overview

This document describes the technical methodology used to evaluate machine-learning models for the task of **concept harmonisation** in social science thesauri. The evaluation is conducted across two independent datasets — **ELSST** (European Language Social Science Thesaurus) and **APA** (American Psychological Association Thesaurus) — using three harmonisation techniques, a character n-gram baseline without a language model, and five diagnostic tests.

This version describes the revised pipeline. The code and settings behind the initially submitted results are available under the git tag `v1.0-submission`.

---

## 1. Experimental Design and Data

### 1.1 Datasets

| Dataset | Role | Terms | Pair Source |
|---------|------|-------|-------------|
| **ELSST** | Calibration (train split, 4,270 terms) + held-out test (test split, 1,876 terms) | 6,146 unique | `datasets/processed_datasets/elsst/` |
| **APA**   | Unseen thesaurus (never used in calibration) | 11,014 unique | `datasets/processed_datasets/apa/` |

Pairs are built by `scripts/process_dataset.py`. Each dataset is represented as a collection of **labelled term pairs**:

- **Positive pairs** (`label = 1`): every combination of two labels (preferred label and entry terms) of the same concept.
- **Negative pairs** (`label = 0`): every combination of a label of one concept with a label of another concept, over **all labels** of both concepts (preferred labels and entry terms). Pairs of two concepts on the same family line (a concept with any of its ancestors or descendants, up to the root) are excluded, because a broader and a narrower term share part of their meaning. Rows whose two strings are identical after lowercasing, or that form a positive pair elsewhere in the split (labels shared by two concepts), are skipped. A label that belongs to two concepts (5 labels in the ELSST training split, none in the test split or APA) gives each term pair only one row, and the row is dropped if any owner of one label is on the family line of an owner of the other. Each negative row records its `pair_type` (preferred × preferred, preferred × entry term, entry term × entry term).
- **Broader/narrower pairs** (separate file, not part of the benchmark): every combination of labels of a concept and one of its ancestors, with the depth (1 = direct parent). Merge rates on these pairs are reported separately.

ELSST is split 70/30 by concept (seed 42); concepts that share a label are kept in the same split, so no term appears in both splits. Calibration uses only the train split, the final evaluation only the test split. APA is built once over the full thesaurus, without a split, so that negative pairs between all concepts are present.

| Split | Concepts | Terms | Positives | Negatives | Broader/narrower |
|-------|---------:|------:|----------:|----------:|-----------------:|
| ELSST train | 2,404 | 4,270 | 3,587 | 9,098,662 | 12,066 |
| ELSST test | 1,031 | 1,876 | 1,557 | 1,754,671 | 2,522 |
| APA (full) | 7,333 | 11,014 | 5,639 | 60,562,625 | 80,327 |

Additional metadata available for each pair:
- `shortest_path`: the graph distance (number of broader/narrower edges) between the two concepts in the expert thesaurus (−1 = not connected). Used in the Structural alignment test.

### 1.2 Models Evaluated

Four transformer-based sentence encoder models are benchmarked:

| Display Name | HuggingFace ID | Architecture |
|-------------|----------------|-------------|
| All-MPNet-Base-v2 | `all-mpnet-base-v2` | MPNet (sentence-transformers fine-tuned) |
| MPNet-Personality | `dwulff/mpnet-personality` | MPNet (psychology domain fine-tuned) |
| SciBERT (SciVocab) | `allenai/scibert_scivocab_uncased` | BERT (scientific text pre-trained) |
| BERT Base | `bert-base-uncased` | BERT (general-purpose pre-trained) |

Each model produces a dense vector representation (embedding) of dimension 768 for each input term.

### 1.3 Hyperparameter Calibration

All technique-specific hyperparameters were selected by **grid search on the ELSST train split** (no cross-validation folds), with F1 as the criterion, and are **frozen** for the final evaluation runs. They are loaded from `scripts/config.py`:

- `BEST_CLUSTERING_PARAMS`: HDBSCAN `min_cluster_size`, `min_samples`, UMAP `n_components` per model.
- `BEST_PAIRWISE_THRESHOLDS`: cosine similarity decision threshold per model.
- `BEST_NGRAM_THRESHOLD`: Jaccard decision threshold of the character 3-gram baseline.
- `BEST_SEEDED_PARAMS`: number of initial seeds and similarity threshold per model (best mean F1 over 10 seed draws per setting).

This strict separation ensures that reported metrics reflect **true generalisation**, not in-sample fit.

---

## 2. Embedding Generation

For each model, embeddings are computed once for all unique terms and cached in memory:

```
E ∈ ℝ^(N × 768)   where N = number of unique terms
```

Each embedding vector is **L2-normalised** after encoding:

$$\hat{e}_i = \frac{e_i}{\|e_i\|_2}$$

This normalisation ensures that cosine similarity reduces to a simple dot product:

$$\cos(e_i, e_j) = \hat{e}_i \cdot \hat{e}_j$$

The cache stores both the raw (unnormalised) embeddings and the normalised versions, as both are required by different parts of the pipeline (raw embeddings are used for noise injection in the Embedding stability test).

---

## 3. Harmonisation Techniques

Three independent harmonisation techniques are evaluated. Each produces a binary prediction $\hat{y} \in \{0, 1\}$ for every term pair, where $\hat{y} = 1$ means "harmonically equivalent."

### 3.1 Technique 1 — UMAP + HDBSCAN Clustering

**Concept:** Terms that belong to the same semantic cluster are predicted to be harmonically equivalent.

**Pipeline:**

**Step 1 — Dimensionality Reduction (UMAP)**

If the best `n_components` parameter is smaller than 768, the embedding matrix is projected from 768 to a lower-dimensional space using Uniform Manifold Approximation and Projection (UMAP):

$$E' = \text{UMAP}(E,\ n\_components)$$

UMAP preserves local topological structure while reducing dimensionality, which helps HDBSCAN discover more coherent clusters.

**Step 2 — Density-Based Clustering (HDBSCAN)**

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is applied to $E'$. It requires two hyperparameters:

- `min_cluster_size`: the minimum number of terms that form a valid cluster.
- `min_samples`: controls cluster conservatism — higher values produce fewer, tighter clusters.

HDBSCAN assigns each term an integer cluster label $c_i \in \{-1, 0, 1, 2, \ldots\}$, where $-1$ denotes a noise point (not assigned to any cluster).

**Step 3 — Prediction**

A pair $(t_i, t_j)$ is predicted as harmonically equivalent if and only if both terms share the same non-noise cluster:

$$\hat{y}_{ij} = \mathbb{1}[c_i = c_j \text{ and } c_i \neq -1]$$

**Key evaluation metrics from the clustering step:**
- `n_clusters`: number of discovered clusters (excluding noise).
- `n_noise`: number of terms labelled as noise ($c_i = -1$).
- `Pos Acc` = Recall: fraction of true positive pairs correctly identified.
- `Neg Acc` = specificity: fraction of true negative pairs correctly rejected.

### 3.2 Technique 2 — Pairwise Cosine Similarity Thresholding

**Concept:** Two terms are predicted to be harmonically equivalent if their embeddings are sufficiently similar, measured by cosine similarity.

**Cosine Similarity:**

$$\text{sim}(t_i, t_j) = \hat{e}_i \cdot \hat{e}_j \in [-1, 1]$$

Because embeddings are L2-normalised, this equals the standard cosine similarity.

**Decision Rule:**

$$\hat{y}_{ij} = \mathbb{1}[\text{sim}(t_i, t_j) > \tau]$$

where $\tau \in (0, 1)$ is the **best threshold**, determined during grid-search calibration on ELSST. The calibrated thresholds are notably high (0.66–0.90), reflecting the conservative nature of synonym detection.

**Matthews Correlation Coefficient (MCC):**

In addition to Precision, Recall, and F1, the pairwise technique also reports MCC, which is robust to class imbalance:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

MCC = 1 is perfect prediction, MCC = 0 is random, MCC = −1 is complete inversion.

### 3.3 Technique 3 — Seeded Clustering

**Concept:** Begin with a small number of "seed" terms whose thematic identity is known; expand clusters by adding any unassigned term whose embedding is sufficiently close to the seed embedding.

**Pipeline:**

**Step 1 — Seed Sampling**

$K$ seed terms are drawn uniformly at random from all unique terms (`sample_initial_seeds()`), without any gold information. The number of seeds $K$ is a calibrated hyperparameter (`n_initial_seeds`). Because the result depends on the drawn seeds, the final evaluation repeats the technique with 10 seed draws (random seed 42 + d) and reports the mean and SD; the draw with the median F1 is used for the diagnostic tests.

**Step 2 — Cluster Expansion**

For each remaining (non-seed) term $t$, compute cosine similarity to each seed $s_k$:

$$\text{sim}(t, s_k) = \hat{e}_t \cdot \hat{e}_{s_k}$$

Assign $t$ to the cluster of seed $s_k^*$ if:

$$s_k^* = \arg\max_{k} \text{sim}(t, s_k) \quad \text{and} \quad \text{sim}(t, s_k^*) > \theta$$

The cluster centre is then updated to the running mean of its members. If no centre exceeds threshold $\theta$, the term starts a new cluster. The remaining terms are processed in a random order drawn with the same random state as the seeds, so that synonyms of one concept do not follow each other.

**Step 3 — Prediction**

Identical to clustering:

$$\hat{y}_{ij} = \mathbb{1}[c_i = c_j \text{ and } c_i \neq -1]$$

### 3.4 Character 3-gram baseline (no language model)

Two terms are predicted as equivalent when the Jaccard similarity of their character 3-gram sets (see Section 5.2) exceeds a threshold $\tau_J$. The threshold is calibrated on the ELSST train split in the same way as the cosine thresholds. The baseline shows what pure surface matching achieves, so that the contribution of the embeddings can be read against it.

---

## 4. Primary Evaluation Metrics

All three techniques are evaluated using the same classification metrics, treating synonym prediction as a **binary classification problem** (label 1 = synonym, label 0 = not synonym):

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Precision** | $\frac{TP}{TP + FP}$ | Fraction of predicted synonyms that are correct |
| **Recall** | $\frac{TP}{TP + FN}$ | Fraction of true synonyms that are found |
| **F1** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean — primary ranking metric |
| **Pos Acc** | $= \text{Recall}$ | Sensitivity to positive pairs |
| **Neg Acc** | $\frac{TN}{TN + FP}$ | Specificity — ability to reject non-synonyms |

The **F1 score** is the primary ranking metric because the dataset is highly class-imbalanced (negative pairs vastly outnumber positive pairs), making accuracy uninformative.

**False-positive rate and F1 at a common prevalence.** Precision and F1 depend on the ratio of negatives to positives, which differs between ELSST and APA. We therefore also report the false-positive rate $\text{FPR} = FP / (FP + TN)$ and the F1 a pipeline would reach at the prevalence of the ELSST test split ($n_+ = 1{,}557$, $n_- = 1{,}754{,}671$), keeping its recall $R$ and FPR fixed:

$$P_{\text{ref}} = \frac{R \cdot n_+}{R \cdot n_+ + \text{FPR} \cdot n_-}, \qquad F1_{\text{ref}} = \frac{2 P_{\text{ref}} R}{P_{\text{ref}} + R}$$

**Uncertainty (concept-level bootstrap).** Pairs are not independent, because one concept appears in many pairs. We resample the concepts of the evaluation set with replacement (1,000 samples). A concept drawn $k$ times counts $k$ times: a pair inside one concept gets weight $k$, a pair of two concepts weight $k_1 k_2$. Precision, recall, F1 and the subset rates of the diagnostic tests are recomputed from the weighted counts, and the 2.5th and 97.5th percentiles give 95 % intervals. The same samples are used for every pipeline, which gives paired intervals for F1 differences. The pipelines are kept fixed (seeded clustering: the median seed draw); seed and noise variation are reported separately.

**Ranking measures.** Top-k recall: for every synonym pair (both directions), whether the synonym is among the k nearest neighbours (k = 1, 5, 10) of the other term among all terms; a synonym tied with other terms counts as the share of tie orders that place it inside the top k. Correct matches per reviewed candidates: all benchmark pairs ranked by similarity (broader/narrower pairs are not part of this list), number of synonym pairs among the first 100, 500 and 1,000.

---

## 5. Diagnostic Tests

Beyond aggregate performance, five diagnostic tests probe failure modes that summary scores can obscure. Each test uses precomputed pair subsets held constant across all models.

### 5.1 Embedding stability test

**Objective:** Quantify how much the output of each pipeline changes under small random perturbations of the embedding space, and, for seeded clustering, under different seed draws.

**Method:**

For each noise level $\sigma_\epsilon \in \{0.05, 0.10, 0.20, 0.30\}$, Gaussian noise is injected into the raw (unnormalised) embeddings:

$$\tilde{e}_i = e_i + \sigma_\epsilon \cdot \hat{\sigma} \cdot z_i, \quad z_i \sim \mathcal{N}(0, I)$$

where $\hat{\sigma}$ is the per-dimension empirical standard deviation of the embedding matrix. The noisy embeddings are re-normalised and the technique is re-run 10 times per noise level (different noise each run; UMAP random state and the seed draw are kept fixed). At $\sigma_\epsilon = 0$, clustering and pairwise thresholding are deterministic (one run); for seeded clustering the runs at $\sigma_\epsilon = 0$ use 10 different seed draws without noise. Seed variation and embedding noise are therefore reported separately.

**Reported per noise level** (reference = noise-free run; for seeded clustering, draw 0 without noise):

- mean F1 and SD of F1 across runs;
- mean change of F1 from the reference run;
- mean agreement of the predicted matches with the reference run, measured as the Jaccard overlap of the two sets of predicted synonym pairs (1 = identical predictions).

### 5.2 Lexical trap test

**Objective:** A valid harmonization system should separate surface form from conceptual content (Campbell & Fiske, 1959). This test detects whether models confuse orthographically similar but semantically unrelated terms (i.e., they are "fooled" by surface-level form rather than meaning).

**Lexical Similarity via Character 3-Gram Jaccard:**

For a term $t$, let $G(t)$ be the set of all character-level 3-grams. The Jaccard similarity between two terms is:

$$J(t_i, t_j) = \frac{|G(t_i) \cap G(t_j)|}{|G(t_i) \cup G(t_j)|}$$

This is computed for all negative pairs (label = 0).

**Subset Definitions:**

- **Lexical traps (hard negatives):** $J(t_i, t_j) > 0.5$ — high surface similarity; these are the difficult cases where a naive model might incorrectly predict synonymy.
- **Easy negatives:** $J(t_i, t_j) = 0.0$ — zero character overlap; these should be trivially rejected by any model.

**Evaluation Metric — False Positive Rate (FPR):**

Because both subsets contain only label-0 pairs (no true positives exist), Precision / Recall / F1 are undefined and trivially zero. The only meaningful metric is:

$$\text{FPR} = \frac{FP}{FP + TN}$$

A low FPR on hard negatives means the model correctly rejects lexically-similar but semantically-distinct pairs — good discriminant validity. A high FPR on hard negatives (relative to easy negatives) reveals that the model is using surface cues rather than semantic understanding.

### 5.3 Rare-word bias test

**Objective:** Embedding models produce degenerate representations for infrequent tokens, collapsing semantic distinctions in narrow regions of the vector space (Yu et al., 2022). This test detects whether models systematically underperform on pairs containing rare or technical vocabulary.

**Term Frequency via Zipf Scale:**

Each term's frequency is measured using the Zipf scale from the `wordfreq` library. The Zipf frequency is defined as:

$$z(w) = \log_{10}(f(w) \cdot 10^9)$$

where $f(w)$ is the word's probability in a large corpus. Zipf values range roughly from 1 (extremely rare) to 7 (extremely common). For multi-word terms, the token-level mean is used.

For each positive pair, a pair-level frequency score is computed as the average Zipf score of the two terms.

**Subset Definitions (Positive Pairs Only):**

- **Common pairs:** top 10th percentile of pair frequency (high Zipf scores).
- **Rare pairs:** bottom 10th percentile of pair frequency (low Zipf scores).

**Recall Gap:**

$$\Delta\text{Recall} = \text{Recall}_{\text{Common}} - \text{Recall}_{\text{Rare}}$$

A large positive $\Delta\text{Recall}$ indicates the model has a systematic disadvantage for rare/technical terms — it recovers common synonyms but misses rare ones. Ideally, $\Delta\text{Recall} \approx 0$ (frequency-invariant performance).

**Retention Rate:**

$$\text{Retention Rate} = \frac{\text{Recall}_{\text{Rare}}}{\text{Recall}_{\text{Common}}}$$

A scale-free complement to $\Delta\text{Recall}$: answers "what fraction of the model's common-word recall survives on rare words?" independently of the model's absolute performance level. Values close to **1.0** indicate low bias; values close to **0** indicate strong rare-word penalty. $\text{NaN}$ when $\text{Recall}_{\text{Common}} = 0$.

**Rare-word gap within overlap bands.** Rare terms may share fewer letters with their synonyms. To separate frequency from lexical overlap, the positive pairs are split into three character 3-gram bands ($J = 0$, $0 < J \le 0.5$, $J > 0.5$) and $\Delta\text{Recall}$ is computed inside each band with the same frequency cut-offs. The pooled gap is the average of the band gaps weighted by $1/(1/n_{\text{rare}} + 1/n_{\text{common}})$.

### 5.4 Semantic decay test

**Objective:** Test whether models can recognise synonymy when there is no lexical overlap between the two terms — i.e., when purely semantic (not surface) understanding is required. Prior work on scientific synonymy detection has shown that high performance on some datasets is partly explained by morphological resemblance rather than genuine semantic understanding (Thießen et al., 2023), which motivates testing this distinction directly.

**Subset Definitions (Positive Pairs Only):**

Using the same character 3-gram Jaccard measure as the Lexical trap test, but applied to positive pairs:

- **Easy positives:** $J(t_i, t_j) > 0.5$ — terms share substantial orthographic overlap (e.g., "psychology" / "psychological"). The model can exploit surface cues.
- **Hard positives:** $J(t_i, t_j) = 0.0$ — zero character overlap (e.g., "happiness" / "wellbeing"). The model must rely entirely on semantic representation.

**Semantic Decay ΔRecall:**

$$\Delta\text{Recall} = \text{Recall}_{\text{Hard}} - \text{Recall}_{\text{Easy}}$$

A negative ΔRecall is expected and normal — models find it harder to match semantically related but lexically different terms. However, a very large negative ΔRecall (e.g., $-0.75$) indicates the model is substantially "cheating" on easy pairs using surface cues. Near-zero ΔRecall for weaker models arises because their easy-pair recall is already low, not because they handle hard pairs well.

**Retention Rate:**

$$\text{Retention Rate} = \frac{\text{Recall}_{\text{Hard}}}{\text{Recall}_{\text{Easy}}}$$

A scale-free complement to $\Delta\text{Recall}$: answers "what fraction of easy-pair recall survives on hard pairs?" independently of the model's absolute performance level. Values close to **1.0** indicate low semantic decay; values close to **0** indicate near-total collapse on purely semantic pairs. $\text{NaN}$ when $\text{Recall}_{\text{Easy}} = 0$.

### 5.5 Structural alignment test

**Objective:** A measurement instrument satisfies the structural aspect of construct validity when the internal organisation it implies matches the internal organisation of the construct (Messick, 1995). This test assesses whether the model's internal distance metric reflects the structure of the expert-curated thesaurus graph.

**Expert Reference:** The `shortest_path` column encodes the graph distance (number of broader/narrower edges) between the two concepts of a pair. Negative pairs have a path length of at least 2 (all ancestor–descendant pairs, at any depth, are excluded), or −1 when the two concepts are not connected.

The test has two parts. Neither of them uses a correlation with the binary synonym decision, because such a correlation can increase when a pipeline wrongly merges nearby but different concepts.

**(A) Model level — similarity against thesaurus distance.** Among connected non-synonym pairs (path length ≥ 2), Spearman's ρ between the similarity score (cosine for the models, Jaccard for the baseline) and the path length. On APA, ρ is computed on a fixed random sample of 5 million pairs (standard error about 0.0005). A negative ρ means that similarity decreases as concepts are further apart in the thesaurus. The mean similarity per path length (2, 3, 4, 5–6, 7+) is reported as well. One value per model.

**(B) Pipeline level — false merges by thesaurus distance.** For each pipeline, the false-positive rate among negative pairs at each path length (2, 3, 4, 5–6, 7+, not connected). Lower is better at every distance. The share of all false merges that fall at path length 2–4 (`near_miss_share`) describes whether a pipeline's errors are near misses.

---

## 6. Implementation Details

### Reproducibility

All stochastic operations use a fixed global seed (`SEED = 42`), including:
- UMAP dimensionality reduction.
- HDBSCAN (stochastic aspects).
- Seeded clustering's random seed sampling (draw d uses seed 42 + d).
- Noise injection in the Embedding stability test (seeds are offset by run index and noise level to ensure non-overlapping random states).
- The concept-level bootstrap.

The GPU, when available, is only used to encode the terms. Similarity matrices, clustering and all metrics are computed on the CPU, so the results do not depend on the device beyond the rounding of the encoder itself.

### Computational Flow

```
unique_terms
    │
    ▼
[Model Encoding] ──────────────────────── embedding_cache (per model)
    │                                           │
    ├──► [UMAP + HDBSCAN] ──────────────────────┤
    │                                           │
    ├──► [Pairwise Cosine > τ] ─────────────────┤──► preds_store
    │                                           │
    └──► [Seeded Clustering > θ] ───────────────┘
                                                │
                    ┌───────────────────────────┤
                    │                           │
             Audit Masks                   preds_store
        (Lexical, Frequency,                    │
         Difficulty, Expert)                    │
                    │                           │
                    └──────────────────────────►│
                                                ▼
                              Diagnostic tests 1–5 (per model × technique)
```

### Key Software Dependencies

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Sentence embedding encoding |
| `torch` | Tensor operations, GPU acceleration |
| `umap-learn` | UMAP dimensionality reduction |
| `scikit-learn` | HDBSCAN density-based clustering, L2 normalisation |
| `wordfreq` | Zipf word frequency lookups (Rare-word bias test) |
| `numpy` | Array operations, metrics from confusion counts, bootstrap |
| `scipy` | Spearman correlation, sparse character n-gram matrices |
| `pyarrow` | Reading and writing the Parquet pair files |

---

## 7. Results

The diagnostic notebooks write all result tables to `analysis/results/elsst/` and `analysis/results/apa/` (primary performance with FPR and F1 at the reference prevalence, seed draws, stability, lexical trap, rare-word bias with and without overlap bands, semantic decay, structural alignment parts A and B, broader/narrower merge rates, bootstrap intervals and F1 differences, top-k recall and correct matches per reviewed candidates). Results of the initially submitted version are kept under the git tag `v1.0-submission`.

---

## 8. Design Decisions and Limitations

- **Negative pair construction:** Negative pairs cover all label combinations of concepts that are not on the same family line. Broader/narrower pairs are excluded from the benchmark and evaluated separately, so F1 does not reflect how well a pipeline separates a concept from its parent.
- **Thesaurus equivalence:** A correct match means that the thesaurus lists the two labels under one concept. This supports indexing; it does not establish that two constructs are theoretically interchangeable.
- **Seeded clustering:** Its output depends on the drawn seeds; the variation over seed draws is reported next to the mean.
- **Bootstrap scope:** The intervals reflect which concepts are in the evaluation set, not re-running the pipelines.

---

*All random operations use `SEED = 42`. Results are reproducible by running the notebooks top-to-bottom in a fresh kernel with the frozen hyperparameters from `scripts/config.py`.*
