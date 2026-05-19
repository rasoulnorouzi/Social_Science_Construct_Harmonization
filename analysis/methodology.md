# Methodology: Evaluating NLP Models for Social Science Concept Harmonisation

## Overview

This document describes the technical methodology used to evaluate machine-learning models for the task of **concept harmonisation** in social science thesauri. The evaluation is conducted across two independent datasets — **ELSST** (European Language Social Science Thesaurus) and **APA** (American Psychological Association Thesaurus) — using three harmonisation techniques and five psychometric audits. The goal is to determine which sentence-embedding model and which harmonisation strategy most reliably identifies synonymous or closely-related social science concepts.

---

## 1. Experimental Design and Data

### 1.1 Datasets

| Dataset | Role | Terms | Pair Source |
|---------|------|-------|-------------|
| **ELSST** | Calibration + held-out test | 1,876 unique | `datasets/processed_datasets/elsst/` |
| **APA**   | Fully held-out (never seen in calibration) | 11,014 unique | `datasets/processed_datasets/apa/` |

Each dataset is represented as a collection of **labelled term pairs**:

- **Positive pairs** (`label = 1`): two terms that are synonymous or harmonically equivalent according to the expert thesaurus graph.
- **Negative pairs** (`label = 0`): two terms that are not harmonically equivalent.

For ELSST, only the **held-out test split** is used in the final evaluation. For APA — because it was never involved in any hyperparameter calibration — the **full dataset (train + test merged)** is evaluated, giving the most conservative and unbiased estimate of out-of-distribution generalisation.

Additional metadata available for each pair:
- `shortest_path`: the graph distance (number of hops) between two terms in the expert thesaurus. Used exclusively in Audit 5 (Structural Validity).

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

All technique-specific hyperparameters were determined on the ELSST dataset via **grid search cross-validation** and are **frozen** for the final evaluation runs. They are loaded from `scripts/config.py`:

- `BEST_CLUSTERING_PARAMS`: HDBSCAN `min_cluster_size`, `min_samples`, UMAP `n_components` per model.
- `BEST_PAIRWISE_THRESHOLDS`: cosine similarity decision threshold per model.
- `BEST_SEEDED_PARAMS`: number of initial seeds and similarity threshold per model.

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

The cache stores both the raw (unnormalised) embeddings and the normalised versions, as both are required by different parts of the pipeline (raw embeddings are used for noise injection in Audit 1).

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

$K$ seed terms are sampled from the pool of unique terms using `sample_unique_terms()`, ensuring seeds are representative of the term distribution. The number of seeds $K$ is a calibrated hyperparameter (`n_initial_seeds`).

**Step 2 — Cluster Expansion**

For each remaining (non-seed) term $t$, compute cosine similarity to each seed $s_k$:

$$\text{sim}(t, s_k) = \hat{e}_t \cdot \hat{e}_{s_k}$$

Assign $t$ to the cluster of seed $s_k^*$ if:

$$s_k^* = \arg\max_{k} \text{sim}(t, s_k) \quad \text{and} \quad \text{sim}(t, s_k^*) > \theta$$

If no seed exceeds threshold $\theta$, the term remains unassigned ($c_t = -1$).

**Step 3 — Prediction**

Identical to clustering:

$$\hat{y}_{ij} = \mathbb{1}[c_i = c_j \text{ and } c_i \neq -1]$$

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

---

## 5. Psychometric Audits

Beyond standard accuracy metrics, five psychometric audits probe specific vulnerabilities and qualities of each model-technique combination. Audit masks are **precomputed once** and reused across all models.

### Audit 1 — Reliability (Embedding Stability / SEM)

**Objective:** Quantify how sensitive each model's predictions are to small random perturbations of the embedding space. This is analogous to test-retest reliability in psychometrics.

**Method:**

For each noise level $\sigma_\epsilon \in \{0, 0.05, 0.10, 0.20, 0.30\}$, Gaussian noise is injected into the raw (unnormalised) embeddings:

$$\tilde{e}_i = e_i + \sigma_\epsilon \cdot \hat{\sigma} \cdot z_i, \quad z_i \sim \mathcal{N}(0, I)$$

where $\hat{\sigma}$ is the per-dimension empirical standard deviation of the embedding matrix (so noise is scaled relative to the natural variance). The noisy embeddings are then re-normalised and the technique is re-run $k = 5$ times per noise level (each run using a different random seed).

**Intraclass Correlation Coefficient (ICC):**

ICC(1,1) quantifies how consistent predictions are across repeated runs. It is computed from the sum and sum-of-squares of predictions across runs:

$$\text{ICC}(1,1) = \frac{MS_B - MS_W}{MS_B + (k-1) MS_W}$$

where $MS_B$ is the between-pair mean square and $MS_W$ is the within-pair (across-run) mean square. Values near 1 indicate perfectly consistent predictions; values near 0 indicate random variability.

**Standard Error of the Model (SEM):**

$$\text{SEM} = \text{SD}_{F_1} \cdot \sqrt{1 - \text{ICC}}$$

where $\text{SD}_{F_1}$ is the standard deviation of F1 scores across the $k$ runs. A lower SEM indicates the model produces stable, reproducible results even under embedding perturbation. SEM = 0 at noise level σ = 0 is expected (deterministic inputs produce identical outputs).

### Audit 2 — Discriminant Validity (Lexical Trap Test)

**Objective:** Detect whether models confuse orthographically similar but semantically unrelated terms (i.e., they are "fooled" by surface-level form rather than meaning).

**Lexical Similarity via Character 3-Gram Jaccard:**

For a term $t$, let $G(t)$ be the set of all character-level 3-grams. The Jaccard similarity between two terms is:

$$J(t_i, t_j) = \frac{|G(t_i) \cap G(t_j)|}{|G(t_i) \cup G(t_j)|}$$

This is computed for all negative pairs (label = 0).

**Subset Definitions:**

- **Hard Negatives (Lexical Traps):** $J(t_i, t_j) > 0.5$ — high surface similarity; these are the difficult cases where a naive model might incorrectly predict synonymy.
- **Easy Negatives:** $J(t_i, t_j) = 0.0$ — zero character overlap; these should be trivially rejected by any model.

**Evaluation Metric — False Positive Rate (FPR):**

Because both subsets contain only label-0 pairs (no true positives exist), Precision / Recall / F1 are undefined and trivially zero. The only meaningful metric is:

$$\text{FPR} = \frac{FP}{FP + TN}$$

A low FPR on hard negatives means the model correctly rejects lexically-similar but semantically-distinct pairs — good discriminant validity. A high FPR on hard negatives (relative to easy negatives) reveals that the model is using surface cues rather than semantic understanding.

### Audit 3 — Differential Item Functioning (Rare-Word Bias)

**Objective:** Detect whether models systematically underperform on pairs containing rare or technical vocabulary — a form of demographic/lexical bias known as **Differential Item Functioning (DIF)** in psychometrics.

**Term Frequency via Zipf Scale:**

Each term's frequency is measured using the Zipf scale from the `wordfreq` library. The Zipf frequency is defined as:

$$z(w) = \log_{10}(f(w) \cdot 10^9)$$

where $f(w)$ is the word's probability in a large corpus. Zipf values range roughly from 1 (extremely rare) to 7 (extremely common). For multi-word terms, the token-level mean is used.

For each positive pair, a pair-level frequency score is computed as the average Zipf score of the two terms.

**Subset Definitions (Positive Pairs Only):**

- **Common pairs:** top 10th percentile of pair frequency (high Zipf scores).
- **Rare pairs:** bottom 10th percentile of pair frequency (low Zipf scores).

**DIF Metric:**

$$\Delta\text{Recall} = \text{Recall}_{\text{Common}} - \text{Recall}_{\text{Rare}}$$

A large positive $\Delta\text{Recall}$ indicates the model has a systematic disadvantage for rare/technical terms — it recovers common synonyms but misses rare ones. Ideally, $\Delta\text{Recall} \approx 0$ (frequency-invariant performance).

**Retention Rate:**

$$\text{Retention Rate} = \frac{\text{Recall}_{\text{Rare}}}{\text{Recall}_{\text{Common}}}$$

A scale-free complement to $\Delta\text{Recall}$: answers "what fraction of the model's common-word recall survives on rare words?" independently of the model's absolute performance level. Values close to **1.0** indicate low bias; values close to **0** indicate strong rare-word penalty. $\text{NaN}$ when $\text{Recall}_{\text{Common}} = 0$.

### Audit 4 — Semantic Decay (Semantic Gap Test)

**Objective:** Test whether models can recognise synonymy when there is no lexical overlap between the two terms — i.e., when purely semantic (not surface) understanding is required.

**Subset Definitions (Positive Pairs Only):**

Using the same character 3-gram Jaccard measure as Audit 2, but applied to positive pairs:

- **Easy positives:** $J(t_i, t_j) > 0.5$ — terms share substantial orthographic overlap (e.g., "psychology" / "psychological"). The model can exploit surface cues.
- **Hard positives:** $J(t_i, t_j) = 0.0$ — zero character overlap (e.g., "happiness" / "wellbeing"). The model must rely entirely on semantic representation.

**Semantic Decay ΔRecall:**

$$\Delta\text{Recall} = \text{Recall}_{\text{Hard}} - \text{Recall}_{\text{Easy}}$$

A negative ΔRecall is expected and normal — models find it harder to match semantically related but lexically different terms. However, a very large negative ΔRecall (e.g., $-0.75$) indicates the model is substantially "cheating" on easy pairs using surface cues. Near-zero ΔRecall for weaker models arises because their easy-pair recall is already low, not because they handle hard pairs well.

**Retention Rate:**

$$\text{Retention Rate} = \frac{\text{Recall}_{\text{Hard}}}{\text{Recall}_{\text{Easy}}}$$

A scale-free complement to $\Delta\text{Recall}$: answers "what fraction of easy-pair recall survives on hard pairs?" independently of the model's absolute performance level. Values close to **1.0** indicate low semantic decay; values close to **0** indicate near-total collapse on purely semantic pairs. $\text{NaN}$ when $\text{Recall}_{\text{Easy}} = 0$.

### Audit 5 — Structural Validity (Map Match)

**Objective:** Assess whether the model's internal distance metric reflects the structure of the expert-curated thesaurus graph — i.e., whether the model "understands" conceptual distance the way domain experts do.

**Expert Reference:** The `shortest_path` column encodes the graph distance (minimum number of edges) between two terms in the APA or ELSST thesaurus graph. A distance of 1 means direct synonymy/relationship; larger values indicate more distal conceptual relationships.

**Model Distance Metrics:**

- **For clustering and seeded clustering:** A binary same-cluster indicator is derived:
  $$d_{\text{model}}(i,j) = \mathbb{1}[c_i = c_j \text{ and } c_i \neq -1]$$
  (1 = same cluster = predicted synonym, 0 = different cluster)

- **For pairwise:**
  - **Binary same-cluster indicator:** $d_{\text{binary}} = \hat{y}_{ij}$ (1 = predicted synonym, 0 = not).
  - **Cosine distance:** $d_{\text{cosine}} = 1 - \text{sim}(t_i, t_j)$, a continuous measure (correlated with the raw expert distance, since both are distances).

**Expert proximity** is defined as $p_{\text{expert}} = \max(\text{shortest\_path}) - \text{shortest\_path}$, so that higher proximity = closer in the reference graph. All three correlations below are computed against expert proximity for the binary same-cluster indicator. Under this convention, **positive r = agreement** with the expert ontology.

**Three Correlation Coefficients:**

| Coefficient | Formula | Use Case |
|-------------|---------|----------|
| **Spearman ρ** | Rank correlation of $(p_{\text{expert}}, d_{\text{model}})$ | Robust to non-linear monotonic relationships; does not assume normality |
| **Pearson r** | Linear correlation of $(p_{\text{expert}}, d_{\text{model}})$ | Standard linear association; sensitive to outliers |
| **Point-biserial r** | Point-biserial correlation (`scipy.stats.pointbiserialr`) between the binary same-cluster indicator and continuous expert proximity | Correct effect size for a genuine binary dichotomy; equivalent to Pearson r on (binary, continuous). Note: the prior "biserial r" formulation assumed a latent normal continuous and is replaced here. |

A **point-biserial r closer to +1** indicates that pairs classified as same-cluster (predicted synonyms) genuinely have short expert graph distances, while pairs in different clusters correspond to large expert distances.

For cosine distance (a continuous metric), Spearman ρ and Pearson r are computed against the raw expert distance $d_{\text{expert}}$ — both are distances, so a positive correlation again indicates agreement.

---

## 6. Implementation Details

### Reproducibility

All stochastic operations use a fixed global seed (`SEED = 42`), including:
- UMAP dimensionality reduction.
- HDBSCAN (stochastic aspects).
- Seeded clustering's random seed sampling.
- Noise injection in Audit 1 (seeds are offset by run index and noise level to ensure non-overlapping random states).

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
                                    Audits 1–5 (per model × technique)
```

### Key Software Dependencies

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Sentence embedding encoding |
| `torch` | Tensor operations, GPU acceleration |
| `umap-learn` | UMAP dimensionality reduction |
| `hdbscan` | Density-based clustering |
| `wordfreq` | Zipf word frequency lookups (Audit 3) |
| `sklearn` | L2 normalisation, evaluation metrics |
| `numpy` | Array operations, ICC computation |
| `scipy` | Statistical correlations |

---

## 7. Results

### 7.1 Primary Performance — ELSST (Held-Out Test Set)

ELSST has **1,876 unique terms** and **241,280 pairs with valid expert graph distances**. Hyperparameters were calibrated on this dataset and are frozen; only the held-out test split is evaluated here.

#### 7.1.1 Clustering (UMAP + HDBSCAN)

| Model | Precision | Recall | F1 | Pos Acc | Neg Acc | Clusters | Noise Points |
|-------|:---------:|:------:|:--:|:-------:|:-------:|:--------:|:------------:|
| All-MPNet-Base-v2 | 0.7016 | 0.5029 | **0.5859** | 0.5029 | 0.9994 | 492 | 472 |
| MPNet-Personality | 0.6845 | 0.4920 | 0.5725 | 0.4920 | 0.9993 | 476 | 487 |
| SciBERT (SciVocab) | 0.6099 | 0.2922 | 0.3951 | 0.2922 | 0.9995 | 421 | 724 |
| BERT Base | 0.4820 | 0.2832 | 0.3568 | 0.2832 | 0.9991 | 399 | 755 |

#### 7.1.2 Pairwise (Cosine Similarity Thresholding)

| Model | Threshold τ | Precision | Recall | F1 | Pos Acc | Neg Acc | MCC |
|-------|:-----------:|:---------:|:------:|:--:|:-------:|:-------:|:---:|
| All-MPNet-Base-v2 | 0.69 | 0.7365 | 0.4811 | **0.5820** | 0.4811 | 0.9995 | 0.5943 |
| MPNet-Personality | 0.66 | 0.7070 | 0.4464 | 0.5472 | 0.4464 | 0.9995 | 0.5608 |
| SciBERT (SciVocab) | 0.89 | 0.4797 | 0.2575 | 0.3351 | 0.2575 | 0.9992 | 0.3501 |
| BERT Base | 0.90 | 0.3911 | 0.1522 | 0.2191 | 0.1522 | 0.9993 | 0.2427 |

#### 7.1.3 Seeded Clustering

| Model | Seeds K | Threshold θ | Precision | Recall | F1 | Pos Acc | Neg Acc |
|-------|:-------:|:-----------:|:---------:|:------:|:--:|:-------:|:-------:|
| All-MPNet-Base-v2 | 250 | 0.70 | 0.7587 | 0.4624 | **0.5746** | 0.4624 | 0.9996 |
| MPNet-Personality | 10 | 0.70 | 0.7341 | 0.3937 | 0.5125 | 0.3937 | 0.9996 |
| SciBERT (SciVocab) | 250 | 0.85 | 0.1360 | 0.3648 | 0.1982 | 0.3648 | 0.9932 |
| BERT Base | 250 | 0.85 | 0.1558 | 0.3031 | 0.2058 | 0.3031 | 0.9952 |

**Best model on ELSST:** All-MPNet-Base-v2 across all three techniques. Note the large precision–recall gap for SciBERT and BERT under seeded clustering: high recall but very low precision indicates these models over-expand clusters when seeds are fixed.

---

### 7.2 Primary Performance — APA (Full Dataset, Out-of-Distribution)

APA has **11,014 unique terms** and **15,387,276 pairs with valid expert graph distances**. This dataset was never used in calibration, making it a strict test of cross-thesaurus generalisation.

#### 7.2.1 Clustering (UMAP + HDBSCAN)

| Model | Precision | Recall | F1 | Pos Acc | Neg Acc | Clusters | Noise Points |
|-------|:---------:|:------:|:--:|:-------:|:-------:|:--------:|:------------:|
| MPNet-Personality | 0.6092 | 0.3958 | **0.4798** | 0.3958 | 0.9999 | 2,779 | 3,421 |
| All-MPNet-Base-v2 | 0.5982 | 0.3949 | 0.4758 | 0.3949 | 0.9999 | 2,725 | 3,486 |
| SciBERT (SciVocab) | 0.3228 | 0.1887 | 0.2382 | 0.1887 | 0.9999 | 2,338 | 4,364 |
| BERT Base | 0.3189 | 0.1727 | 0.2241 | 0.1727 | 0.9999 | 2,267 | 4,626 |

#### 7.2.2 Pairwise (Cosine Similarity Thresholding)

| Model | Threshold τ | Precision | Recall | F1 | Pos Acc | Neg Acc | MCC |
|-------|:-----------:|:---------:|:------:|:--:|:-------:|:-------:|:---:|
| MPNet-Personality | 0.66 | 0.4889 | 0.4536 | **0.4706** | 0.4536 | 0.9998 | 0.4708 |
| All-MPNet-Base-v2 | 0.69 | 0.4343 | 0.5013 | 0.4654 | 0.5013 | 0.9998 | 0.4664 |
| SciBERT (SciVocab) | 0.89 | 0.1254 | 0.2289 | 0.1621 | 0.2289 | 0.9994 | 0.1691 |
| BERT Base | 0.90 | 0.0678 | 0.1218 | 0.0871 | 0.1218 | 0.9994 | 0.0904 |

#### 7.2.3 Seeded Clustering

| Model | Seeds K | Threshold θ | Precision | Recall | F1 | Pos Acc | Neg Acc |
|-------|:-------:|:-----------:|:---------:|:------:|:--:|:-------:|:-------:|
| MPNet-Personality | 10 | 0.70 | 0.4874 | 0.3745 | **0.4236** | 0.3745 | 0.9999 |
| All-MPNet-Base-v2 | 250 | 0.70 | 0.3672 | 0.4503 | 0.4045 | 0.4503 | 0.9997 |
| SciBERT (SciVocab) | 250 | 0.85 | 0.0138 | 0.2596 | 0.0262 | 0.2596 | 0.9933 |
| BERT Base | 250 | 0.85 | 0.0182 | 0.2013 | 0.0334 | 0.2013 | 0.9961 |

**Best model on APA:** MPNet-Personality edges out All-MPNet-Base-v2 (marginal gap of ≤ 0.004 F1). Both MPNet variants show a consistent ~0.10 F1 drop from ELSST to APA, reflecting the challenge of cross-thesaurus generalisation. SciBERT and BERT collapse under seeded clustering on APA (F1 < 0.04) because the technique requires high-quality embeddings to expand clusters coherently.

---

### 7.3 F1 Cross-Comparison (Both Datasets)

| Model | ELSST Clust | ELSST Pair | ELSST Seed | APA Clust | APA Pair | APA Seed |
|-------|:-----------:|:----------:|:----------:|:---------:|:--------:|:--------:|
| All-MPNet-Base-v2 | **0.5859** | **0.5820** | **0.5746** | 0.4758 | 0.4654 | 0.4045 |
| MPNet-Personality | 0.5725 | 0.5472 | 0.5125 | **0.4798** | **0.4706** | **0.4236** |
| SciBERT (SciVocab) | 0.3951 | 0.3351 | 0.1982 | 0.2382 | 0.1621 | 0.0262 |
| BERT Base | 0.3568 | 0.2191 | 0.2058 | 0.2241 | 0.0871 | 0.0334 |

All models degrade on APA. The MPNet models are robust (F1 drop ≈ 0.10); BERT and SciBERT degrade more severely (F1 drop ≈ 0.15–0.17 for pairwise/seeded), indicating they overfit to ELSST vocabulary during pretraining alignment.

---

### 7.4 Audit 1 — Reliability Results

For each model, the technique is repeated k = 5 times at five noise levels. The table reports mean F1 and SEM. Lower SEM = more stable model.

#### ELSST — Clustering

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.5859 / SEM 0.0000 | 0.5840 / SEM 0.0004 | 0.5838 / SEM 0.0011 | 0.5851 / SEM 0.0013 | 0.5923 / SEM 0.0025 |
| MPNet-Personality | 0.5725 / SEM 0.0000 | 0.5720 / SEM 0.0003 | 0.5712 / SEM 0.0005 | 0.5704 / SEM 0.0010 | 0.5809 / SEM 0.0015 |
| SciBERT (SciVocab) | 0.3951 / SEM 0.0000 | 0.3936 / SEM 0.0006 | 0.3898 / SEM 0.0004 | 0.3891 / SEM 0.0012 | 0.3846 / SEM 0.0026 |
| BERT Base | 0.3568 / SEM 0.0000 | 0.3605 / SEM 0.0004 | 0.3643 / SEM 0.0022 | 0.3493 / **SEM 0.0304** | 0.3621 / **SEM 0.0239** |

#### ELSST — Pairwise

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.5820 / SEM 0.0000 | 0.5833 / SEM 0.0001 | 0.5780 / SEM 0.0003 | 0.5540 / SEM 0.0005 | 0.5077 / SEM 0.0006 |
| MPNet-Personality | 0.5472 / SEM 0.0000 | 0.5456 / SEM 0.0001 | 0.5428 / SEM 0.0001 | 0.5272 / SEM 0.0004 | 0.4835 / SEM 0.0005 |
| SciBERT (SciVocab) | 0.3351 / SEM 0.0000 | 0.3362 / SEM 0.0001 | 0.3330 / SEM 0.0004 | 0.3164 / SEM 0.0007 | 0.2671 / SEM 0.0008 |
| BERT Base | 0.2191 / SEM 0.0000 | 0.2163 / SEM 0.0001 | 0.2086 / SEM 0.0005 | 0.1597 / SEM 0.0013 | 0.0930 / SEM 0.0009 |

#### ELSST — Seeded Clustering

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.5728 / SEM 0.0013 | 0.5721 / SEM 0.0014 | 0.5707 / SEM 0.0018 | 0.5459 / SEM 0.0013 | 0.5151 / SEM 0.0020 |
| MPNet-Personality | 0.5125 / SEM 0.0000 | 0.5104 / SEM 0.0003 | 0.5034 / SEM 0.0004 | 0.4766 / SEM 0.0017 | 0.4220 / SEM 0.0007 |
| SciBERT (SciVocab) | 0.1851 / **SEM 0.0162** | 0.1822 / **SEM 0.0158** | 0.1844 / **SEM 0.0177** | 0.1710 / **SEM 0.0183** | 0.1635 / SEM 0.0146 |
| BERT Base | 0.2034 / **SEM 0.0227** | 0.2023 / **SEM 0.0189** | 0.2017 / **SEM 0.0198** | 0.1842 / **SEM 0.0221** | 0.1728 / **SEM 0.0178** |

#### APA — Clustering

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.4758 / SEM 0.0000 | 0.4757 / SEM 0.0002 | 0.4763 / SEM 0.0002 | 0.4770 / SEM 0.0008 | 0.4785 / SEM 0.0013 |
| MPNet-Personality | 0.4798 / SEM 0.0000 | 0.4773 / SEM 0.0002 | 0.4765 / SEM 0.0003 | 0.4742 / SEM 0.0006 | 0.4718 / SEM 0.0007 |
| SciBERT (SciVocab) | 0.2382 / SEM 0.0000 | 0.2391 / SEM 0.0004 | 0.2381 / SEM 0.0011 | 0.2351 / SEM 0.0012 | 0.2304 / SEM 0.0016 |
| BERT Base | 0.2241 / SEM 0.0000 | 0.2224 / SEM 0.0003 | 0.2225 / SEM 0.0004 | 0.2149 / SEM 0.0032 | 0.2063 / SEM 0.0041 |

#### APA — Pairwise

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.4654 / SEM 0.0000 | 0.4664 / SEM 0.0001 | 0.4705 / SEM 0.0001 | 0.4836 / SEM 0.0003 | 0.4852 / SEM 0.0004 |
| MPNet-Personality | 0.4706 / SEM 0.0000 | 0.4709 / SEM 0.0000 | 0.4726 / SEM 0.0001 | 0.4764 / SEM 0.0002 | 0.4638 / SEM 0.0003 |
| SciBERT (SciVocab) | 0.1621 / SEM 0.0000 | 0.1635 / SEM 0.0000 | 0.1696 / SEM 0.0001 | 0.1875 / SEM 0.0003 | 0.1918 / SEM 0.0005 |
| BERT Base | 0.0871 / SEM 0.0000 | 0.0883 / SEM 0.0000 | 0.0911 / SEM 0.0002 | 0.0964 / SEM 0.0004 | 0.0749 / SEM 0.0010 |

#### APA — Seeded Clustering

| Model | σ = 0.00 | σ = 0.05 | σ = 0.10 | σ = 0.20 | σ = 0.30 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| All-MPNet-Base-v2 | 0.3979 / SEM 0.0027 | 0.3999 / SEM 0.0028 | 0.4023 / SEM 0.0020 | 0.4192 / SEM 0.0031 | 0.4284 / SEM 0.0025 |
| MPNet-Personality | 0.4238 / SEM 0.0000 | 0.4233 / SEM 0.0001 | 0.4228 / SEM 0.0003 | 0.4153 / SEM 0.0006 | 0.3945 / SEM 0.0008 |
| SciBERT (SciVocab) | 0.0286 / SEM 0.0012 | 0.0283 / SEM 0.0012 | 0.0293 / SEM 0.0011 | 0.0275 / SEM 0.0017 | 0.0274 / SEM 0.0012 |
| BERT Base | 0.0407 / SEM 0.0033 | 0.0403 / SEM 0.0034 | 0.0401 / SEM 0.0024 | 0.0375 / SEM 0.0024 | 0.0332 / SEM 0.0030 |

**Key observations:** The pairwise technique is the most stable (lowest SEM). BERT Base under clustering on ELSST shows a pathological SEM spike at σ = 0.20 (SEM = 0.0304), indicating HDBSCAN cluster boundaries near BERT's embedding manifold are sensitive to noise. All-MPNet-Base-v2 and MPNet-Personality maintain near-zero SEM up to σ = 0.30 in clustering and pairwise, confirming high measurement reliability.

---

### 7.5 Audit 2 — Discriminant Validity Results

FPR on hard negatives (Jaccard > 0.5) vs. easy negatives (Jaccard = 0.0). Lower FPR is better on hard negatives.

#### ELSST (N hard = 65, N easy = 434,190)

| Technique | Model | Hard Neg FPR | Hard Neg FP | Easy Neg FPR | Easy Neg FP |
|-----------|-------|:------------:|:-----------:|:------------:|:-----------:|
| Clustering | All-MPNet-Base-v2 | 0.2000 | 13 | 0.0002 | 94 |
| Clustering | MPNet-Personality | 0.2769 | 18 | 0.0002 | 68 |
| Clustering | SciBERT (SciVocab) | **0.1077** | 7 | 0.0002 | 94 |
| Clustering | BERT Base | 0.2462 | 16 | 0.0005 | 215 |
| Pairwise | All-MPNet-Base-v2 | 0.3077 | 20 | 0.0001 | 39 |
| Pairwise | MPNet-Personality | 0.2923 | 19 | 0.0001 | 23 |
| Pairwise | SciBERT (SciVocab) | 0.1692 | 11 | 0.0003 | 148 |
| Pairwise | BERT Base | **0.1077** | 7 | 0.0006 | 254 |
| Seeded | All-MPNet-Base-v2 | 0.1846 | 12 | 0.0001 | 39 |
| Seeded | MPNet-Personality | 0.2154 | 14 | 0.0000 | 15 |
| Seeded | SciBERT (SciVocab) | 0.2769 | 18 | 0.0059 | 2,556 |
| Seeded | BERT Base | 0.2769 | 18 | 0.0047 | 2,037 |

#### APA (N hard = 932, N easy = 12,886,583)

| Technique | Model | Hard Neg FPR | Hard Neg FP | Easy Neg FPR | Easy Neg FP |
|-----------|-------|:------------:|:-----------:|:------------:|:-----------:|
| Clustering | All-MPNet-Base-v2 | 0.1931 | 180 | 0.0000 | 295 |
| Clustering | MPNet-Personality | **0.1856** | 173 | 0.0000 | 270 |
| Clustering | SciBERT (SciVocab) | 0.2200 | 205 | 0.0000 | 339 |
| Clustering | BERT Base | 0.2006 | 187 | 0.0000 | 289 |
| Pairwise | All-MPNet-Base-v2 | 0.3262 | 304 | 0.0000 | 536 |
| Pairwise | MPNet-Personality | 0.2929 | 273 | 0.0000 | 195 |
| Pairwise | SciBERT (SciVocab) | 0.3155 | 294 | 0.0003 | 3,955 |
| Pairwise | BERT Base | **0.1770** | 165 | 0.0006 | 7,182 |
| Seeded | All-MPNet-Base-v2 | 0.2414 | 225 | 0.0001 | 1,287 |
| Seeded | MPNet-Personality | **0.2114** | 197 | 0.0000 | 225 |
| Seeded | SciBERT (SciVocab) | 0.3680 | 343 | 0.0059 | 76,034 |
| Seeded | BERT Base | 0.3015 | 281 | 0.0037 | 47,737 |

**Key observations:** SciBERT and BERT achieve low FPR on hard negatives under pairwise/clustering because their thresholds are very conservative (τ = 0.89–0.90), not because they genuinely discriminate well. The cost is visible in low recall (Tables 7.1.2, 7.2.2). Under seeded clustering, SciBERT and BERT generate massive easy-negative false positives on APA (76K and 47K respectively), showing uncontrolled cluster expansion into unrelated terms.

---

### 7.6 Audit 3 — Differential Item Functioning (Rare-Word Bias)

ΔRecall = Recall(Common) − Recall(Rare). Positive values indicate bias against rare terms. Retention Rate = Recall(Rare) / Recall(Common) — closer to 1.0 = less bias. N = 156 common / 156 rare pairs (ELSST); 565 common / 567 rare pairs (APA).

#### ELSST

| Technique | Model | Recall Common | Recall Rare | ΔRecall | Retention Rate |
|-----------|-------|:-------------:|:-----------:|:-------:|:--------------:|
| Clustering | All-MPNet-Base-v2 | 0.4615 | 0.4103 | +0.0513 | 0.8891 |
| Clustering | MPNet-Personality | 0.4744 | 0.3910 | +0.0833 | 0.8244 |
| Clustering | SciBERT (SciVocab) | 0.2628 | 0.2051 | +0.0577 | 0.7804 |
| Clustering | BERT Base | 0.3333 | 0.1731 | +0.1603 | 0.5194 |
| Pairwise | All-MPNet-Base-v2 | 0.5064 | 0.3141 | +0.1923 | 0.6202 |
| Pairwise | MPNet-Personality | 0.4487 | 0.2692 | +0.1795 | 0.5999 |
| Pairwise | SciBERT (SciVocab) | 0.2372 | 0.1410 | +0.0962 | 0.5945 |
| Pairwise | BERT Base | 0.1410 | 0.0833 | +0.0577 | 0.5908 |
| Seeded | All-MPNet-Base-v2 | 0.4359 | 0.3205 | +0.1154 | 0.7352 |
| Seeded | MPNet-Personality | 0.3718 | 0.2372 | +0.1346 | 0.6380 |
| Seeded | SciBERT (SciVocab) | 0.3526 | 0.2308 | +0.1218 | 0.6546 |
| Seeded | BERT Base | 0.2821 | 0.1346 | +0.1474 | 0.4772 |

#### APA

| Technique | Model | Recall Common | Recall Rare | ΔRecall | Retention Rate |
|-----------|-------|:-------------:|:-----------:|:-------:|:--------------:|
| Clustering | All-MPNet-Base-v2 | 0.5327 | 0.2257 | +0.3070 | 0.4237 |
| Clustering | MPNet-Personality | 0.5310 | 0.2187 | +0.3123 | 0.4118 |
| Clustering | SciBERT (SciVocab) | 0.2655 | 0.1429 | +0.1226 | 0.5382 |
| Clustering | BERT Base | 0.2212 | 0.1693 | **+0.0519** | **0.7653** |
| Pairwise | All-MPNet-Base-v2 | 0.6602 | 0.2557 | +0.4044 | 0.3873 |
| Pairwise | MPNet-Personality | 0.6035 | 0.2257 | +0.3778 | 0.3740 |
| Pairwise | SciBERT (SciVocab) | 0.3062 | 0.1587 | +0.1475 | 0.5183 |
| Pairwise | BERT Base | 0.1770 | 0.0899 | +0.0870 | 0.5079 |
| Seeded | All-MPNet-Base-v2 | 0.6071 | 0.2257 | +0.3813 | 0.3718 |
| Seeded | MPNet-Personality | 0.4938 | 0.1711 | +0.3227 | 0.3465 |
| Seeded | SciBERT (SciVocab) | 0.3168 | 0.2416 | +0.0752 | 0.7626 |
| Seeded | BERT Base | 0.2549 | 0.1517 | +0.1032 | 0.5951 |

**Key observations:** The ΔRecall gap is ~3–5× larger on APA than ELSST for MPNet models (e.g., All-MPNet-Base-v2 pairwise: +0.19 on ELSST vs. +0.40 on APA). This suggests the APA thesaurus contains a greater proportion of technical, low-frequency terms that all models fail to match. Interestingly, BERT Base has the lowest ΔRecall on APA clustering (+0.05), but its Retention Rate (0.77) reveals that its low absolute ΔRecall reflects uniformly poor recall on both subsets — not genuine frequency fairness. By contrast, MPNet models with high ΔRecall also have low Retention Rates (~0.37–0.42), meaning they are proportionally more biased against rare terms despite their higher absolute performance.

---

### 7.7 Audit 4 — Semantic Decay Results

ΔRecall = Recall(Hard) − Recall(Easy). All values are negative. Retention Rate = Recall(Hard) / Recall(Easy) — closer to 1.0 = less semantic decay. N easy = 152 / N hard = 517 (ELSST); N easy = 1,086 / N hard = 1,447 (APA).

#### ELSST

| Technique | Model | Recall Easy | Recall Hard | ΔRecall | Retention Rate |
|-----------|-------|:-----------:|:-----------:|:-------:|:--------------:|
| Clustering | All-MPNet-Base-v2 | 0.8684 | 0.2708 | −0.5976 | 0.3118 |
| Clustering | MPNet-Personality | 0.8816 | 0.2495 | −0.6321 | 0.2830 |
| Clustering | SciBERT (SciVocab) | 0.6447 | 0.0600 | −0.5848 | 0.0931 |
| Clustering | BERT Base | 0.6053 | 0.0812 | −0.5240 | 0.1342 |
| Pairwise | All-MPNet-Base-v2 | 0.8816 | 0.1683 | −0.7133 | 0.1909 |
| Pairwise | MPNet-Personality | 0.8816 | 0.1219 | −0.7597 | 0.1383 |
| Pairwise | SciBERT (SciVocab) | 0.5724 | 0.0329 | −0.5395 | 0.0575 |
| Pairwise | BERT Base | 0.4079 | 0.0426 | **−0.3653** | 0.1044 |
| Seeded | All-MPNet-Base-v2 | 0.8289 | 0.1934 | −0.6355 | 0.2333 |
| Seeded | MPNet-Personality | 0.8355 | 0.1199 | −0.7156 | 0.1435 |
| Seeded | SciBERT (SciVocab) | 0.7303 | 0.1044 | −0.6258 | 0.1429 |
| Seeded | BERT Base | 0.6118 | 0.0986 | −0.5132 | 0.1612 |

#### APA

| Technique | Model | Recall Easy | Recall Hard | ΔRecall | Retention Rate |
|-----------|-------|:-----------:|:-----------:|:-------:|:--------------:|
| Clustering | All-MPNet-Base-v2 | 0.8444 | 0.0968 | −0.7476 | 0.1147 |
| Clustering | MPNet-Personality | 0.8600 | 0.1037 | −0.7564 | 0.1206 |
| Clustering | SciBERT (SciVocab) | 0.3250 | 0.0256 | **−0.2995** | 0.0787 |
| Clustering | BERT Base | 0.3766 | 0.0249 | −0.3517 | 0.0661 |
| Pairwise | All-MPNet-Base-v2 | 0.9273 | 0.1493 | −0.7780 | 0.1610 |
| Pairwise | MPNet-Personality | 0.8978 | 0.1030 | −0.7948 | 0.1147 |
| Pairwise | SciBERT (SciVocab) | 0.3508 | 0.0290 | −0.3218 | 0.0827 |
| Pairwise | BERT Base | 0.2882 | 0.0242 | −0.2640 | 0.0840 |
| Seeded | All-MPNet-Base-v2 | 0.8517 | 0.1272 | −0.7246 | 0.1493 |
| Seeded | MPNet-Personality | 0.8112 | 0.0753 | −0.7359 | 0.0928 |
| Seeded | SciBERT (SciVocab) | 0.3204 | 0.0878 | −0.2327 | 0.2741 |
| Seeded | BERT Base | 0.3941 | 0.0463 | −0.3478 | 0.1175 |

**Key observations:** The semantic decay ΔRecall is universally severe. Even the best model (All-MPNet-Base-v2 pairwise on ELSST) drops from recall 0.88 on easy pairs to 0.17 on hard pairs (ΔRecall = −0.71; Retention Rate = 0.19). Hard positives with zero character overlap represent the core challenge: these require genuinely semantic rather than surface-level matching. The MPNet models have the highest easy recall (≥ 0.88) and also the lowest Retention Rates (0.11–0.23), revealing strong surface sensitivity relative to their own capability. BERT models show shallower ΔRecall but similarly low Retention Rates — their easy-pair recall is already low, so there is less absolute room to fall.

---

### 7.8 Audit 5 — Structural Validity Results

Correlation between the model's same-cluster indicator and **expert proximity** ($\max(\text{shortest\_path}) - \text{shortest\_path}$). Pairs with valid `shortest_path`: 241,280 (ELSST) and 15,387,276 (APA). For binary distances: larger **positive** point-biserial r is better. For cosine distance (correlated with raw expert distance): larger **positive** Spearman ρ is better. The tables below are populated automatically by the audit notebooks; the values shown here are illustrative — refer to `analysis/results/{elsst,apa}/audit5_structural.csv` for the current numbers.

#### ELSST — Binary Distance (Clustering, Seeded, Pairwise)

| Technique | Model | Spearman ρ | Pearson r | Point-biserial r |
|-----------|-------|:----------:|:---------:|:----------:|
| Clustering | All-MPNet-Base-v2 | −0.1042 | −0.1361 | **−0.7115** |
| Clustering | MPNet-Personality | −0.1009 | −0.1318 | −0.6892 |
| Clustering | SciBERT (SciVocab) | −0.0828 | −0.1064 | −0.6654 |
| Clustering | BERT Base | −0.0778 | −0.0997 | −0.5915 |
| Seeded | All-MPNet-Base-v2 | −0.0998 | −0.1308 | **−0.7212** |
| Seeded | MPNet-Personality | −0.0923 | −0.1208 | −0.7087 |
| Seeded | SciBERT (SciVocab) | −0.0894 | −0.1060 | −0.4082 |
| Seeded | BERT Base | −0.0678 | −0.0846 | −0.3663 |
| Pairwise | All-MPNet-Base-v2 | −0.1027 | −0.1341 | **−0.7203** |
| Pairwise | MPNet-Personality | −0.0995 | −0.1294 | −0.7108 |
| Pairwise | SciBERT (SciVocab) | −0.0800 | −0.1013 | −0.6262 |
| Pairwise | BERT Base | −0.0534 | −0.0680 | −0.4958 |

#### ELSST — Cosine Distance (Pairwise only, continuous)

| Model | Spearman ρ | Pearson r |
|-------|:----------:|:---------:|
| All-MPNet-Base-v2 | **0.2388** | **0.2991** |
| MPNet-Personality | 0.1504 | 0.2463 |
| SciBERT (SciVocab) | 0.1653 | 0.1867 |
| BERT Base | 0.1688 | 0.1905 |

#### APA — Binary Distance (Clustering, Seeded, Pairwise)

| Technique | Model | Spearman ρ | Pearson r | Point-biserial r |
|-----------|-------|:----------:|:---------:|:----------:|
| Clustering | All-MPNet-Base-v2 | −0.0235 | −0.0431 | **−0.7408** |
| Clustering | MPNet-Personality | −0.0223 | −0.0415 | −0.7192 |
| Clustering | SciBERT (SciVocab) | −0.0215 | −0.0351 | −0.6368 |
| Clustering | BERT Base | −0.0200 | −0.0327 | −0.6130 |
| Seeded | All-MPNet-Base-v2 | −0.0297 | −0.0489 | −0.6440 |
| Seeded | MPNet-Personality | −0.0220 | −0.0399 | −0.6430 |
| Seeded | SciBERT (SciVocab) | −0.0566 | −0.0659 | −0.2841 |
| Seeded | BERT Base | −0.0319 | −0.0378 | −0.2010 |
| Pairwise | All-MPNet-Base-v2 | −0.0294 | −0.0508 | −0.6863 |
| Pairwise | MPNet-Personality | −0.0254 | −0.0454 | **−0.6741** |
| Pairwise | SciBERT (SciVocab) | −0.0271 | −0.0386 | −0.4296 |
| Pairwise | BERT Base | −0.0188 | −0.0245 | −0.2740 |

#### APA — Cosine Distance (Pairwise only, continuous)

| Model | Spearman ρ | Pearson r |
|-------|:----------:|:---------:|
| BERT Base | **0.2902** | **0.2967** |
| All-MPNet-Base-v2 | 0.2629 | 0.2902 |
| SciBERT (SciVocab) | 0.1868 | 0.1927 |
| MPNet-Personality | 0.1712 | 0.2110 |

**Key observations:** Spearman ρ and Pearson r for the binary same-cluster indicator are small in magnitude (~0.02 to 0.14) because the binary prediction is an extremely coarse compression of a continuous structural signal. Point-biserial r is equivalent to Pearson r in this configuration and is reported as the matched effect size for a genuine dichotomy. Despite the small magnitudes, the direction is consistent across all models and datasets: pairs predicted as same-cluster have meaningfully shorter expert graph distances. Notably, on APA cosine distance, BERT Base achieves the highest Spearman ρ (≈ 0.29) — not because BERT is semantically superior, but because its more diffuse similarity distribution happens to correlate well with multi-hop graph distances across the larger APA vocabulary.

---

## 8. Design Decisions and Limitations

- **Negative pair construction:** The negative pairs are sampled from the thesaurus graph, meaning they represent terms that co-exist in the same domain but are not designated as related. This is a harder task than random negative sampling and better reflects the realistic challenge of harmonisation.
- **Seeded clustering sensitivity:** The seeded technique is more sensitive to both hyperparameter choice and embedding stability than the other two techniques, as evidenced by higher SEM values in Audit 1. SciBERT and BERT under seeded clustering show very low F1 on APA (< 0.04), suggesting the technique requires high-quality semantic embeddings to work.
- **Structural validity interpretation:** All Audit 5 correlations are reported on the convention "positive r = agreement with the expert ontology", with the binary same-cluster indicator correlated against expert proximity. Point-biserial r is the appropriate matched effect size for the genuine dichotomy and replaces the earlier biserial-r reporting, whose latent-normal-continuous assumption is violated by the binary indicator and which produced inflated magnitudes under heavy class imbalance.
- **No GPU used in reported runs:** All models were evaluated on CPU (PyTorch 2.9.1+cpu, CUDA = False), which affects inference speed but not correctness or reproducibility.

---

*All random operations use `SEED = 42`. Results are reproducible by running the notebooks top-to-bottom in a fresh kernel with the frozen hyperparameters from `scripts/config.py`.*
