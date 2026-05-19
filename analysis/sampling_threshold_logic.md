# Methodological Report: Data Curation & Threshold Determination

## 1. Metric Selection

**Metric:** Character-level 3-gram Jaccard Similarity.
**Justification:**

* 
**Theoretical:** This metric aligns with the **Lexical trap test**, which specifically aims to distinguish surface form (spelling) from conceptual content (meaning).


* 
**Technical:** Unlike word-level tokenization, 3-grams accurately capture morphological roots (e.g., "Social" vs. "Socialist"), making them superior for identifying "Lexical Traps" (Hard Negatives) and confirming true "Semantic Gaps" (Hard Positives).



## 2. Threshold Determination

The following thresholds classify the dataset into four distinct audit sets. These cut-offs were derived by combining the *Performance Scorecard* theoretical framework with the statistical distribution of the specific dataset.

| Category | Type | Threshold (3-gram) | Sample Count () | Definition & Purpose |
| --- | --- | --- | --- | --- |
| **Hard Positives** | Synonyms | **Exactly 0.00** | ~517 | <br>**Semantic decay test:** Pairs with identical meaning but *zero* shared character sequences (e.g., "Debts" vs "Arrears"). Tests deep semantic encoding.

 |
| **Easy Positives** | Synonyms | **> 0.50** | 152 | **Lexical Anchors:** Synonyms with high orthographic overlap. Serves as the control group for the Semantic decay test.

 |
| **Hard Negatives** | Unrelated | **> 0.50*** | 65 | **Lexical trap test (traps):** Unrelated concepts that look similar. *Note: Threshold adjusted from >0.6 to >0.5 to ensure statistical power ()*.

 |
| **Easy Negatives** | Unrelated | **< 0.10** | ~511k | **Baseline Noise:** Distinct concepts with little to no spelling overlap. The model should easily distinguish these.

 |

## 3. Statistical & Academic Justification

### Hard Negatives (> 0.50)

* 
**Protocol:** The study defines these as "Lexical Traps" or "False Positives" for the Lexical trap test.


* 
**Adjustment:** While the original text suggests a threshold of 0.6, the dataset statistics revealed this would yield only 28 samples (too volatile). Lowering the threshold to **0.50** increases the sample size to **65**, satisfying the Central Limit Theorem () for robust error estimation while maintaining high lexical similarity.



### Hard Positives (= 0.00)

* 
**Protocol:** The Semantic decay test requires measuring performance as lexical overlap vanishes ().


* **Validation:** 33.2% of the positive pairs in the dataset have a similarity of exactly 0.00. This provides a massive sample size (~517), allowing for a strict "Zero-Overlap" condition without needing to relax the threshold to <0.1.

### Easy Positives (> 0.50)

* 
**Protocol:** Defined explicitly as "Easy (Jaccard > 0.5)" in the audit methodology.


* **Validation:** This captures the top 9.8% of the distribution, creating a valid "High Similarity" control group.

---

## 4. Diagnostic Test Mapping

| Subset | Used by diagnostic test |
| --- | --- |
| Hard Negatives (Jaccard > 0.5) | Lexical trap test |
| Easy Negatives (Jaccard = 0.0) | Lexical trap test (baseline) |
| Hard Positives (Jaccard = 0.0) | Semantic decay test |
| Easy Positives (Jaccard > 0.5) | Semantic decay test (baseline) |