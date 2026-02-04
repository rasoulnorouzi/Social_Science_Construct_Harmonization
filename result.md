============================================================
  LOADING TEST DATA
============================================================
Loading test datasets with the_loading...
  Positive pairs: 1557
  Negative pairs: 530219
  Total samples:  531776
  Dataset includes shortest_path distances from ELSST graph.

--- Precomputing Audit Masks ---
  Lexical trap subset (Jaccard > 0.6 negatives): 6
  Easy pairs (Jaccard > 0.5): 55
  Hard pairs (Jaccard == 0): 762
  Common pairs (top 10% freq): 156
  Rare pairs (bottom 10% freq): 156

============================================================
  LOADING MODEL EMBEDDINGS
============================================================
Loading embeddings once for reuse across all techniques and audits...

  Loading all-mpnet-base-v2... Loading Model (all-mpnet-base-v2)...
  Encoding 1876 unique terms...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:09<00:00, 12.16it/s]
Done
  Loading dwulff/mpnet-personality... Loading Model (dwulff/mpnet-personality)...
  Encoding 1876 unique terms...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:08<00:00, 14.51it/s]
Done
  Loading allenai/scibert_scivocab_uncased... Loading Model (allenai/scibert_scivocab_uncased)...
  Encoding 1876 unique terms...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:08<00:00, 14.18it/s]
Done
  Loading bert-base-uncased... Loading Model (bert-base-uncased)...
  Encoding 1876 unique terms...
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:08<00:00, 14.41it/s]
Done

============================================================
  FINAL EVALUATION — CLUSTERING
============================================================
Model                            | Precision | Recall | F1     | TP  | FP  | TN     | FN  
---------------------------------+-----------+--------+--------+-----+-----+--------+-----
all-mpnet-base-v2                | 0.7016    | 0.5029 | 0.5859 | 783 | 333 | 529886 | 774 
dwulff/mpnet-personality         | 0.6845    | 0.4920 | 0.5725 | 766 | 353 | 529866 | 791 
allenai/scibert_scivocab_uncased | 0.6099    | 0.2922 | 0.3951 | 455 | 291 | 529928 | 1102
bert-base-uncased                | 0.4820    | 0.2832 | 0.3568 | 441 | 474 | 529745 | 1116

============================================================
  FINAL EVALUATION — PAIRWISE
============================================================
Model                            | Precision | Recall | F1     | TP  | FP  | TN     | FN  
---------------------------------+-----------+--------+--------+-----+-----+--------+-----
all-mpnet-base-v2                | 0.7365    | 0.4811 | 0.5820 | 749 | 268 | 529951 | 808 
dwulff/mpnet-personality         | 0.7070    | 0.4464 | 0.5472 | 695 | 288 | 529931 | 862 
allenai/scibert_scivocab_uncased | 0.4797    | 0.2575 | 0.3351 | 401 | 435 | 529784 | 1156
bert-base-uncased                | 0.3911    | 0.1522 | 0.2191 | 237 | 369 | 529850 | 1320

============================================================
  FINAL EVALUATION — SEEDED CLUSTERING
============================================================
  Unclustered (new seeds created): 872
  Unclustered (new seeds created): 1238
  Unclustered (new seeds created): 400
  Unclustered (new seeds created): 746
Model                            | Precision | Recall | F1     | TP  | FP   | TN     | FN  
---------------------------------+-----------+--------+--------+-----+------+--------+-----
all-mpnet-base-v2                | 0.7587    | 0.4624 | 0.5746 | 720 | 229  | 529990 | 837 
dwulff/mpnet-personality         | 0.7341    | 0.3937 | 0.5125 | 613 | 222  | 529997 | 944 
allenai/scibert_scivocab_uncased | 0.1360    | 0.3648 | 0.1982 | 568 | 3607 | 526612 | 989 
bert-base-uncased                | 0.1558    | 0.3031 | 0.2058 | 472 | 2558 | 527661 | 1085

============================================================
  AUDIT 1 — RELIABILITY (STABILITY TEST)
============================================================

What this means:
  Measures Standard Error of Model (SEM) under stochastic embedding noise.
  Tests robustness across noise levels: 0.0 (baseline), 0.05, 0.1, 0.2, 0.3
  SEM = SD_F1 * sqrt(1 - ICC), where ICC = Intraclass Correlation Coefficient.
  Lower SEM = more stable model. 5 runs per noise level.


--- Technique: CLUSTERING ---
  Testing all-mpnet-base-v2... Done
  Testing dwulff/mpnet-personality... Done
  Testing allenai/scibert_scivocab_uncased... Done
  Testing bert-base-uncased... Done
Model                            | F1@0.0 | F1@0.05 | F1@0.1 | F1@0.2 | F1@0.3 | Best     
---------------------------------+--------+---------+--------+--------+--------+----------
all-mpnet-base-v2                | 0.5859 | 0.5850  | 0.5860 | 0.5863 | 0.5964 | Noise=0.3
dwulff/mpnet-personality         | 0.5725 | 0.5723  | 0.5719 | 0.5698 | 0.5792 | Noise=0.3
allenai/scibert_scivocab_uncased | 0.3951 | 0.3947  | 0.3905 | 0.3898 | 0.3885 | Baseline 
bert-base-uncased                | 0.3568 | 0.3600  | 0.3639 | 0.3655 | 0.3766 | Noise=0.3


--- Technique: PAIRWISE ---
  Testing all-mpnet-base-v2... Done
  Testing dwulff/mpnet-personality... Done
  Testing allenai/scibert_scivocab_uncased... Done
  Testing bert-base-uncased... Done
Model                            | F1@0.0 | F1@0.05 | F1@0.1 | F1@0.2 | F1@0.3 | Best      
---------------------------------+--------+---------+--------+--------+--------+-----------
all-mpnet-base-v2                | 0.5820 | 0.5832  | 0.5765 | 0.5537 | 0.5077 | Noise=0.05
dwulff/mpnet-personality         | 0.5472 | 0.5457  | 0.5430 | 0.5275 | 0.4841 | Baseline  
allenai/scibert_scivocab_uncased | 0.3351 | 0.3363  | 0.3339 | 0.3154 | 0.2662 | Noise=0.05
bert-base-uncased                | 0.2191 | 0.2165  | 0.2093 | 0.1597 | 0.0940 | Baseline  


--- Technique: SEEDED ---
  Testing all-mpnet-base-v2... Done
  Testing dwulff/mpnet-personality... Done
  Testing allenai/scibert_scivocab_uncased... Done
  Testing bert-base-uncased... Done
Model                            | F1@0.0 | F1@0.05 | F1@0.1 | F1@0.2 | F1@0.3 | Best      
---------------------------------+--------+---------+--------+--------+--------+-----------
all-mpnet-base-v2                | 0.5729 | 0.5687  | 0.5686 | 0.5445 | 0.5155 | Baseline  
dwulff/mpnet-personality         | 0.5122 | 0.5106  | 0.5044 | 0.4782 | 0.4209 | Baseline  
allenai/scibert_scivocab_uncased | 0.1979 | 0.1654  | 0.1525 | 0.1659 | 0.1525 | Baseline  
bert-base-uncased                | 0.2001 | 0.2002  | 0.1941 | 0.1878 | 0.1584 | Noise=0.05


============================================================
  AUDIT 2 — DISCRIMINANT VALIDITY (LEXICAL TRAP)
============================================================

What this means:
  Measures False Positive Rate on negative pairs with high lexical similarity.
  FPR = FP / (FP + TN). Lower is better (avoids 'lexical traps').
  Subset: negative pairs with Jaccard > 0.6 (N = 6)

Technique  | Model                            | Subset_N | FP | TN | FPR   
-----------+----------------------------------+----------+----+----+-------
clustering | all-mpnet-base-v2                | 6        | 4  | 2  | 0.6667
clustering | dwulff/mpnet-personality         | 6        | 4  | 2  | 0.6667
clustering | allenai/scibert_scivocab_uncased | 6        | 2  | 4  | 0.3333
clustering | bert-base-uncased                | 6        | 4  | 2  | 0.6667
pairwise   | all-mpnet-base-v2                | 6        | 5  | 1  | 0.8333
pairwise   | dwulff/mpnet-personality         | 6        | 5  | 1  | 0.8333
pairwise   | allenai/scibert_scivocab_uncased | 6        | 3  | 3  | 0.5000
pairwise   | bert-base-uncased                | 6        | 2  | 4  | 0.3333
seeded     | all-mpnet-base-v2                | 6        | 4  | 2  | 0.6667
seeded     | dwulff/mpnet-personality         | 6        | 5  | 1  | 0.8333
seeded     | allenai/scibert_scivocab_uncased | 6        | 4  | 2  | 0.6667
seeded     | bert-base-uncased                | 6        | 5  | 1  | 0.8333

============================================================
  AUDIT 3 — STRUCTURAL VALIDITY (MAP MATCH)
============================================================

What this means:
  Correlates model distances with expert ELSST graph shortest paths.
  Clustering/Seeded: binary distance (0=same cluster, 1=different).
  Pairwise: binary component, model-graph shortest path, and cosine distance.
  Uses Spearman rank correlation (ρ). Higher = better structural alignment.

  Pairs with valid expert distances: 241280

Technique  | Model                            | Spearman ρ                                          
-----------+----------------------------------+-----------------------------------------------------
clustering | all-mpnet-base-v2                | binary_dist=0.1042                                  
clustering | dwulff/mpnet-personality         | binary_dist=0.1009                                  
clustering | allenai/scibert_scivocab_uncased | binary_dist=0.0828                                  
clustering | bert-base-uncased                | binary_dist=0.0778                                  
seeded     | all-mpnet-base-v2                | binary_dist=0.0998                                  
seeded     | dwulff/mpnet-personality         | binary_dist=0.0923                                  
seeded     | allenai/scibert_scivocab_uncased | binary_dist=0.0894                                  
seeded     | bert-base-uncased                | binary_dist=0.0678                                  
pairwise   | all-mpnet-base-v2                | bin_comp=-0.1143 | graph_sp=0.5930 | cos_dist=0.2388
pairwise   | dwulff/mpnet-personality         | bin_comp=-0.1080 | graph_sp=0.4488 | cos_dist=0.1504
pairwise   | allenai/scibert_scivocab_uncased | bin_comp=-0.0704 | graph_sp=0.4502 | cos_dist=0.1653
pairwise   | bert-base-uncased                | bin_comp=-0.0513 | graph_sp=0.4060 | cos_dist=0.1688

============================================================
  AUDIT 4 — DIFFERENTIAL ITEM FUNCTIONING (RARE WORD TEST)
============================================================

What this means:
  Detects bias against technical/rare terminology.
  ΔRecall = Recall_Common - Recall_Rare. Higher gap = more bias against rare terms.
  Common = top 10% freq (156), Rare = bottom 10% freq (156)

Technique  | Model                            | Recall_Common | Recall_Rare | ΔRecall | N_Common | N_Rare
-----------+----------------------------------+---------------+-------------+---------+----------+-------
clustering | all-mpnet-base-v2                | 0.4615        | 0.4103      | 0.0513  | 156      | 156   
clustering | dwulff/mpnet-personality         | 0.4744        | 0.3910      | 0.0833  | 156      | 156   
clustering | allenai/scibert_scivocab_uncased | 0.2628        | 0.2051      | 0.0577  | 156      | 156   
clustering | bert-base-uncased                | 0.3333        | 0.1731      | 0.1603  | 156      | 156   
pairwise   | all-mpnet-base-v2                | 0.5064        | 0.3141      | 0.1923  | 156      | 156   
pairwise   | dwulff/mpnet-personality         | 0.4487        | 0.2692      | 0.1795  | 156      | 156   
pairwise   | allenai/scibert_scivocab_uncased | 0.2372        | 0.1410      | 0.0962  | 156      | 156   
pairwise   | bert-base-uncased                | 0.1410        | 0.0833      | 0.0577  | 156      | 156   
seeded     | all-mpnet-base-v2                | 0.4359        | 0.3205      | 0.1154  | 156      | 156   
seeded     | dwulff/mpnet-personality         | 0.3718        | 0.2372      | 0.1346  | 156      | 156   
seeded     | allenai/scibert_scivocab_uncased | 0.3526        | 0.2308      | 0.1218  | 156      | 156   
seeded     | bert-base-uncased                | 0.2821        | 0.1346      | 0.1474  | 156      | 156   

============================================================
  AUDIT 5 — SEMANTIC DECAY (SEMANTIC GAP TEST)
============================================================

What this means:
  Tests robustness as keyword overlap vanishes.
  Slope = (Recall_Hard - Recall_Easy) / ΔDifficulty.
  Negative slope = performance degrades on harder (no-overlap) pairs.
  Easy = Jaccard > 0.5 (55), Hard = Jaccard == 0 (762)

Technique  | Model                            | Recall_Easy | Recall_Hard | Slope   | N_Easy | N_Hard
-----------+----------------------------------+-------------+-------------+---------+--------+-------
clustering | all-mpnet-base-v2                | 0.9091      | 0.3438      | -0.5653 | 55     | 762   
clustering | dwulff/mpnet-personality         | 0.9091      | 0.3412      | -0.5679 | 55     | 762   
clustering | allenai/scibert_scivocab_uncased | 0.7091      | 0.1089      | -0.6002 | 55     | 762   
clustering | bert-base-uncased                | 0.7273      | 0.0853      | -0.6420 | 55     | 762   
pairwise   | all-mpnet-base-v2                | 0.9636      | 0.2520      | -0.7117 | 55     | 762   
pairwise   | dwulff/mpnet-personality         | 0.9636      | 0.2205      | -0.7432 | 55     | 762   
pairwise   | allenai/scibert_scivocab_uncased | 0.6545      | 0.0643      | -0.5902 | 55     | 762   
pairwise   | bert-base-uncased                | 0.3818      | 0.0394      | -0.3424 | 55     | 762   
seeded     | all-mpnet-base-v2                | 0.9091      | 0.2795      | -0.6296 | 55     | 762   
seeded     | dwulff/mpnet-personality         | 0.9273      | 0.2178      | -0.7094 | 55     | 762   
seeded     | allenai/scibert_scivocab_uncased | 0.8364      | 0.1575      | -0.6789 | 55     | 762   
seeded     | bert-base-uncased                | 0.6364      | 0.1194      | -0.5169 | 55     | 762   

============================================================
  EVALUATION COMPLETE
============================================================

All models evaluated across 3 techniques with 5 audits.
Results are printed above in formatted tables.
No graph loading required — shortest_path data is included in test CSVs.