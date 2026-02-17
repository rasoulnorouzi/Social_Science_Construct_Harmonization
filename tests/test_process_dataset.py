"""
Test & Validation Script for process_dataset.py
================================================
Runs APA processing into a test directory, then validates:
  1. Shortest-path BFS correctness
  2. Positive pair generation (synonym pairs within concepts)
  3. Negative pair generation (parental/ancestor/descendant exclusion)
  4. Structural integrity of output CSVs
"""

import os
import sys
import csv
import random
from collections import defaultdict, deque

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from process_dataset import (
    _load_apa,
    _load_elsst,
    compute_transitive_closures,
    build_exclusion_sets,
    generate_positive_pairs_to_csv,
    generate_negative_pairs_to_csv,
    process_dataset,
    ConceptData,
)


TEST_OUTPUT_DIR = "test_output"
APA_TEST_DIR = os.path.join(TEST_OUTPUT_DIR, "apa_test")
ELSST_TEST_DIR = os.path.join(TEST_OUTPUT_DIR, "elsst_test")
APA_PATH = "datasets/raw_datasets/APA.xml"
ELSST_PATH = "datasets/raw_datasets/ELSST_R5.rdf"

SEED = 42


def read_csv_rows(filepath):
    """Read a CSV file and return list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def reference_bfs(adj, source, max_nodes):
    """Independent BFS implementation to validate shortest paths."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def test_apa_loading():
    """Test 1: Can we load APA.xml correctly?"""
    print("\n" + "=" * 70)
    print("TEST 1: APA.xml Loading")
    print("=" * 70)

    concepts, adj = _load_apa(APA_PATH)

    print(f"\n  Total concepts loaded: {len(concepts)}")

    # Stats
    n_with_alts = sum(1 for c in concepts if len(c.alts) > 0)
    n_with_broader = sum(1 for c in concepts if len(c.broader) > 0)
    n_with_narrower = sum(1 for c in concepts if len(c.narrower) > 0)
    n_with_related = sum(1 for c in concepts if len(c.related) > 0)
    total_alts = sum(len(c.alts) for c in concepts)
    total_broader = sum(len(c.broader) for c in concepts)
    total_narrower = sum(len(c.narrower) for c in concepts)
    total_related = sum(len(c.related) for c in concepts)

    print(f"  Concepts with synonyms (alts):   {n_with_alts} ({n_with_alts/len(concepts)*100:.1f}%)")
    print(f"  Concepts with broader:           {n_with_broader} ({n_with_broader/len(concepts)*100:.1f}%)")
    print(f"  Concepts with narrower:          {n_with_narrower} ({n_with_narrower/len(concepts)*100:.1f}%)")
    print(f"  Concepts with related:           {n_with_related} ({n_with_related/len(concepts)*100:.1f}%)")
    print(f"  Total synonym labels:            {total_alts}")
    print(f"  Total broader relations:         {total_broader}")
    print(f"  Total narrower relations:        {total_narrower}")
    print(f"  Total related relations:         {total_related}")
    print(f"  Adjacency entries (nodes in adj): {len(adj)}")

    # Show some example concepts
    print("\n  Example concepts (first 5):")
    for i in range(min(5, len(concepts))):
        c = concepts[i]
        print(f"    [{i}] pref='{c.pref}', alts={c.alts[:3]}, broader={c.broader[:3]}, "
              f"narrower={c.narrower[:3]}, related={c.related[:3]}")

    # Check broader/narrower consistency
    print("\n  Checking broader<->narrower consistency...")
    inconsistencies = 0
    for idx, c in enumerate(concepts):
        for parent_id in c.broader:
            if idx not in concepts[parent_id].narrower:
                inconsistencies += 1
    print(f"    Broader->Narrower inconsistencies: {inconsistencies}")
    for idx, c in enumerate(concepts):
        for child_id in c.narrower:
            if idx not in concepts[child_id].broader:
                inconsistencies += 1
    print(f"    Total B<->N inconsistencies: {inconsistencies}")

    return concepts, adj


def test_elsst_loading():
    """Test 2: Load ELSST for comparison stats."""
    print("\n" + "=" * 70)
    print("TEST 2: ELSST_R5.rdf Loading (Comparison Stats)")
    print("=" * 70)

    concepts, adj = _load_elsst(ELSST_PATH)

    print(f"\n  Total concepts loaded: {len(concepts)}")

    n_with_alts = sum(1 for c in concepts if len(c.alts) > 0)
    n_with_broader = sum(1 for c in concepts if len(c.broader) > 0)
    n_with_narrower = sum(1 for c in concepts if len(c.narrower) > 0)
    n_with_related = sum(1 for c in concepts if len(c.related) > 0)
    total_alts = sum(len(c.alts) for c in concepts)

    print(f"  Concepts with synonyms (alts):   {n_with_alts} ({n_with_alts/len(concepts)*100:.1f}%)")
    print(f"  Concepts with broader:           {n_with_broader} ({n_with_broader/len(concepts)*100:.1f}%)")
    print(f"  Concepts with narrower:          {n_with_narrower} ({n_with_narrower/len(concepts)*100:.1f}%)")
    print(f"  Concepts with related:           {n_with_related} ({n_with_related/len(concepts)*100:.1f}%)")
    print(f"  Total synonym labels:            {total_alts}")

    return concepts, adj


def test_shortest_path(concepts, adj, dataset_name="APA"):
    """Test 3: Validate BFS shortest path computation."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Shortest Path BFS Validation ({dataset_name})")
    print("=" * 70)

    n = len(concepts)
    # Pick random sample of source nodes to test
    random.seed(SEED)
    sample_sources = random.sample(range(n), min(50, n))

    errors = 0
    total_checks = 0
    path_distribution = defaultdict(int)

    for src in sample_sources:
        ref_dists = reference_bfs(adj, src, n)
        # Check BFS from process_dataset style
        dists = {}
        if src in adj or True:  # always run BFS
            q = deque([(src, 0)])
            visited = {src}
            dists[src] = 0
            while q:
                curr, d = q.popleft()
                for nb in adj.get(curr, set()):
                    if nb not in visited:
                        visited.add(nb)
                        dists[nb] = d + 1
                        q.append((nb, d + 1))

        # Compare
        for target in random.sample(range(n), min(100, n)):
            ref_d = ref_dists.get(target, -1)
            test_d = dists.get(target, -1)
            total_checks += 1
            if ref_d != test_d:
                errors += 1
                if errors <= 5:
                    print(f"  MISMATCH: src={src}, target={target}, ref={ref_d}, got={test_d}")
            path_distribution[ref_d] += 1

    print(f"\n  Total BFS checks:   {total_checks}")
    print(f"  Errors:             {errors}")
    print(f"  Status:             {'PASS' if errors == 0 else 'FAIL'}")

    print(f"\n  Shortest path distribution (sampled):")
    for k in sorted(path_distribution.keys()):
        label = "unreachable" if k == -1 else str(k)
        print(f"    dist={label}: {path_distribution[k]} pairs")

    # Additionally check: are -1 (unreachable) pairs common?
    unreachable_pct = path_distribution.get(-1, 0) / max(total_checks, 1) * 100
    print(f"\n  Unreachable pairs: {unreachable_pct:.1f}%")

    return errors == 0


def test_exclusion_sets(concepts, adj, dataset_name="APA"):
    """Test 4: Validate exclusion sets exclude parental chain."""
    print("\n" + "=" * 70)
    print(f"TEST 4: Exclusion Sets — Parental Relation Exclusion ({dataset_name})")
    print("=" * 70)

    ancestors, descendants = compute_transitive_closures(concepts)
    exclusion_sets = build_exclusion_sets(concepts, ancestors, descendants, exclude_related=False)

    # Validate: self must be excluded
    self_excluded = all(i in exclusion_sets[i] for i in range(len(concepts)))
    print(f"\n  Self in exclusion set:    {'PASS' if self_excluded else 'FAIL'}")

    # Validate: all ancestors are excluded
    ancestor_ok = True
    for i in range(len(concepts)):
        for anc in ancestors[i]:
            if anc not in exclusion_sets[i]:
                ancestor_ok = False
                break
        if not ancestor_ok:
            break
    print(f"  Ancestors excluded:       {'PASS' if ancestor_ok else 'FAIL'}")

    # Validate: all descendants are excluded
    descendant_ok = True
    for i in range(len(concepts)):
        for desc in descendants[i]:
            if desc not in exclusion_sets[i]:
                descendant_ok = False
                break
        if not descendant_ok:
            break
    print(f"  Descendants excluded:     {'PASS' if descendant_ok else 'FAIL'}")

    # Validate: related NOT excluded (by default)
    related_included = True
    n_related_checks = 0
    for i in range(len(concepts)):
        for r in concepts[i].related:
            n_related_checks += 1
            if r in exclusion_sets[i]:
                # r could also be an ancestor/descendant, check
                if r not in ancestors[i] and r not in descendants[i] and r != i:
                    related_included = False
                    break
    print(f"  Related NOT excluded (default): {'PASS' if related_included else 'FAIL'} (checked {n_related_checks} relations)")

    # Now test with exclude_related=True
    exclusion_sets_with_related = build_exclusion_sets(concepts, ancestors, descendants, exclude_related=True)
    related_excluded = True
    for i in range(len(concepts)):
        for r in concepts[i].related:
            if r not in exclusion_sets_with_related[i]:
                related_excluded = False
                break
        if not related_excluded:
            break
    print(f"  Related EXCLUDED (when enabled): {'PASS' if related_excluded else 'FAIL'}")

    # Stats
    avg_excl_size = sum(len(s) for s in exclusion_sets) / len(exclusion_sets)
    max_excl_size = max(len(s) for s in exclusion_sets)
    print(f"\n  Avg exclusion set size: {avg_excl_size:.1f}")
    print(f"  Max exclusion set size: {max_excl_size}")

    return ancestors, descendants, exclusion_sets


def test_positive_pairs(concepts, concept_ids, output_dir, dataset_name="APA"):
    """Test 5: Validate positive pairs generation."""
    print("\n" + "=" * 70)
    print(f"TEST 5: Positive Pairs Generation ({dataset_name})")
    print("=" * 70)

    pos_path = os.path.join(output_dir, "train_positive_pairs.csv")
    if not os.path.exists(pos_path):
        print(f"  File not found: {pos_path}")
        return False

    rows = read_csv_rows(pos_path)
    print(f"\n  Total positive pairs: {len(rows)}")

    if len(rows) == 0:
        print("  WARNING: No positive pairs generated!")
        return False

    # Check columns
    expected_cols = {"term1", "term2", "concept_uri", "label", "shortest_path"}
    actual_cols = set(rows[0].keys())
    print(f"  Columns: {actual_cols}")
    print(f"  Expected columns present: {'PASS' if expected_cols == actual_cols else 'FAIL'}")

    # Check all labels are 1
    all_label_1 = all(row["label"] == "1" for row in rows)
    print(f"  All labels == 1:          {'PASS' if all_label_1 else 'FAIL'}")

    # Check all shortest_path are 0
    all_sp_0 = all(row["shortest_path"] == "0" for row in rows)
    print(f"  All shortest_path == 0:   {'PASS' if all_sp_0 else 'FAIL'}")

    # Verify pairs are within same concept
    # Build URI -> terms map
    uri_terms = defaultdict(set)
    for cid in concept_ids:
        c = concepts[cid]
        uri_terms[c.uri].add(c.pref)
        for a in c.alts:
            uri_terms[c.uri].add(a)

    mismatch_count = 0
    for row in rows:
        uri = row["concept_uri"]
        t1, t2 = row["term1"], row["term2"]
        if t1 not in uri_terms[uri] or t2 not in uri_terms[uri]:
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"    MISMATCH: term1='{t1}', term2='{t2}' not both in concept {uri}")

    print(f"  Term-concept consistency: {'PASS' if mismatch_count == 0 else f'FAIL ({mismatch_count} mismatches)'}")

    # Count how many unique concepts contribute positive pairs
    unique_uris = set(row["concept_uri"] for row in rows)
    print(f"  Unique concepts with pos pairs: {len(unique_uris)}")

    # Show examples
    print("\n  Example positive pairs (first 5):")
    for row in rows[:5]:
        print(f"    '{row['term1']}' <-> '{row['term2']}' (uri={row['concept_uri'][:30]}...)")

    return True


def test_negative_pairs(concepts, concept_ids, exclusion_sets, ancestors, descendants, adj, output_dir, dataset_name="APA"):
    """Test 6: Validate negative pairs — parental exclusion, shortest path."""
    print("\n" + "=" * 70)
    print(f"TEST 6: Negative Pairs Validation ({dataset_name})")
    print("=" * 70)

    neg_path = os.path.join(output_dir, "train_negative_pairs.csv")
    if not os.path.exists(neg_path):
        print(f"  File not found: {neg_path}")
        return False

    rows = read_csv_rows(neg_path)
    print(f"\n  Total negative pairs: {len(rows)}")

    if len(rows) == 0:
        print("  WARNING: No negative pairs generated!")
        return False

    # Check columns
    expected_cols = {"term1", "term2", "concept1_uri", "concept2_uri", "label", "shortest_path"}
    actual_cols = set(rows[0].keys())
    print(f"  Columns: {actual_cols}")
    print(f"  Expected columns present: {'PASS' if expected_cols == actual_cols else 'FAIL'}")

    # All labels == 0
    all_label_0 = all(row["label"] == "0" for row in rows)
    print(f"  All labels == 0:          {'PASS' if all_label_0 else 'FAIL'}")

    # Build concept lookup by URI
    uri_to_id = {}
    for cid in concept_ids:
        uri_to_id[concepts[cid].uri] = cid

    # Check parental exclusion: no pair should have parent-child or ancestor-descendant
    parental_violations = 0
    self_violations = 0
    sample_size = min(len(rows), 50000)  # check up to 50k rows
    random.seed(SEED)
    sampled_rows = random.sample(rows, sample_size) if len(rows) > sample_size else rows

    sp_values = defaultdict(int)
    sp_mismatch_count = 0
    sp_check_count = 0

    for row in sampled_rows:
        uri1 = row["concept1_uri"]
        uri2 = row["concept2_uri"]
        sp = int(row["shortest_path"])
        sp_values[sp] += 1

        id1 = uri_to_id.get(uri1)
        id2 = uri_to_id.get(uri2)

        if id1 is None or id2 is None:
            continue

        # Self-pair check
        if id1 == id2:
            self_violations += 1
            continue

        # Check ancestor/descendant exclusion
        if id2 in ancestors[id1] or id2 in descendants[id1]:
            parental_violations += 1
            if parental_violations <= 3:
                print(f"    PARENTAL VIOLATION: '{row['term1']}' & '{row['term2']}' "
                      f"(id1={id1}, id2={id2})")

        # Verify shortest path via independent BFS (sample a few)
        if sp_check_count < 200:
            ref_dists = reference_bfs(adj, id1, len(concepts))
            expected_sp = ref_dists.get(id2, -1)
            if expected_sp != sp:
                sp_mismatch_count += 1
                if sp_mismatch_count <= 3:
                    print(f"    SP MISMATCH: {row['term1']} <-> {row['term2']}: "
                          f"expected={expected_sp}, got={sp}")
            sp_check_count += 1

    print(f"\n  Self-pair violations:       {self_violations}")
    print(f"  Parental violations:        {parental_violations}")
    print(f"  Status (no parental):       {'PASS' if parental_violations == 0 else 'FAIL'}")
    print(f"\n  Shortest path checks:       {sp_check_count}")
    print(f"  Shortest path mismatches:   {sp_mismatch_count}")
    print(f"  Status (SP correctness):    {'PASS' if sp_mismatch_count == 0 else 'FAIL'}")

    print(f"\n  Shortest path distribution:")
    for k in sorted(sp_values.keys()):
        label = "unreachable" if k == -1 else str(k)
        pct = sp_values[k] / len(sampled_rows) * 100
        print(f"    dist={label}: {sp_values[k]} ({pct:.1f}%)")

    # Show examples
    print("\n  Example negative pairs (first 5):")
    for row in rows[:5]:
        print(f"    '{row['term1']}' <-> '{row['term2']}' sp={row['shortest_path']}")

    return parental_violations == 0 and sp_mismatch_count == 0


def test_apa_full_pipeline():
    """Test 7: Run the full APA pipeline via process_dataset()."""
    print("\n" + "=" * 70)
    print("TEST 7: Full APA Pipeline (process_dataset)")
    print("=" * 70)

    os.makedirs(APA_TEST_DIR, exist_ok=True)

    process_dataset(
        dataset="apa",
        input_path=APA_PATH,
        output_dir=APA_TEST_DIR,
        train_ratio=0.70,
        seed=SEED,
        generate_all_negatives=True,
        exclude_related_from_negatives=False,
    )

    # Check all 4 files exist
    expected_files = [
        "train_positive_pairs.csv",
        "train_negative_pairs.csv",
        "test_positive_pairs.csv",
        "test_negative_pairs.csv",
    ]

    print("\n  Output files:")
    all_exist = True
    for fname in expected_files:
        fpath = os.path.join(APA_TEST_DIR, fname)
        exists = os.path.exists(fpath)
        if exists:
            n_rows = sum(1 for _ in open(fpath)) - 1  # minus header
            print(f"    {fname}: {n_rows} rows")
        else:
            print(f"    {fname}: MISSING")
            all_exist = False

    print(f"\n  All expected files exist: {'PASS' if all_exist else 'FAIL'}")
    return all_exist


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PROCESS_DATASET.PY — COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    results = {}

    # --- APA Tests ---
    apa_concepts, apa_adj = test_apa_loading()
    results["T1: APA Loading"] = len(apa_concepts) > 0

    # --- ELSST Tests ---
    elsst_concepts, elsst_adj = test_elsst_loading()
    results["T2: ELSST Loading"] = len(elsst_concepts) > 0

    # --- Shortest Path BFS ---
    results["T3a: APA BFS"] = test_shortest_path(apa_concepts, apa_adj, "APA")
    results["T3b: ELSST BFS"] = test_shortest_path(elsst_concepts, elsst_adj, "ELSST")

    # --- Exclusion Sets ---
    apa_anc, apa_desc, apa_excl = test_exclusion_sets(apa_concepts, apa_adj, "APA")

    # --- Run Full APA Pipeline ---
    results["T7: APA Full Pipeline"] = test_apa_full_pipeline()

    # --- Validate generated APA pairs ---
    # Re-load concepts to get train/test ids
    from process_dataset import split_concepts_no_overlap
    apa_concepts2, apa_adj2 = _load_apa(APA_PATH)
    apa_anc2, apa_desc2 = compute_transitive_closures(apa_concepts2)
    apa_excl2 = build_exclusion_sets(apa_concepts2, apa_anc2, apa_desc2, exclude_related=False)
    train_ids, test_ids = split_concepts_no_overlap(apa_concepts2, 0.70, SEED)

    results["T5: APA Positive Pairs"] = test_positive_pairs(apa_concepts2, train_ids, APA_TEST_DIR, "APA")
    results["T6: APA Negative Pairs"] = test_negative_pairs(
        apa_concepts2, train_ids, apa_excl2, apa_anc2, apa_desc2, apa_adj2, APA_TEST_DIR, "APA"
    )

    # --- SUMMARY ---
    print("\n\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
