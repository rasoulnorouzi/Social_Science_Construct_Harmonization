"""
ELSST RDF -> Train/Test synonym pairs (+ taxonomy-safe negatives)
==============================================================
OPTIMIZED VERSION
- Speed improvements: Integer IDs, Vectorized Parsing, SSSP BFS.
- Stability: Sorts URIs and Labels to ensure deterministic Train/Test splits.
"""

from __future__ import annotations

import os
import csv
import random
import gc
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Iterable
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.term import URIRef, Literal

# Progress bar
from tqdm import tqdm

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RDF_FILE_PATH = str(PROJECT_ROOT / "datasets/raw_datasets/ELSST_R5.rdf")
OUTPUT_DIR = str(PROJECT_ROOT / "datasets/processed_datasets/elsst")

LANGUAGE = "en"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70

# Negatives can explode quadratically.
GENERATE_ALL_NEGATIVES = True
NEGATIVE_SAMPLES_PER_CONCEPT = 50  # used only if GENERATE_ALL_NEGATIVES=False

# If True, negatives will also exclude "skos:related" links.
EXCLUDE_SKOS_RELATED_FROM_NEGATIVES = False

# Batch size for writing to CSV to save memory
WRITE_BATCH_SIZE = 50000

# =========================
# RDF / SKOS NAMESPACES
# =========================

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDF  = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# =========================
# DATA STRUCTURES
# =========================

# We use Integer IDs internally for speed.
ConceptID = int

@dataclass
class ConceptData:
    uri: str
    pref: str
    alts: List[str]
    # We store relations as lists of integer IDs
    broader: List[ConceptID] = field(default_factory=list)
    narrower: List[ConceptID] = field(default_factory=list)
    related: List[ConceptID] = field(default_factory=list)

# =========================
# OPTIMIZED PARSING
# =========================

def load_concepts_optimized(rdf_path: str, lang: str) -> Tuple[List[ConceptData], Dict[int, Set[int]]]:
    """
    Parses RDF and returns:
    1. A list of ConceptData (index = ConceptID)
    2. An adjacency list (undirected) for shortest path calc: adj[id] = {neighbor_ids}
    """
    g = Graph()
    print("[1/8] Loading RDF into memory...")
    g.parse(rdf_path)

    print("  Indexing Concept URIs...")
    # 1. Identify all skos:Concept subjects
    concept_uris_set = set(g.subjects(RDF.type, SKOS.Concept))

    # Map URI -> Integer ID
    # Sort for deterministic ID assignment across runs
    sorted_uris = sorted(list(concept_uris_set))
    uri_to_id = {u: i for i, u in enumerate(sorted_uris)}
    id_to_uri = {i: u for i, u in enumerate(sorted_uris)}
    n_concepts = len(sorted_uris)

    # Initialize storage
    # Using arrays/lists indexed by ID is faster than dicts
    prefs: List[Optional[str]] = [None] * n_concepts
    alts: List[List[str]] = [[] for _ in range(n_concepts)]
    broader: List[List[int]] = [[] for _ in range(n_concepts)]
    related: List[List[int]] = [[] for _ in range(n_concepts)]
    narrower: List[List[int]] = [[] for _ in range(n_concepts)]

    # 2. Vectorized property fetching (Faster than iterating objects per concept)
    print("  Extracting labels and relations (Vectorized)...")

    # PrefLabels
    for s, o in g.subject_objects(SKOS.prefLabel):
        if s in uri_to_id and isinstance(o, Literal) and o.language == lang:
            # If multiple, logic says pick first found (or overwrite)
            prefs[uri_to_id[s]] = str(o).strip()

    # AltLabels
    for s, o in g.subject_objects(SKOS.altLabel):
        if s in uri_to_id and isinstance(o, Literal) and o.language == lang:
            txt = str(o).strip()
            if txt:
                alts[uri_to_id[s]].append(txt)

    # Broader
    for s, o in g.subject_objects(SKOS.broader):
        if s in uri_to_id and o in uri_to_id:
            src_id = uri_to_id[s]
            dst_id = uri_to_id[o]
            broader[src_id].append(dst_id)

    # Related
    for s, o in g.subject_objects(SKOS.related):
        if s in uri_to_id and o in uri_to_id:
            src_id = uri_to_id[s]
            dst_id = uri_to_id[o]
            related[src_id].append(dst_id)

    # 3. Derive Narrower (Invert Broader)
    print("  Deriving narrower relations...")
    narrower: List[List[int]] = [[] for _ in range(n_concepts)]
    for child_id in range(n_concepts):
        for parent_id in broader[child_id]:
            narrower[parent_id].append(child_id)

    # 4. Build Final Structures
    valid_ids = [i for i, p in enumerate(prefs) if p is not None]

    final_concepts: List[ConceptData] = []
    old_id_to_new_id = {}

    for new_idx, old_idx in enumerate(valid_ids):
        old_id_to_new_id[old_idx] = new_idx

    for old_idx in valid_ids:
        # Remap relations to new IDs
        b_new = [old_id_to_new_id[x] for x in broader[old_idx] if x in old_id_to_new_id]
        n_new = [old_id_to_new_id[x] for x in narrower[old_idx] if x in old_id_to_new_id]
        r_new = [old_id_to_new_id[x] for x in related[old_idx] if x in old_id_to_new_id]

        # Sort alts to ensure deterministic iteration order for Split logic
        sorted_alts = sorted(alts[old_idx])

        final_concepts.append(ConceptData(
            uri=str(id_to_uri[old_idx]),
            pref=prefs[old_idx], # type: ignore
            alts=sorted_alts,
            broader=b_new,
            narrower=n_new,
            related=r_new
        ))

    # 5. Build Undirected Adjacency (for shortest path)
    # Map: ID -> Set[ID]
    adj: Dict[int, Set[int]] = defaultdict(set)
    include_related_in_graph = False # Original logic was False. Set True if you want related links in path calc.

    for idx, c in enumerate(final_concepts):
        for p in c.broader:
            adj[idx].add(p)
            adj[p].add(idx)
        for ch in c.narrower:
            adj[idx].add(ch)
            adj[ch].add(idx)

        if include_related_in_graph:
            for r in c.related:
                adj[idx].add(r)
                adj[r].add(idx)

    # Free memory
    del g, prefs, alts, broader, narrower, related, uri_to_id, id_to_uri
    gc.collect()

    return final_concepts, adj

# =========================
# ANCESTORS / DESCENDANTS (Integer Optimized)
# =========================

def compute_transitive_closures(concepts: List[ConceptData]) -> Tuple[List[Set[int]], List[Set[int]]]:
    """
    Computes ancestors and descendants using memoized DFS on integer IDs.
    Returns list of sets indexed by ConceptID.
    """
    n = len(concepts)
    ancestors = [None] * n
    descendants = [None] * n

    # Ancestors
    def get_ancestors(u: int) -> Set[int]:
        if ancestors[u] is not None:
            return ancestors[u]

        out = set()
        for p in concepts[u].broader:
            out.add(p)
            out.update(get_ancestors(p))

        ancestors[u] = out
        return out

    # Descendants
    def get_descendants(u: int) -> Set[int]:
        if descendants[u] is not None:
            return descendants[u]

        out = set()
        for ch in concepts[u].narrower:
            out.add(ch)
            out.update(get_descendants(ch))

        descendants[u] = out
        return out

    # Fill all with progress bar
    for i in tqdm(range(n), desc="  Computing Hierarchy", unit="concept"):
        get_ancestors(i)
        get_descendants(i)

    return ancestors, descendants # type: ignore

def build_exclusion_sets(
    concepts: List[ConceptData],
    ancestors: List[Set[int]],
    descendants: List[Set[int]],
    exclude_related: bool
) -> List[Set[int]]:
    n = len(concepts)
    excl = []
    for i in range(n):
        s = {i} # self
        s.update(ancestors[i])
        s.update(descendants[i])
        if exclude_related:
            s.update(concepts[i].related)
        excl.append(s)
    return excl

# =========================
# TRAIN/TEST SPLIT
# =========================

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int) -> int:
        path = []
        root = x
        while self.parent[root] != root:
            path.append(root)
            root = self.parent[root]
        for node in path:
            self.parent[node] = root
        return root

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def split_concepts_no_overlap(
    concepts: List[ConceptData],
    ratio: float,
    seed: int
) -> Tuple[List[int], List[int]]:

    random.seed(seed)
    n = len(concepts)

    # Map term -> list of concept IDs
    # Using sorted concepts ensures iteration order is fixed.
    term_to_ids = defaultdict(list)
    for idx, c in enumerate(concepts):
        # Pref
        term_to_ids[c.pref].append(idx)
        # Alts
        # c.alts is already sorted in load_concepts_optimized
        for alt in c.alts:
            term_to_ids[alt].append(idx)

    uf = UnionFind(n)
    # The order of iteration here depends on insertion order into term_to_ids
    # Because we iterated sorted concepts and sorted alts, this is deterministic.
    for ids in term_to_ids.values():
        if len(ids) > 1:
            head = ids[0]
            for other in ids[1:]:
                uf.union(head, other)

    # Group by root
    groups = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        groups[root].append(idx)

    group_list = list(groups.values())
    # This shuffle is now fully deterministic
    random.shuffle(group_list)

    train_ids = []
    test_ids = []
    target_train = int(n * ratio)

    current_train = 0
    for grp in group_list:
        if current_train < target_train:
            train_ids.extend(grp)
            current_train += len(grp)
        else:
            test_ids.extend(grp)

    return train_ids, test_ids

# =========================
# CSV WRITER HELPER
# =========================

class IncrementalCSVWriter:
    """Handles writing rows to CSV in chunks to avoid OOM with large lists."""
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.buffer = []
        self.total_written = 0

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Initialize file and write header
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def add(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= WRITE_BATCH_SIZE:
            self.flush()

    def add_batch(self, rows):
        self.buffer.extend(rows)
        if len(self.buffer) >= WRITE_BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.total_written += len(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
        print(f"  -> Saved {self.total_written} rows to {self.filepath}")

# =========================
# PAIR GENERATION (The Engine)
# =========================

def generate_positive_pairs_to_csv(
    concept_ids: List[int],
    concepts: List[ConceptData],
    output_path: str
):
    writer = IncrementalCSVWriter(output_path, ["term1", "term2", "concept_uri", "label", "shortest_path"])

    for cid in tqdm(concept_ids, desc="  Positive Pairs", unit="concept"):
        c = concepts[cid]
        terms = [c.pref] + c.alts
        # clique
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                writer.add({
                    "term1": terms[i],
                    "term2": terms[j],
                    "concept_uri": c.uri,
                    "label": 1,
                    "shortest_path": 0
                })
    writer.close()

def generate_negative_pairs_to_csv(
    concept_ids: List[int],
    concepts: List[ConceptData],
    exclusion_sets: List[Set[int]],
    adj: Dict[int, Set[int]],
    seed: int,
    output_path: str
):
    """
    Optimized Negative Generation using Single-Source Shortest Path (SSSP).
    Instead of running BFS for every pair (N^2 BFS), we run BFS once per concept (N BFS).
    """
    writer = IncrementalCSVWriter(output_path, ["term1", "term2", "concept1_uri", "concept2_uri", "label", "shortest_path"])

    sorted_ids = sorted(concept_ids) # Sort for determinism
    n = len(sorted_ids)

    random.seed(seed)

    # Progress bar setup
    mode_str = 'ALL' if GENERATE_ALL_NEGATIVES else 'SAMPLED'
    pbar = tqdm(total=n, desc=f"  Negative Pairs ({mode_str})", unit="concept")

    for i in range(n):
        u1_id = sorted_ids[i]
        c1 = concepts[u1_id]
        excl = exclusion_sets[u1_id]

        # 1. Run ONE BFS from u1 to find distances to ALL connected nodes
        dists = {}
        if u1_id in adj:
            q = deque([(u1_id, 0)])
            visited = {u1_id}
            dists[u1_id] = 0

            while q:
                curr, d = q.popleft()
                for nb in adj[curr]:
                    if nb not in visited:
                        visited.add(nb)
                        dists[nb] = d + 1
                        q.append((nb, d + 1))

        # 2. Iterate potential partners
        if GENERATE_ALL_NEGATIVES:
            # Check all subsequent concepts
            for j in range(i + 1, n):
                u2_id = sorted_ids[j]

                # Exclusion Check
                if u2_id in excl:
                    continue
                if u1_id in exclusion_sets[u2_id]:
                    continue

                d = dists.get(u2_id, -1)

                c2 = concepts[u2_id]
                writer.add({
                    "term1": c1.pref,
                    "term2": c2.pref,
                    "concept1_uri": c1.uri,
                    "concept2_uri": c2.uri,
                    "label": 0,
                    "shortest_path": d
                })
        else:
            # Sampled approach
            valid_samples = []
            attempts = 0
            max_attempts = NEGATIVE_SAMPLES_PER_CONCEPT * 3

            while len(valid_samples) < NEGATIVE_SAMPLES_PER_CONCEPT and attempts < max_attempts:
                attempts += 1
                u2_id = random.choice(sorted_ids)

                if u2_id == u1_id: continue
                if u1_id >= u2_id: continue

                if u2_id in excl: continue
                if u1_id in exclusion_sets[u2_id]: continue

                if u2_id not in valid_samples:
                    valid_samples.append(u2_id)

            for u2_id in valid_samples:
                c2 = concepts[u2_id]
                d = dists.get(u2_id, -1)
                writer.add({
                    "term1": c1.pref,
                    "term2": c2.pref,
                    "concept1_uri": c1.uri,
                    "concept2_uri": c2.uri,
                    "label": 0,
                    "shortest_path": d
                })

        pbar.update(1)

    pbar.close()
    writer.close()

# =========================
# MAIN
# =========================

def main():
    print("=" * 70)
    print("ELSST PAIR GENERATION (OPTIMIZED & DETERMINISTIC)")
    print(f"RDF: {RDF_FILE_PATH}")
    print(f"Mode: {'ALL NEGATIVES' if GENERATE_ALL_NEGATIVES else 'SAMPLED'}")
    print("=" * 70)

    # 1. Load
    concepts, adj = load_concepts_optimized(RDF_FILE_PATH, LANGUAGE)
    print(f"Loaded {len(concepts)} concepts.")

    # 2. Hierarchy
    print("[2-4/8] Computing hierarchy (Ancestors/Descendants/Exclusions)...")
    ancestors, descendants = compute_transitive_closures(concepts)
    exclusion_sets = build_exclusion_sets(concepts, ancestors, descendants, EXCLUDE_SKOS_RELATED_FROM_NEGATIVES)

    # 3. Split
    print("[6/8] Splitting Train/Test (No term overlap)...")
    # We pass the list of all indices [0..N-1] implicitly via the function
    train_ids, test_ids = split_concepts_no_overlap(concepts, TRAIN_RATIO, RANDOM_SEED)
    print(f"Train: {len(train_ids)} | Test: {len(test_ids)}")

    # 4. Generate
    print("[8/8] Generating Pairs (Direct to CSV)...")

    # Train
    generate_positive_pairs_to_csv(train_ids, concepts, os.path.join(OUTPUT_DIR, "train_positive_pairs.csv"))
    generate_negative_pairs_to_csv(train_ids, concepts, exclusion_sets, adj, RANDOM_SEED, os.path.join(OUTPUT_DIR, "train_negative_pairs.csv"))

    # Test
    generate_positive_pairs_to_csv(test_ids, concepts, os.path.join(OUTPUT_DIR, "test_positive_pairs.csv"))
    generate_negative_pairs_to_csv(test_ids, concepts, exclusion_sets, adj, RANDOM_SEED + 1, os.path.join(OUTPUT_DIR, "test_negative_pairs.csv"))

    print("\nDone.")

if __name__ == "__main__":
    main()