"""
Unified Dataset Processor: ELSST (RDF/SKOS) & APA (Zthes XML) -> Train/Test Pairs
==================================================================================
Converts raw taxonomy files into positive/negative pair CSVs for benchmarking.

Supports two formats:
  - "elsst" : SKOS/RDF  (prefLabel, altLabel, broader, narrower, related)
  - "apa"   : Zthes XML (PT terms with UF synonyms, BT/NT/RT relations)

Usage:
  # Default: ELSST with 70/30 split, seed=42
  process_dataset()

  # APA with 80/20 split
  process_dataset(dataset="apa", train_ratio=0.80)

  # Full dataset (no split)
  process_dataset(dataset="elsst", train_ratio=None)
"""

from __future__ import annotations

import os
import csv
import random
import gc
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from tqdm import tqdm

# =========================
# PROJECT ROOT & DEFAULTS
# =========================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70

DEFAULT_INPUT_PATHS = {
    "elsst": str(PROJECT_ROOT / "datasets/raw_datasets/ELSST_R5.rdf"),
    "apa": str(PROJECT_ROOT / "datasets/raw_datasets/APA.xml"),
}

DEFAULT_OUTPUT_DIRS = {
    "elsst": str(PROJECT_ROOT / "datasets/processed_datasets/elsst"),
    "apa": str(PROJECT_ROOT / "datasets/processed_datasets/apa"),
}

# Negatives config
GENERATE_ALL_NEGATIVES = True
NEGATIVE_SAMPLES_PER_CONCEPT = 50  # used only if GENERATE_ALL_NEGATIVES=False
EXCLUDE_SKOS_RELATED_FROM_NEGATIVES = False
WRITE_BATCH_SIZE = 50000

# =========================
# DATA STRUCTURES
# =========================

ConceptID = int


@dataclass
class ConceptData:
    uri: str          # Unique identifier (URI for ELSST, termId for APA)
    pref: str         # Preferred label
    alts: List[str]   # Synonym labels
    broader: List[ConceptID] = field(default_factory=list)
    narrower: List[ConceptID] = field(default_factory=list)
    related: List[ConceptID] = field(default_factory=list)


# =========================
# ELSST RDF/SKOS PARSER
# =========================

def _load_elsst(rdf_path: str, lang: str = "en") -> Tuple[List[ConceptData], Dict[int, Set[int]]]:
    """
    Parses an ELSST SKOS/RDF file and returns:
      1. A list of ConceptData (index = ConceptID)
      2. An adjacency list (undirected) for shortest-path BFS
    """
    from rdflib import Graph, Namespace
    from rdflib.term import Literal

    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

    g = Graph()
    print("[ELSST] Loading RDF into memory...")
    g.parse(rdf_path)

    print("  Indexing Concept URIs...")
    concept_uris_set = set(g.subjects(RDF_NS.type, SKOS.Concept))
    sorted_uris = sorted(list(concept_uris_set))
    uri_to_id = {u: i for i, u in enumerate(sorted_uris)}
    id_to_uri = {i: u for i, u in enumerate(sorted_uris)}
    n_concepts = len(sorted_uris)

    prefs: List[Optional[str]] = [None] * n_concepts
    alts: List[List[str]] = [[] for _ in range(n_concepts)]
    broader: List[List[int]] = [[] for _ in range(n_concepts)]
    related: List[List[int]] = [[] for _ in range(n_concepts)]

    print("  Extracting labels and relations (Vectorized)...")

    for s, o in g.subject_objects(SKOS.prefLabel):
        if s in uri_to_id and isinstance(o, Literal) and o.language == lang:
            prefs[uri_to_id[s]] = str(o).strip()

    for s, o in g.subject_objects(SKOS.altLabel):
        if s in uri_to_id and isinstance(o, Literal) and o.language == lang:
            txt = str(o).strip()
            if txt:
                alts[uri_to_id[s]].append(txt)

    for s, o in g.subject_objects(SKOS.broader):
        if s in uri_to_id and o in uri_to_id:
            broader[uri_to_id[s]].append(uri_to_id[o])

    for s, o in g.subject_objects(SKOS.related):
        if s in uri_to_id and o in uri_to_id:
            related[uri_to_id[s]].append(uri_to_id[o])

    print("  Deriving narrower relations...")
    narrower: List[List[int]] = [[] for _ in range(n_concepts)]
    for child_id in range(n_concepts):
        for parent_id in broader[child_id]:
            narrower[parent_id].append(child_id)

    # Filter to concepts that have a prefLabel
    valid_ids = [i for i, p in enumerate(prefs) if p is not None]
    old_id_to_new_id = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_ids)}

    final_concepts: List[ConceptData] = []
    for old_idx in valid_ids:
        b_new = [old_id_to_new_id[x] for x in broader[old_idx] if x in old_id_to_new_id]
        n_new = [old_id_to_new_id[x] for x in narrower[old_idx] if x in old_id_to_new_id]
        r_new = [old_id_to_new_id[x] for x in related[old_idx] if x in old_id_to_new_id]

        sorted_alts = sorted(alts[old_idx])
        final_concepts.append(ConceptData(
            uri=str(id_to_uri[old_idx]),
            pref=prefs[old_idx],  # type: ignore
            alts=sorted_alts,
            broader=b_new,
            narrower=n_new,
            related=r_new,
        ))

    # Build undirected adjacency (broader/narrower only)
    adj: Dict[int, Set[int]] = defaultdict(set)
    for idx, c in enumerate(final_concepts):
        for p in c.broader:
            adj[idx].add(p)
            adj[p].add(idx)
        for ch in c.narrower:
            adj[idx].add(ch)
            adj[ch].add(idx)

    del g, prefs, alts, broader, narrower, related, uri_to_id, id_to_uri
    gc.collect()

    return final_concepts, adj


# =========================
# APA ZTHES XML PARSER
# =========================

def _load_apa(xml_path: str) -> Tuple[List[ConceptData], Dict[int, Set[int]]]:
    """
    Parses an APA Zthes XML file and returns:
      1. A list of ConceptData (index = ConceptID)
      2. An adjacency list (undirected) for shortest-path BFS

    APA Zthes mapping:
      - PT (Preferred Term) entries become concepts.
      - ND (Non-Descriptor) entries are synonyms — they have USE pointing to a PT.
        The PT side captures them via UF relations, so NDs are skipped as standalone concepts.
      - UF relations on PT  -> altLabels (synonyms, e.g. Paw, Wing for Animal Limb)
      - BT relations on PT  -> broader
      - NT relations on PT  -> narrower
      - RT relations on PT  -> related
    """
    print("[APA] Parsing Zthes XML...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # First pass: collect only PT terms, build id -> termName map for all terms
    all_term_names: Dict[str, str] = {}  # termId -> termName (for all terms)
    pt_terms: Dict[str, dict] = {}  # termId -> {name, alts, broader_ids, narrower_ids, related_ids}

    for term_elem in tqdm(root.findall("term"), desc="  Pass 1: Indexing terms"):
        tid_el = term_elem.find("termId")
        tname_el = term_elem.find("termName")
        ttype_el = term_elem.find("termType")

        if tid_el is None or tname_el is None or ttype_el is None:
            continue

        tid = tid_el.text.strip()
        tname = tname_el.text.strip()
        ttype = ttype_el.text.strip()

        all_term_names[tid] = tname

        if ttype != "PT":
            continue

        # Collect relations
        alts = []
        broader_ids = []
        narrower_ids = []
        related_ids = []

        for rel_elem in term_elem.findall("relation"):
            rel_type_el = rel_elem.find("relationType")
            rel_tid_el = rel_elem.find("termId")
            rel_name_el = rel_elem.find("termName")

            if rel_type_el is None or rel_tid_el is None:
                continue

            rel_type = rel_type_el.text.strip()
            rel_tid = rel_tid_el.text.strip()
            rel_name = rel_name_el.text.strip() if rel_name_el is not None else None

            if rel_type == "UF":
                # UF = "Used For" — these are synonyms
                if rel_name:
                    alts.append(rel_name)
            elif rel_type == "BT":
                broader_ids.append(rel_tid)
            elif rel_type == "NT":
                narrower_ids.append(rel_tid)
            elif rel_type == "RT":
                related_ids.append(rel_tid)

        pt_terms[tid] = {
            "name": tname,
            "alts": alts,
            "broader_ids": broader_ids,
            "narrower_ids": narrower_ids,
            "related_ids": related_ids,
        }

    # Second pass: assign integer IDs to PT terms (sorted for determinism)
    sorted_pt_ids = sorted(pt_terms.keys())
    tid_to_int: Dict[str, int] = {tid: i for i, tid in enumerate(sorted_pt_ids)}

    print(f"  Building {len(sorted_pt_ids)} concepts from PT terms...")

    final_concepts: List[ConceptData] = []
    for tid in sorted_pt_ids:
        info = pt_terms[tid]

        b_new = [tid_to_int[x] for x in info["broader_ids"] if x in tid_to_int]
        n_new = [tid_to_int[x] for x in info["narrower_ids"] if x in tid_to_int]
        r_new = [tid_to_int[x] for x in info["related_ids"] if x in tid_to_int]

        sorted_alts = sorted(info["alts"])
        final_concepts.append(ConceptData(
            uri=tid,  # use termId as the URI equivalent
            pref=info["name"],
            alts=sorted_alts,
            broader=b_new,
            narrower=n_new,
            related=r_new,
        ))

    # Build undirected adjacency (broader/narrower only, same as ELSST)
    adj: Dict[int, Set[int]] = defaultdict(set)
    for idx, c in enumerate(final_concepts):
        for p in c.broader:
            adj[idx].add(p)
            adj[p].add(idx)
        for ch in c.narrower:
            adj[idx].add(ch)
            adj[ch].add(idx)

    del tree, root, all_term_names, pt_terms
    gc.collect()

    return final_concepts, adj


# =========================
# SHARED PIPELINE FUNCTIONS
# =========================

def compute_transitive_closures(
    concepts: List[ConceptData],
) -> Tuple[List[Set[int]], List[Set[int]]]:
    """Computes ancestors and descendants using memoized DFS on integer IDs."""
    n = len(concepts)
    ancestors: List[Optional[Set[int]]] = [None] * n
    descendants: List[Optional[Set[int]]] = [None] * n

    def get_ancestors(u: int) -> Set[int]:
        if ancestors[u] is not None:
            return ancestors[u]  # type: ignore
        out: Set[int] = set()
        for p in concepts[u].broader:
            out.add(p)
            out.update(get_ancestors(p))
        ancestors[u] = out
        return out

    def get_descendants(u: int) -> Set[int]:
        if descendants[u] is not None:
            return descendants[u]  # type: ignore
        out: Set[int] = set()
        for ch in concepts[u].narrower:
            out.add(ch)
            out.update(get_descendants(ch))
        descendants[u] = out
        return out

    for i in tqdm(range(n), desc="  Computing Hierarchy", unit="concept"):
        get_ancestors(i)
        get_descendants(i)

    return ancestors, descendants  # type: ignore


def build_exclusion_sets(
    concepts: List[ConceptData],
    ancestors: List[Set[int]],
    descendants: List[Set[int]],
    exclude_related: bool,
) -> List[Set[int]]:
    n = len(concepts)
    excl = []
    for i in range(n):
        s = {i}
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
    seed: int,
) -> Tuple[List[int], List[int]]:
    random.seed(seed)
    n = len(concepts)

    term_to_ids: Dict[str, List[int]] = defaultdict(list)
    for idx, c in enumerate(concepts):
        term_to_ids[c.pref].append(idx)
        for alt in c.alts:
            term_to_ids[alt].append(idx)

    uf = UnionFind(n)
    for ids in term_to_ids.values():
        if len(ids) > 1:
            head = ids[0]
            for other in ids[1:]:
                uf.union(head, other)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        root = uf.find(idx)
        groups[root].append(idx)

    group_list = list(groups.values())
    random.shuffle(group_list)

    train_ids: List[int] = []
    test_ids: List[int] = []
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
# CSV WRITER
# =========================

class IncrementalCSVWriter:
    """Writes rows to CSV in chunks to avoid OOM."""

    def __init__(self, filepath: str, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.buffer: List[dict] = []
        self.total_written = 0

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def add(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= WRITE_BATCH_SIZE:
            self.flush()

    def add_batch(self, rows: List[dict]):
        self.buffer.extend(rows)
        if len(self.buffer) >= WRITE_BATCH_SIZE:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.total_written += len(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
        print(f"  -> Saved {self.total_written} rows to {self.filepath}")


# =========================
# PAIR GENERATION
# =========================

def generate_positive_pairs_to_csv(
    concept_ids: List[int],
    concepts: List[ConceptData],
    output_path: str,
):
    writer = IncrementalCSVWriter(
        output_path, ["term1", "term2", "concept_uri", "label", "shortest_path"]
    )
    for cid in tqdm(concept_ids, desc="  Positive Pairs", unit="concept"):
        c = concepts[cid]
        terms = [c.pref] + c.alts
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                writer.add(
                    {
                        "term1": terms[i],
                        "term2": terms[j],
                        "concept_uri": c.uri,
                        "label": 1,
                        "shortest_path": 0,
                    }
                )
    writer.close()


def generate_negative_pairs_to_csv(
    concept_ids: List[int],
    concepts: List[ConceptData],
    exclusion_sets: List[Set[int]],
    adj: Dict[int, Set[int]],
    seed: int,
    output_path: str,
):
    """
    Negative generation using Single-Source Shortest Path (SSSP / BFS).
    Runs BFS once per concept instead of per pair.
    """
    writer = IncrementalCSVWriter(
        output_path,
        ["term1", "term2", "concept1_uri", "concept2_uri", "label", "shortest_path"],
    )

    sorted_ids = sorted(concept_ids)
    n = len(sorted_ids)
    random.seed(seed)

    mode_str = "ALL" if GENERATE_ALL_NEGATIVES else "SAMPLED"
    pbar = tqdm(total=n, desc=f"  Negative Pairs ({mode_str})", unit="concept")

    for i in range(n):
        u1_id = sorted_ids[i]
        c1 = concepts[u1_id]
        excl = exclusion_sets[u1_id]

        # BFS from u1
        dists: Dict[int, int] = {}
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

        if GENERATE_ALL_NEGATIVES:
            for j in range(i + 1, n):
                u2_id = sorted_ids[j]
                if u2_id in excl:
                    continue
                if u1_id in exclusion_sets[u2_id]:
                    continue
                d = dists.get(u2_id, -1)
                c2 = concepts[u2_id]
                writer.add(
                    {
                        "term1": c1.pref,
                        "term2": c2.pref,
                        "concept1_uri": c1.uri,
                        "concept2_uri": c2.uri,
                        "label": 0,
                        "shortest_path": d,
                    }
                )
        else:
            valid_samples: List[int] = []
            attempts = 0
            max_attempts = NEGATIVE_SAMPLES_PER_CONCEPT * 3
            while len(valid_samples) < NEGATIVE_SAMPLES_PER_CONCEPT and attempts < max_attempts:
                attempts += 1
                u2_id = random.choice(sorted_ids)
                if u2_id == u1_id:
                    continue
                if u1_id >= u2_id:
                    continue
                if u2_id in excl:
                    continue
                if u1_id in exclusion_sets[u2_id]:
                    continue
                if u2_id not in valid_samples:
                    valid_samples.append(u2_id)

            for u2_id in valid_samples:
                c2 = concepts[u2_id]
                d = dists.get(u2_id, -1)
                writer.add(
                    {
                        "term1": c1.pref,
                        "term2": c2.pref,
                        "concept1_uri": c1.uri,
                        "concept2_uri": c2.uri,
                        "label": 0,
                        "shortest_path": d,
                    }
                )

        pbar.update(1)

    pbar.close()
    writer.close()


# =========================
# UNIFIED ENTRY POINT
# =========================

def process_dataset(
    dataset: str = "elsst",
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    train_ratio: Optional[float] = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
    generate_all_negatives: bool = True,
    exclude_related_from_negatives: bool = False,
):
    """
    Unified function to process a taxonomy file into positive/negative pair CSVs.

    Parameters
    ----------
    dataset : str
        Which dataset to process. "elsst" (default) or "apa".
    input_path : str or None
        Path to the raw taxonomy file. If None, uses the default for the chosen dataset.
    output_dir : str or None
        Directory to save CSVs. If None, uses the default for the chosen dataset.
    train_ratio : float or None
        Fraction of concepts for training (default 0.70).
        If None, produces a single full dataset (no train/test split).
    seed : int
        Random seed for reproducibility (default 42).
    generate_all_negatives : bool
        If True, generate all valid negative pairs. If False, sample a fixed number.
    exclude_related_from_negatives : bool
        If True, also exclude skos:related / RT links from negative pairs.
    """
    # --- Apply module-level config from arguments ---
    global GENERATE_ALL_NEGATIVES, EXCLUDE_SKOS_RELATED_FROM_NEGATIVES
    GENERATE_ALL_NEGATIVES = generate_all_negatives
    EXCLUDE_SKOS_RELATED_FROM_NEGATIVES = exclude_related_from_negatives

    dataset = dataset.lower()
    if dataset not in ("elsst", "apa"):
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'elsst' or 'apa'.")

    if input_path is None:
        input_path = DEFAULT_INPUT_PATHS[dataset]
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIRS[dataset]

    split_mode = "SPLIT" if train_ratio is not None else "FULL"
    neg_mode = "ALL NEGATIVES" if GENERATE_ALL_NEGATIVES else "SAMPLED"

    print("=" * 70)
    print(f"DATASET PAIR GENERATION — {dataset.upper()}")
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print(f"Mode  : {split_mode} | {neg_mode} | seed={seed}")
    if train_ratio is not None:
        print(f"Split : {train_ratio:.0%} train / {1 - train_ratio:.0%} test")
    print("=" * 70)

    # 1. Load concepts using the appropriate parser
    if dataset == "elsst":
        concepts, adj = _load_elsst(input_path)
    else:
        concepts, adj = _load_apa(input_path)

    print(f"Loaded {len(concepts)} concepts.")

    # 2. Hierarchy closures + exclusion sets
    print("Computing hierarchy (Ancestors/Descendants/Exclusions)...")
    ancestors, descendants = compute_transitive_closures(concepts)
    exclusion_sets = build_exclusion_sets(
        concepts, ancestors, descendants, EXCLUDE_SKOS_RELATED_FROM_NEGATIVES
    )

    # 3. Generate pairs
    if train_ratio is not None:
        # --- Train/Test split mode ---
        print("Splitting Train/Test (No term overlap)...")
        train_ids, test_ids = split_concepts_no_overlap(concepts, train_ratio, seed)
        print(f"Train: {len(train_ids)} | Test: {len(test_ids)}")

        print("Generating Pairs (Direct to CSV)...")

        # Train
        generate_positive_pairs_to_csv(
            train_ids, concepts, os.path.join(output_dir, "train_positive_pairs.csv")
        )
        generate_negative_pairs_to_csv(
            train_ids, concepts, exclusion_sets, adj, seed,
            os.path.join(output_dir, "train_negative_pairs.csv"),
        )

        # Test
        generate_positive_pairs_to_csv(
            test_ids, concepts, os.path.join(output_dir, "test_positive_pairs.csv")
        )
        generate_negative_pairs_to_csv(
            test_ids, concepts, exclusion_sets, adj, seed + 1,
            os.path.join(output_dir, "test_negative_pairs.csv"),
        )
    else:
        # --- Full dataset mode (no split) ---
        all_ids = list(range(len(concepts)))
        print("Generating Pairs for FULL dataset (Direct to CSV)...")

        generate_positive_pairs_to_csv(
            all_ids, concepts, os.path.join(output_dir, "positive_pairs.csv")
        )
        generate_negative_pairs_to_csv(
            all_ids, concepts, exclusion_sets, adj, seed,
            os.path.join(output_dir, "negative_pairs.csv"),
        )

    print("\nDone.")


if __name__ == "__main__":
    # process ELSST with default settings
    
    process_dataset(
        dataset="elsst", input_path=str(PROJECT_ROOT / 'datasets/raw_datasets/ELSST_R5.rdf'), 
        output_dir=str(PROJECT_ROOT / 'datasets/processed_datasets/elsst'),
        train_ratio=0.70, seed=42, generate_all_negatives=True, 
        exclude_related_from_negatives=False
    )
    
    # process APA with the same settings
    process_dataset(
        dataset="apa", input_path=str(PROJECT_ROOT / 'datasets/raw_datasets/APA.xml'), 
        output_dir=str(PROJECT_ROOT / 'datasets/processed_datasets/apa'),
        train_ratio=0.70, seed=42, generate_all_negatives=True, 
        exclude_related_from_negatives=False
    )

