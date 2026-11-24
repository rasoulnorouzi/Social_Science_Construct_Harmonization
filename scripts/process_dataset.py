"""
Unified Dataset Processor: ELSST (RDF/SKOS) & APA (Zthes XML) -> Benchmark Pairs
=================================================================================
Converts raw taxonomy files into term pairs for benchmarking.

Supports two formats:
  - "elsst" : SKOS/RDF  (prefLabel, altLabel, broader, narrower, related)
  - "apa"   : Zthes XML (PT terms with UF synonyms, BT/NT/RT relations)

Pair construction:
  - Positive pairs: every combination of two labels of the same concept.
  - Negative pairs: every combination of a label of one concept with a label of
    another concept, over all labels (preferred label and entry terms). A concept
    is never paired with itself, with any of its ancestors or with any of its
    descendants (the whole family line up to the root).
  - Broader/narrower pairs (a concept with each of its ancestors) are written to a
    separate "hierarchy" file and are not part of the benchmark.
  - ELSST is split 70/30 by concept for calibration and evaluation. APA is used
    only for evaluation and is built once over the whole thesaurus (no split).
  - Pairs are stored as integer term ids in Parquet files; the strings are in a
    separate terms file (APA has about 60 million negative pairs).

Output files per split (prefix "train_" / "test_" for ELSST, "" for APA):
  {prefix}terms.parquet            term_id, term
  {prefix}concepts.parquet         concept_id, uri, pref_label
  {prefix}positive_pairs.parquet   idx1, idx2, concept1, concept2, label, shortest_path, pair_type
  {prefix}negative_pairs.parquet   idx1, idx2, concept1, concept2, label, shortest_path, pair_type
  {prefix}hierarchy_pairs.parquet  idx1, idx2, concept1, concept2, depth, pair_type
idx1 / idx2 are row numbers in the terms file. pair_type: 0 = preferred-preferred,
1 = preferred-entry term, 2 = entry term-entry term.

Usage:
  python scripts/process_dataset.py               # build ELSST (70/30 split) and APA (full)
  python scripts/process_dataset.py --count-only  # print the counts, write nothing
"""

from __future__ import annotations

import os
import sys
import random
import gc
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
EXCLUDE_SKOS_RELATED_FROM_NEGATIVES = False

# Rows kept in memory before they are written to the Parquet file
WRITE_BATCH_SIZE = 5_000_000

# Codes stored in the pair_type column
PAIR_TYPES = ["pref_pref", "pref_alt", "alt_alt"]

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



def make_exclusions_symmetric(exclusion_sets: List[Set[int]]) -> List[Set[int]]:
    """
    A pair is excluded when u2 is excluded for u1 OR u1 is excluded for u2.
    Making every set symmetric allows a single membership test.
    """
    for i, excl in enumerate(exclusion_sets):
        for j in list(excl):
            exclusion_sets[j].add(i)
    return exclusion_sets


# =========================
# PARQUET WRITER
# =========================

class PairWriter:
    """
    Collects pair columns in memory and writes them to a Parquet
    file in batches. With path=None nothing is written and only the rows are counted
    (used by --count-only).
    """

    def __init__(self, path: Optional[str], columns: Dict[str, str]):
        self.path = path
        self.columns = columns  # column name -> numpy dtype
        self.buffers = {name: [] for name in columns}
        self.n_buffered = 0
        self.n_written = 0
        self.type_counts = np.zeros(len(PAIR_TYPES), dtype=np.int64)
        self.writer = None
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            schema = pa.schema([(name, pa.from_numpy_dtype(np.dtype(dt))) for name, dt in columns.items()])
            self.writer = pq.ParquetWriter(path, schema)

    def add(self, **arrays):
        n = len(arrays["idx2"])
        if n == 0:
            return
        self.type_counts += np.bincount(arrays["pair_type"], minlength=len(PAIR_TYPES))
        for name, dtype in self.columns.items():
            values = arrays[name]
            if np.ndim(values) == 0:  # a single value for all rows
                values = np.full(n, values)
            self.buffers[name].append(np.asarray(values, dtype=dtype))
        self.n_buffered += n
        if self.n_buffered >= WRITE_BATCH_SIZE:
            self.flush()

    def flush(self):
        if self.n_buffered == 0:
            return
        if self.writer is not None:
            table = pa.table({name: np.concatenate(parts) for name, parts in self.buffers.items()})
            self.writer.write_table(table)
        self.n_written += self.n_buffered
        self.buffers = {name: [] for name in self.columns}
        self.n_buffered = 0

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
            print(f"  -> Saved {self.n_written:,} rows to {self.path}")
        return self.n_written


PAIR_COLUMNS = {
    "idx1": "int32", "idx2": "int32",
    "concept1": "int32", "concept2": "int32",
    "label": "int8", "shortest_path": "int16", "pair_type": "int8",
}
HIERARCHY_COLUMNS = {
    "idx1": "int32", "idx2": "int32",
    "concept1": "int32", "concept2": "int32",
    "depth": "int16", "pair_type": "int8",
}


# =========================
# TERMS OF ONE SPLIT
# =========================

def normalise(term: str) -> str:
    """Lowercase and strip; used to skip pairs of identical strings."""
    return term.strip().lower()


def concept_labels(c: ConceptData) -> List[str]:
    """Preferred label first, then entry terms, exact duplicates removed."""
    labels = []
    for label in [c.pref] + c.alts:
        if label not in labels:
            labels.append(label)
    return labels


def build_term_table(split_ids: List[int], concepts: List[ConceptData]):
    """
    One row per unique label string in the split, in concept order
    (preferred label first). The row number is the term id used in all pair files.
    """
    terms: List[str] = []
    term_id: Dict[str, int] = {}
    for cid in split_ids:
        for label in concept_labels(concepts[cid]):
            if label not in term_id:
                term_id[label] = len(terms)
                terms.append(label)
    return terms, term_id


def build_label_rows(split_ids: List[int], concepts: List[ConceptData], term_id: Dict[str, int]):
    """
    One row per (concept, label) in the split. Returned as numpy arrays
    so that all labels of a set of partner concepts can be selected in one step.
    """
    norm_id: Dict[str, int] = {}
    rows_concept, rows_term, rows_pref, rows_norm = [], [], [], []
    for cid in split_ids:
        for k, label in enumerate(concept_labels(concepts[cid])):
            norm = normalise(label)
            if norm not in norm_id:
                norm_id[norm] = len(norm_id)
            rows_concept.append(cid)
            rows_term.append(term_id[label])
            rows_pref.append(k == 0)
            rows_norm.append(norm_id[norm])
    return {
        "concept": np.array(rows_concept, dtype=np.int32),
        "term": np.array(rows_term, dtype=np.int32),
        "is_pref": np.array(rows_pref, dtype=bool),
        "norm": np.array(rows_norm, dtype=np.int32),
    }


def build_positive_partners(split_ids, concepts, label_rows):
    """
    For every normalised label, the normalised labels it forms a positive
    pair with in this split. A negative row that joins two such strings is skipped (this
    only happens when two concepts share a label).
    """
    partners: Dict[int, Set[int]] = defaultdict(set)
    for cid in split_ids:
        norms = label_rows["norm"][label_rows["concept"] == cid].tolist()
        for a in norms:
            for b in norms:
                if a != b:
                    partners[a].add(b)
    return partners


def shared_label_owners(label_rows):
    """
    Labels that belong to more than one concept in the split.
    Returns (owners, is_shared): owners[term] = set of concept ids for every term,
    is_shared[term] = True when the term has more than one owner.
    """
    owners = defaultdict(set)
    for term, cid in zip(label_rows["term"].tolist(), label_rows["concept"].tolist()):
        owners[term].add(cid)
    n_terms = int(label_rows["term"].max()) + 1
    is_shared = np.zeros(n_terms, dtype=bool)
    for term, own in owners.items():
        if len(own) > 1:
            is_shared[term] = True
    return owners, is_shared


def keep_shared_row(term1, term2, owners, exclusion_sets, seen):
    """
    Rule for a negative row that involves a shared label: drop it when any owner of term1
    is on the family line of any owner of term2, and write the same term pair only once.
    """
    for o1 in owners[term1]:
        for o2 in owners[term2]:
            if o1 == o2 or o2 in exclusion_sets[o1]:
                return False
    return first_time(term1, term2, seen)


def first_time(term1, term2, seen):
    """True the first time an unordered term pair is seen (and remembers it)."""
    key = (min(term1, term2), max(term1, term2))
    if key in seen:
        return False
    seen.add(key)
    return True


def pair_types(is_pref_1, is_pref_2):
    """0 = both preferred labels, 1 = one preferred label, 2 = no preferred label."""
    both = is_pref_1 & is_pref_2
    one = is_pref_1 | is_pref_2
    return np.where(both, 0, np.where(one, 1, 2)).astype(np.int8)


def bfs_distances(source: int, adj: Dict[int, Set[int]], n_concepts: int) -> np.ndarray:
    """Concept-level shortest path over broader/narrower edges (-1 = not connected)."""
    dist = np.full(n_concepts, -1, dtype=np.int16)
    dist[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nb in adj.get(node, ()):
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


# =========================
# PAIR GENERATION
# =========================

def generate_positive_pairs(split_ids, concepts, term_id, output_path):
    """
    Positive pairs: every combination of labels within one concept.
    """
    writer = PairWriter(output_path, PAIR_COLUMNS)
    for cid in tqdm(split_ids, desc="  Positive Pairs", unit="concept"):
        c = concepts[cid]
        terms = [c.pref] + c.alts
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                writer.add(
                    idx1=[term_id[terms[i]]], idx2=[term_id[terms[j]]],
                    concept1=cid, concept2=cid, label=1, shortest_path=0,
                    pair_type=np.array([1 if i == 0 else 2], dtype=np.int8),
                )
    n = writer.close()
    return n, writer.type_counts


def generate_negative_pairs(split_ids, concepts, exclusion_sets, adj, label_rows,
                            positive_partners, output_path):
    """
    Negative pairs over all labels.
    For every allowed concept pair (u1 < u2, not on the same family line), write one row
    for every combination of a label of u1 with a label of u2. Two kinds of rows are skipped:
      - the two strings are equal after lowercasing;
      - the two strings form a positive pair elsewhere in the split (shared labels).
    shortest_path is the concept-level graph distance, the same for all rows of a concept pair.

    A label can belong to two concepts (e.g. the preferred label of one concept is an entry
    term of another). A row with such a shared label is kept only if no owner of one label is
    on the family line of an owner of the other label, and it is written only once.
    """
    writer = PairWriter(output_path, PAIR_COLUMNS)
    n_concepts = len(concepts)
    split_arr = np.array(split_ids, dtype=np.int32)
    owners, is_shared = shared_label_owners(label_rows)
    seen_shared = set()

    for u1 in tqdm(split_ids, desc="  Negative Pairs", unit="concept"):
        # Partner concepts: later in the split order and not on the family line of u1
        allowed = np.zeros(n_concepts, dtype=bool)
        allowed[split_arr[split_arr > u1]] = True
        allowed[list(exclusion_sets[u1])] = False
        partner_rows = allowed[label_rows["concept"]]
        if not partner_rows.any():
            continue

        p_term = label_rows["term"][partner_rows]
        p_concept = label_rows["concept"][partner_rows]
        p_pref = label_rows["is_pref"][partner_rows]
        p_norm = label_rows["norm"][partner_rows]
        p_dist = bfs_distances(u1, adj, n_concepts)[p_concept]

        own_rows = np.where(label_rows["concept"] == u1)[0]
        for r in own_rows:
            norm = label_rows["norm"][r]
            blocked = [norm] + list(positive_partners.get(norm, ()))
            keep = ~np.isin(p_norm, blocked)
            own_term = label_rows["term"][r]
            if is_shared[own_term]:
                check = np.flatnonzero(keep)
            else:
                check = np.flatnonzero(keep & is_shared[p_term])
            for j in check:
                keep[j] = keep_shared_row(own_term, p_term[j], owners, exclusion_sets, seen_shared)
            writer.add(
                idx1=own_term, idx2=p_term[keep],
                concept1=u1, concept2=p_concept[keep],
                label=0, shortest_path=p_dist[keep],
                pair_type=pair_types(label_rows["is_pref"][r], p_pref[keep]),
            )
    n = writer.close()
    return n, writer.type_counts


def generate_hierarchy_pairs(split_ids, concepts, ancestors, label_rows, positive_partners, output_path):
    """
    Broader/narrower pairs, kept apart from the benchmark.
    For each concept and each of its ancestors in the same split: all label combinations.
    depth = number of broader steps (1 = direct parent, 2 = grandparent, ...).
    concept1 is the narrower concept, concept2 the ancestor. Same skip rules as negatives.
    """
    writer = PairWriter(output_path, HIERARCHY_COLUMNS)
    in_split = set(split_ids)
    _, is_shared = shared_label_owners(label_rows)
    seen_shared = set()

    for cid in split_ids:
        targets = ancestors[cid] & in_split
        if not targets:
            continue
        # Depth of every ancestor: breadth-first search upwards over broader links
        depth_of = {}
        frontier = [cid]
        depth = 0
        while frontier:
            depth += 1
            next_frontier = []
            for node in frontier:
                for parent in concepts[node].broader:
                    if parent not in depth_of:
                        depth_of[parent] = depth
                        next_frontier.append(parent)
            frontier = next_frontier

        own_rows = np.where(label_rows["concept"] == cid)[0]
        for anc in sorted(targets):
            anc_rows = np.where(label_rows["concept"] == anc)[0]
            for r in own_rows:
                norm = label_rows["norm"][r]
                blocked = [norm] + list(positive_partners.get(norm, ()))
                keep = anc_rows[~np.isin(label_rows["norm"][anc_rows], blocked)]
                own_term = label_rows["term"][r]
                # A shared label can reach the same term pair twice; write it once
                keep = [k for k in keep
                        if not (is_shared[own_term] or is_shared[label_rows["term"][k]])
                        or first_time(own_term, label_rows["term"][k], seen_shared)]
                keep = np.array(keep, dtype=np.int64)
                writer.add(
                    idx1=label_rows["term"][r], idx2=label_rows["term"][keep],
                    concept1=cid, concept2=anc, depth=depth_of[anc],
                    pair_type=pair_types(label_rows["is_pref"][r], label_rows["is_pref"][keep]),
                )
    n = writer.close()
    return n, writer.type_counts


# =========================
# CHECKS
# =========================

def check_split_files(output_dir: str, prefix: str, concepts, exclusion_sets):
    """
    Reads the written files back and checks the construction rules.
    Raises an error if a rule is broken.
    """
    n_concepts = len(concepts)
    family = np.zeros((n_concepts, n_concepts), dtype=bool)
    for i, excl in enumerate(exclusion_sets):
        family[i, list(excl)] = True

    terms = pd.read_parquet(os.path.join(output_dir, f"{prefix}terms.parquet"))
    pos = pd.read_parquet(os.path.join(output_dir, f"{prefix}positive_pairs.parquet"))
    neg = pd.read_parquet(os.path.join(output_dir, f"{prefix}negative_pairs.parquet"),
                          columns=["idx1", "idx2", "concept1", "concept2"])
    hier = pd.read_parquet(os.path.join(output_dir, f"{prefix}hierarchy_pairs.parquet"))

    assert (pos["concept1"] == pos["concept2"]).all(), "a positive pair joins two concepts"
    assert not family[neg["concept1"].to_numpy(), neg["concept2"].to_numpy()].any(), \
        "a negative pair joins two concepts on the same family line"
    assert family[hier["concept1"].to_numpy(), hier["concept2"].to_numpy()].all(), \
        "a hierarchy pair is not an ancestor-descendant pair"

    def pair_keys(df):
        a = df["idx1"].to_numpy(dtype=np.int64)
        b = df["idx2"].to_numpy(dtype=np.int64)
        return np.minimum(a, b) * len(terms) + np.maximum(a, b)

    neg_keys = pair_keys(neg)
    assert len(np.unique(neg_keys)) == len(neg_keys), "a negative term pair appears twice"
    assert not np.isin(pair_keys(hier), neg_keys).any(), "a hierarchy pair is also a negative"
    del neg_keys

    pos_terms = set(pos["idx1"]) | set(pos["idx2"])
    neg_terms = set(neg["idx1"].unique()) | set(neg["idx2"].unique())
    assert pos_terms <= neg_terms, "a term of the positives never appears in the negatives"
    assert max(neg_terms) < len(terms), "term id out of range"
    print(f"  Checks passed for '{prefix or 'full'}' "
          f"({len(terms):,} terms, {len(pos):,} positives, {len(neg):,} negatives, {len(hier):,} hierarchy pairs)")


# =========================
# UNIFIED ENTRY POINT
# =========================

def process_dataset(
    dataset: str = "elsst",
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    train_ratio: Optional[float] = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SEED,
    exclude_related_from_negatives: bool = False,
    count_only: bool = False,
):
    """
    Process a taxonomy file into positive, negative and hierarchy pair files.

    Parameters
    ----------
    dataset : "elsst" or "apa".
    input_path, output_dir : default paths are used when None.
    train_ratio : fraction of concepts for training (ELSST: 0.70). None = one full set (APA).
    seed : random seed for the concept split.
    exclude_related_from_negatives : also exclude skos:related / RT links from negatives.
    count_only : print the counts and write nothing.
    """
    global EXCLUDE_SKOS_RELATED_FROM_NEGATIVES
    EXCLUDE_SKOS_RELATED_FROM_NEGATIVES = exclude_related_from_negatives

    dataset = dataset.lower()
    if dataset not in ("elsst", "apa"):
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'elsst' or 'apa'.")
    input_path = input_path or DEFAULT_INPUT_PATHS[dataset]
    output_dir = output_dir or DEFAULT_OUTPUT_DIRS[dataset]

    print("=" * 70)
    print(f"DATASET PAIR GENERATION — {dataset.upper()}" + ("  [COUNT ONLY]" if count_only else ""))
    print(f"Input : {input_path}")
    print(f"Output: {'(nothing written)' if count_only else output_dir}")
    if train_ratio is not None:
        print(f"Split : {train_ratio:.0%} train / {1 - train_ratio:.0%} test, seed={seed}")
    else:
        print("Split : none (full thesaurus)")
    print("=" * 70)

    # 1. Load concepts
    if dataset == "elsst":
        concepts, adj = _load_elsst(input_path)
    else:
        concepts, adj = _load_apa(input_path)
    print(f"Loaded {len(concepts)} concepts.")

    # 2. Hierarchy closures + exclusion sets (self, all ancestors, all descendants)
    print("Computing hierarchy (Ancestors/Descendants/Exclusions)...")
    ancestors, descendants = compute_transitive_closures(concepts)
    exclusion_sets = build_exclusion_sets(
        concepts, ancestors, descendants, EXCLUDE_SKOS_RELATED_FROM_NEGATIVES
    )
    exclusion_sets = make_exclusions_symmetric(exclusion_sets)

    # 3. Splits: (name, concept ids, file prefix)
    if train_ratio is not None:
        print("Splitting Train/Test (No term overlap)...")
        train_ids, test_ids = split_concepts_no_overlap(concepts, train_ratio, seed)
        print(f"Train: {len(train_ids)} | Test: {len(test_ids)}")
        splits = [("train", sorted(train_ids), "train_"), ("test", sorted(test_ids), "test_")]
    else:
        splits = [("full", list(range(len(concepts))), "")]

    # 4. Write the files of every split
    summary = []
    for name, split_ids, prefix in splits:
        print(f"\n[{dataset} {name}]")
        terms, term_id = build_term_table(split_ids, concepts)
        label_rows = build_label_rows(split_ids, concepts, term_id)
        positive_partners = build_positive_partners(split_ids, concepts, label_rows)

        def out(kind):
            return None if count_only else os.path.join(output_dir, f"{prefix}{kind}.parquet")

        if not count_only:
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame({"term_id": np.arange(len(terms), dtype=np.int32), "term": terms}).to_parquet(
                out("terms"), index=False)
            pd.DataFrame({
                "concept_id": np.array(split_ids, dtype=np.int32),
                "uri": [concepts[c].uri for c in split_ids],
                "pref_label": [concepts[c].pref for c in split_ids],
            }).to_parquet(out("concepts"), index=False)

        n_pos, _ = generate_positive_pairs(split_ids, concepts, term_id, out("positive_pairs"))
        n_neg, neg_types = generate_negative_pairs(
            split_ids, concepts, exclusion_sets, adj, label_rows, positive_partners, out("negative_pairs"))
        n_hier, _ = generate_hierarchy_pairs(
            split_ids, concepts, ancestors, label_rows, positive_partners, out("hierarchy_pairs"))

        row = {"split": f"{dataset}_{name}", "concepts": len(split_ids), "unique_terms": len(terms),
               "positives": n_pos, "negatives": n_neg,
               **{f"neg_{t}": int(c) for t, c in zip(PAIR_TYPES, neg_types)},
               "hierarchy_pairs": n_hier}
        summary.append(row)
        for k, v in row.items():
            if k != "split":
                print(f"  {k:<16} {v:>14,}")

        if not count_only:
            check_split_files(output_dir, prefix, concepts, exclusion_sets)
        gc.collect()

    print("\nDone.")
    return summary


if __name__ == "__main__":
    # Usage: python scripts/process_dataset.py [elsst|apa] [--count-only]
    # Without a dataset name both are processed.
    count_only = "--count-only" in sys.argv
    names = [a for a in sys.argv[1:] if a in ("elsst", "apa")] or ["elsst", "apa"]

    if "elsst" in names:
        # ELSST: 70/30 concept split, seed 42 (calibration on train, evaluation on test)
        process_dataset(dataset="elsst", train_ratio=0.70, seed=42, count_only=count_only)

    if "apa" in names:
        # APA: one full set, no split (evaluation only, never used for calibration)
        process_dataset(dataset="apa", train_ratio=None, seed=42, count_only=count_only)
