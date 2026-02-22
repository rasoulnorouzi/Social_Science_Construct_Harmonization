"""
Build ELSST root-to-leaf embedding cache for the web-concept-clustering app.

Parses ELSST_R5.rdf, extracts the SKOS hierarchy, finds all leaf concepts,
builds full root→leaf path strings, embeds each path with MULTIPLE models,
and saves per-model cache files.

Outputs (per model):
  - elsst_paths.json                     (shared metadata, original-case for display)
  - elsst_embeddings_<model_key>.bin     (raw Float32, n×dim)
  - elsst_embeddings_<model_key>_l2.bin  (L2-normalized Float32, n×dim)

Embeddings are computed on LOWERCASED path strings for case-insensitive
matching, since users typically input lowercase queries.

Usage:
    cd web-concept-clustering/
    python build_elsst_cache.py
"""

import json
import struct
import os
import sys
from collections import defaultdict

import numpy as np
from rdflib import Graph, Namespace, URIRef
from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
RDF_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'raw_datasets', 'ELSST_R5.rdf')
OUT_DIR  = os.path.dirname(__file__) or '.'
LANG = 'en'  # use English prefLabels only

# Models to build caches for.
# key: short identifier (used in filenames and JS model selector)
# python_model: HuggingFace model name for embedding
# browser_model: Xenova model name used in the browser
MODELS = {
    'allmpnet': {
        'python_model':  'all-mpnet-base-v2',
        'browser_model': 'Xenova/all-mpnet-base-v2',
        'display_name':  'All-MPNet-Base-v2',
    },
}

SKOS = Namespace('http://www.w3.org/2004/02/skos/core#')


def parse_elsst(rdf_path: str):
    """Parse ELSST RDF and return (labels, children, parents, top_concepts)."""
    print(f'Parsing {rdf_path} …')
    g = Graph()
    g.parse(rdf_path, format='xml')

    # ── Collect concept URIs (only skos:Concept typed) ────────────────────
    RDF_TYPE = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
    concept_uris = set()
    for s, p, o in g.triples((None, RDF_TYPE, SKOS.Concept)):
        concept_uris.add(str(s))

    # ── Collect English prefLabels (only for actual concepts) ──────────────
    labels = {}  # URI → English label
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        if hasattr(o, 'language') and o.language == LANG and str(s) in concept_uris:
            labels[str(s)] = str(o)

    # ── Collect broader / narrower ─────────────────────────────────────────
    children = defaultdict(set)   # parent URI → {child URIs}
    parents  = defaultdict(set)   # child URI → {parent URIs}

    for s, p, o in g.triples((None, SKOS.narrower, None)):
        children[str(s)].add(str(o))
        parents[str(o)].add(str(s))

    for s, p, o in g.triples((None, SKOS.broader, None)):
        children[str(o)].add(str(s))
        parents[str(s)].add(str(o))

    # ── Top concepts (roots of the hierarchy) ──────────────────────────────
    top_concepts = set()
    for s, p, o in g.triples((None, SKOS.topConceptOf, None)):
        top_concepts.add(str(s))
    for s, p, o in g.triples((None, SKOS.hasTopConcept, None)):
        top_concepts.add(str(o))

    print(f'  {len(labels)} English concepts, '
          f'{sum(len(v) for v in children.values())} parent→child links, '
          f'{len(top_concepts)} top concepts')
    return labels, dict(children), dict(parents), top_concepts


def build_root_to_leaf_paths(labels, children, parents, top_concepts):
    """
    Find all leaf concepts and build every root→leaf path.
    A leaf is a concept with no children in the narrower relation.

    Returns list of dicts:
        { "leaf": "DOMESTIC SAFETY",           ← original case (for display)
          "path": "BUILDINGS > RESIDENTIAL BUILDINGS > DOMESTIC SAFETY",
          "path_lower": "buildings > residential buildings > domestic safety" }
    """
    all_uris = set(labels.keys())
    parent_uris = set(children.keys())
    leaf_uris = all_uris - parent_uris

    print(f'  {len(leaf_uris)} leaf concepts (no narrower children)')

    def get_paths_to_root(uri, visited=None):
        """Return list of paths (each path = list of URIs from root→node)."""
        if visited is None:
            visited = set()
        if uri in visited:
            return []  # cycle guard
        visited.add(uri)

        parent_set = parents.get(uri, set())
        if not parent_set or uri in top_concepts:
            return [[uri]]

        paths = []
        for parent_uri in parent_set:
            for ancestor_path in get_paths_to_root(parent_uri, visited.copy()):
                paths.append(ancestor_path + [uri])
        return paths

    results = []
    for leaf_uri in sorted(leaf_uris):
        leaf_label = labels.get(leaf_uri)
        if not leaf_label:
            continue

        paths = get_paths_to_root(leaf_uri)
        if not paths:
            paths = [[leaf_uri]]

        for path_uris in paths:
            path_labels = []
            for uri in path_uris:
                label = labels.get(uri, uri.split('/')[-1])
                path_labels.append(label)
            path_str = ' > '.join(path_labels)
            results.append({
                'leaf': leaf_label,
                'path': path_str,
                'path_lower': path_str.lower(),
            })

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = r['path']
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f'  {len(deduped)} unique root→leaf paths')
    return deduped


def compute_embeddings(paths_data, model_cfg):
    """Compute RAW embeddings (no L2 norm) for lowercased path strings."""
    model_name = model_cfg['python_model']
    path_strings = [d['path_lower'] for d in paths_data]

    print(f'Loading model {model_name} ...')
    model = SentenceTransformer(model_name)
    print(f'Embedding {len(path_strings)} lowercased paths ...')
    embeddings = model.encode(
        path_strings,
        show_progress_bar=True,
        normalize_embeddings=False,
    )

    print(f'  Embedding shape: {embeddings.shape}')
    return embeddings.astype(np.float32)


def l2_normalize(embeddings):
    """Return L2-normalized copy of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (embeddings / norms).astype(np.float32)


def save_binary(embeddings, path):
    """Save Float32 embeddings with uint32 header (n, dim)."""
    n, dim = embeddings.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', n, dim))
        f.write(embeddings.tobytes())
    size = os.path.getsize(path)
    print(f'  Saved {path} ({size / 1024:.1f} KB, {n}×{dim} float32)')


def save_cache(paths_data, out_dir):
    """Save shared paths JSON (original case for display, no path_lower)."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON metadata — keep original case for display, drop path_lower
    display_data = [{'leaf': d['leaf'], 'path': d['path']} for d in paths_data]
    json_path = os.path.join(out_dir, 'elsst_paths.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(display_data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
    size = os.path.getsize(json_path)
    print(f'  Saved {json_path} ({size / 1024:.1f} KB, {len(display_data)} entries)')


def main():
    rdf_path = os.path.abspath(RDF_PATH)
    if not os.path.isfile(rdf_path):
        print(f'ERROR: RDF file not found: {rdf_path}')
        sys.exit(1)

    out_dir = os.path.abspath(OUT_DIR)

    labels, children, parents, top_concepts = parse_elsst(rdf_path)
    paths_data = build_root_to_leaf_paths(labels, children, parents, top_concepts)

    if not paths_data:
        print('ERROR: No root→leaf paths found. Check the RDF file.')
        sys.exit(1)

    # Save shared metadata once
    save_cache(paths_data, out_dir)

    # Build embeddings for each model
    for model_key, model_cfg in MODELS.items():
        print(f'\n{"="*60}')
        print(f'Model: {model_cfg["display_name"]} ({model_key})')
        print(f'{"="*60}')

        raw_embs = compute_embeddings(paths_data, model_cfg)
        l2_embs  = l2_normalize(raw_embs)

        raw_path = os.path.join(out_dir, f'elsst_embeddings_{model_key}.bin')
        l2_path  = os.path.join(out_dir, f'elsst_embeddings_{model_key}_l2.bin')

        save_binary(raw_embs, raw_path)
        save_binary(l2_embs,  l2_path)

    print(f'\nDone! All cache files saved to {out_dir}/')


if __name__ == '__main__':
    main()
