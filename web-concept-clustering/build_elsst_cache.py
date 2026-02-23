"""
Build ELSST root-to-leaf embedding cache for the web-concept-clustering app.

Parses ELSST_R5.rdf, extracts the SKOS hierarchy, alternative labels (altLabels),
and scope notes (scopeNotes). Finds all leaf concepts, builds full root->leaf path
strings, and embeds each path under MULTIPLE strategies.

Strategies:
  leaf      – embed leaf name only
  path      – embed full root->leaf path
  anchor    – "leaf: root > ... > leaf"
  context   – "leaf is related to parent, grandparent, ..."
  bracket   – "leaf (parent, grandparent, ...)"
  enriched  – leaf + altLabels + scope note + path (best for free-text queries)

Outputs:
  - elsst_paths.json                              (shared metadata incl. altLabels/scopeNotes)
  - elsst_embeddings_<model>_<strategy>_l2.bin   (L2-normalized Float32)

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
MODELS = {
    'allmpnet': {
        'python_model':  'all-mpnet-base-v2',
        'browser_model': 'Xenova/all-mpnet-base-v2',
        'display_name':  'All-MPNet-Base-v2',
    },
    'bge_base': {
        'python_model':  'BAAI/bge-base-en-v1.5',
        'browser_model': 'Xenova/bge-base-en-v1.5',
        'display_name':  'BGE-Base-EN v1.5',
    },
}

# Path representation strategies.
# Each strategy produces a different text string from the same path.
STRATEGIES = {
    'leaf': {
        'display': 'Leaf Only',
        'description': 'Embed only the leaf concept name — strongest direct semantic match.',
    },
    'anchor': {
        'display': 'Leaf-Anchored',
        'description': 'Leaf name prepended to the full path ("C: A > B > C") — emphasises the leaf while keeping context.',
    },
    'context': {
        'display': 'Contextual',
        'description': 'Natural-language framing ("C is related to B, A") — best for free-text queries.',
    },
    'bracket': {
        'display': 'Bracketed',
        'description': 'Leaf with parents in brackets ("C (B, A)") — good for disambiguation.',
    },
    'enriched': {
        'display': 'Enriched',
        'description': 'Leaf + alternative labels + scope note + path — maximum semantic coverage for free-text queries.',
    },
}

SKOS = Namespace('http://www.w3.org/2004/02/skos/core#')


def parse_elsst(rdf_path: str):
    """Parse ELSST RDF and return (labels, alt_labels, scope_notes, children, parents, top_concepts)."""
    print(f'Parsing {rdf_path} …')
    g = Graph()
    g.parse(rdf_path, format='xml')

    # ── Collect concept URIs (only skos:Concept typed) ────────────────────
    RDF_TYPE = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
    concept_uris = set()
    for s, p, o in g.triples((None, RDF_TYPE, SKOS.Concept)):
        concept_uris.add(str(s))

    # ── Collect English prefLabels (only for actual concepts) ──────────────
    labels = {}  # URI -> English prefLabel
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        if hasattr(o, 'language') and o.language == LANG and str(s) in concept_uris:
            labels[str(s)] = str(o)

    # ── Collect English altLabels (synonyms / entry terms) ────────────────
    alt_labels = defaultdict(list)  # URI -> [altLabel, ...]
    for s, p, o in g.triples((None, SKOS.altLabel, None)):
        if hasattr(o, 'language') and o.language == LANG and str(s) in concept_uris:
            alt_labels[str(s)].append(str(o))

    # ── Collect English scopeNotes (definitions / usage notes) ───────────
    scope_notes = {}  # URI -> scopeNote string
    for s, p, o in g.triples((None, SKOS.scopeNote, None)):
        if hasattr(o, 'language') and o.language == LANG and str(s) in concept_uris:
            scope_notes[str(s)] = str(o)
    # Also try skos:definition
    SKOS_DEF = SKOS.definition
    for s, p, o in g.triples((None, SKOS_DEF, None)):
        uri = str(s)
        if uri in concept_uris and uri not in scope_notes:
            if not hasattr(o, 'language') or o.language == LANG:
                scope_notes[uri] = str(o)

    n_alt = sum(len(v) for v in alt_labels.values())
    print(f'  {len(labels)} prefLabels, {n_alt} altLabels, {len(scope_notes)} scopeNotes')

    # ── Collect broader / narrower ─────────────────────────────────────────
    children = defaultdict(set)   # parent URI -> {child URIs}
    parents  = defaultdict(set)   # child URI -> {parent URIs}

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
          f'{sum(len(v) for v in children.values())} parent->child links, '
          f'{len(top_concepts)} top concepts')
    return labels, dict(alt_labels), dict(scope_notes), dict(children), dict(parents), top_concepts


def build_root_to_leaf_paths(labels, alt_labels, scope_notes, children, parents, top_concepts):
    """
    Find all leaf concepts and build every root->leaf path.
    A leaf is a concept with no children in the narrower relation.

    Returns list of dicts:
        { "leaf": "DOMESTIC SAFETY",
          "path": "BUILDINGS > RESIDENTIAL BUILDINGS > DOMESTIC SAFETY",
          "path_lower": "...",
          "alt_labels": ["Home safety", ...],   ← English altLabels
          "scope_note": "..."                    ← English scopeNote (may be empty)
        }
    """
    all_uris = set(labels.keys())
    parent_uris = set(children.keys())
    leaf_uris = all_uris - parent_uris

    print(f'  {len(leaf_uris)} leaf concepts (no narrower children)')

    def get_paths_to_root(uri, visited=None):
        """Return list of paths (each path = list of URIs from root->node)."""
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

        leaf_alts   = alt_labels.get(leaf_uri, [])
        leaf_scope  = scope_notes.get(leaf_uri, '')

        for path_uris in paths:
            path_labels = []
            for uri in path_uris:
                label = labels.get(uri, uri.split('/')[-1])
                path_labels.append(label)
            path_str = ' > '.join(path_labels)
            results.append({
                'leaf':       leaf_label,
                'path':       path_str,
                'path_lower': path_str.lower(),
                'alt_labels': leaf_alts,
                'scope_note': leaf_scope,
            })

    # Deduplicate
    seen = set()
    deduped = []
    for r in results:
        key = r['path']
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f'  {len(deduped)} unique root->leaf paths')
    return deduped


def strategy_text(entry, strategy_key):
    """Produce the text string for a given path entry and strategy."""
    leaf       = entry['leaf'].lower()
    path_lower = entry['path_lower']          # "a > b > c"
    parts      = [p.strip() for p in entry['path'].lower().split('>')]
    parents    = parts[:-1]                   # everything except the leaf
    alts       = [a.lower() for a in entry.get('alt_labels', [])]
    scope      = entry.get('scope_note', '').strip().lower()

    if strategy_key == 'leaf':
        return leaf
    elif strategy_key == 'anchor':
        return f'{leaf}: {path_lower}'
    elif strategy_key == 'context':
        if parents:
            return f'{leaf} is related to {", ".join(reversed(parents))}'
        return leaf
    elif strategy_key == 'bracket':
        if parents:
            return f'{leaf} ({", ".join(reversed(parents))})'
        return leaf
    elif strategy_key == 'enriched':
        # Combine: leaf / altLabels. scope note. path hierarchy.
        parts_out = [leaf]
        if alts:
            parts_out.append('; '.join(alts))
        if scope:
            parts_out.append(scope)
        parts_out.append(path_lower)
        return '. '.join(parts_out)
    else:
        raise ValueError(f'Unknown strategy: {strategy_key}')


def compute_embeddings(texts, model):
    """Compute RAW embeddings (no L2 norm) for a list of text strings."""
    print(f'  Embedding {len(texts)} texts ...')
    embeddings = model.encode(
        texts,
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

    # JSON metadata — include altLabels and scopeNote for enriched browser strategy
    display_data = [
        {
            'leaf':       d['leaf'],
            'path':       d['path'],
            'alt_labels': d.get('alt_labels', []),
            'scope_note': d.get('scope_note', ''),
        }
        for d in paths_data
    ]
    json_path = os.path.join(out_dir, 'elsst_paths.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(display_data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
    size = os.path.getsize(json_path)
    n_with_alts  = sum(1 for d in display_data if d['alt_labels'])
    n_with_scope = sum(1 for d in display_data if d['scope_note'])
    print(f'  Saved {json_path} ({size / 1024:.1f} KB, {len(display_data)} entries, '
          f'{n_with_alts} with altLabels, {n_with_scope} with scopeNotes)')


def main():
    rdf_path = os.path.abspath(RDF_PATH)
    if not os.path.isfile(rdf_path):
        print(f'ERROR: RDF file not found: {rdf_path}')
        sys.exit(1)

    out_dir = os.path.abspath(OUT_DIR)

    labels, alt_labels, scope_notes, children, parents, top_concepts = parse_elsst(rdf_path)
    paths_data = build_root_to_leaf_paths(labels, alt_labels, scope_notes, children, parents, top_concepts)

    if not paths_data:
        print('ERROR: No root->leaf paths found. Check the RDF file.')
        sys.exit(1)

    # Save shared metadata once
    save_cache(paths_data, out_dir)

    # Build embeddings for each model × strategy combination
    # Only save _l2.bin (the browser uses only L2-normalised files)
    for model_key, model_cfg in MODELS.items():
        print(f'\nLoading model {model_cfg["python_model"]} ...')
        model = SentenceTransformer(model_cfg['python_model'])

        for strat_key, strat_cfg in STRATEGIES.items():
            l2_path = os.path.join(out_dir, f'elsst_embeddings_{model_key}_{strat_key}_l2.bin')
            print(f'\n{"="*60}')
            print(f'Model: {model_cfg["display_name"]} ({model_key})  |  Strategy: {strat_cfg["display"]} ({strat_key})')
            print(f'{"="*60}')

            texts = [strategy_text(d, strat_key) for d in paths_data]
            # Show a few examples
            for t in texts[:3]:
                print(f'  e.g. "{t}"')

            raw_embs = compute_embeddings(texts, model)
            l2_embs  = l2_normalize(raw_embs)
            save_binary(l2_embs, l2_path)

    print(f'\nDone! All cache files saved to {out_dir}/')


if __name__ == '__main__':
    main()
