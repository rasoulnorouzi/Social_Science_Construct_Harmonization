"""
Smoke test for shortest path calculation between concept pairs.
Shows examples with path 0, 1, 2, 3, and >3 if available.
"""

from process_elsst_train_test import parse_rdf_file, extract_concepts, find_shortest_path

RDF_FILE_PATH = "datasets/raw_datasets/ELSST_R5.rdf"

# Load concepts
tree_root = parse_rdf_file(RDF_FILE_PATH)
concepts = extract_concepts(tree_root)
uris = list(concepts.keys())

results = {0: [], 1: [], 2: [], 3: [], 'gt3': []}

# Try to find pairs for each path length
for i, uri1 in enumerate(uris[:50]):
    for j, uri2 in enumerate(uris[i+1:i+50]):
        if uri1 == uri2:
            continue
        path = find_shortest_path(uri1, uri2, concepts)
        if path == 0 and len(results[0]) < 2:
            results[0].append((uri1, uri2, path))
        elif path == 1 and len(results[1]) < 2:
            results[1].append((uri1, uri2, path))
        elif path == 2 and len(results[2]) < 2:
            results[2].append((uri1, uri2, path))
        elif path == 3 and len(results[3]) < 2:
            results[3].append((uri1, uri2, path))
        elif path > 3 and len(results['gt3']) < 2:
            results['gt3'].append((uri1, uri2, path))
        if all(len(results[k]) >= 2 for k in results):
            break
    if all(len(results[k]) >= 2 for k in results):
        break


def get_path_sequence(uri1, uri2, concepts):
    # BFS to reconstruct the path
    from collections import deque
    if uri1 == uri2:
        return [uri1]
    visited = {uri1}
    queue = deque([(uri1, [uri1])])
    while queue:
        current, path = queue.popleft()
        neighbors = set(concepts.get(current, {}).get('broader', []) + concepts.get(current, {}).get('narrower', []))
        for neighbor in neighbors:
            if neighbor == uri2:
                return path + [neighbor]
            if neighbor not in visited and neighbor in concepts:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


print("Smoke test: Shortest path between concept pairs (with full path)")
for k in [0, 1, 2, 3, 'gt3']:
    for uri1, uri2, pathlen in results[k]:
        label1 = concepts[uri1]['prefLabel']
        label2 = concepts[uri2]['prefLabel']
        path_uris = get_path_sequence(uri1, uri2, concepts)
        path_labels = [concepts[u]['prefLabel'] if u in concepts else u for u in path_uris]
        print(f"Pair: '{label1}' <-> '{label2}' | Shortest path: {pathlen}")
        print("  Path: " + " -> ".join(path_labels))

# Specific test for TEACHING PROFESSION <-> FERTILITY TREATMENT
def find_uri_by_label(label):
    for uri, c in concepts.items():
        if c['prefLabel'] == label:
            return uri
    return None

uri_tp = find_uri_by_label('LOANS')
uri_ft = find_uri_by_label('ADMINISTRATION OF JUSTICE')
if uri_tp and uri_ft:
    pathlen = find_shortest_path(uri_tp, uri_ft, concepts)
    path_uris = get_path_sequence(uri_tp, uri_ft, concepts)
    path_labels = [concepts[u]['prefLabel'] if u in concepts else u for u in path_uris]
    print(f"\nPair: 'LOANS' <-> 'ADMINISTRATION OF JUSTICE' | Shortest path: {pathlen}")
    print("  Path: " + " -> ".join(path_labels))
else:
    print("\nCould not find both 'LOANS' and 'ADMINISTRATION OF JUSTICE' in the concepts.")