
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

def test_hdbscan():
    print("Generating synthetic data...")
    # Simulate embeddings
    np.random.seed(42)
    X = np.random.rand(100, 768).astype(np.float64)
    X = normalize(X, norm='l2', axis=1)
    # Precompute distances
    dist_matrix = euclidean_distances(X, X).astype(np.float64)
    # Ensure zero diagonal
    np.fill_diagonal(dist_matrix, 0.0)
    
    min_cluster_sizes = [2]
    cluster_selection_epsilons = [0.75, 1.0]
    
    print("Starting loop (leaf method)...")
    for min_size in min_cluster_sizes:
        for epsilon in cluster_selection_epsilons:
            print(f"Testing min_size={min_size}, epsilon={epsilon}")
            try:
                clusterer = HDBSCAN(
                    min_cluster_size=int(min_size), 
                    cluster_selection_epsilon=float(epsilon), 
                    metric='precomputed',
                    cluster_selection_method='leaf',
                    copy=True
                )
                labels = clusterer.fit_predict(dist_matrix)
                print(f"  Success.")
            except Exception as e:
                print(f"  FAILED with {e}")

if __name__ == "__main__":
    test_hdbscan()
