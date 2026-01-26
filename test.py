# %%
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
# %%
def get_unique_terms_embeddings(model, df):
    """
    Extracts unique terms and computes their embeddings.
    """
    unique_terms = pd.unique(df[['term1', 'term2']].values.ravel('K'))
    print(f"Computed embeddings for {len(unique_terms)} unique terms.")
    embeddings = model.encode(unique_terms, convert_to_tensor=False, show_progress_bar=False)
    # L2 normalize for cosine similarity simulation with Euclidean distance
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # Ensure standard double precision and contiguous memory for Scikit-Learn/Cython compatibility
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)
    return unique_terms, embeddings
# %%
# bert uncased
checkpoint = "bert-base-uncased"

word_embedding_model = models.Transformer(checkpoint, model_args={'local_files_only': True})
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# %%
concepts = [
    "heart attack",
    "myocardial infarction",
    "diabetes mellitus",
    "high blood sugar",
    "hypertension",
    "high blood pressure",
    "stroke",
    "cerebrovascular accident",
    "sodium chloride",
    "table salt",
    "social media",
    "online networking",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks",
    "astronomy",
    "study of stars",
    "quantum mechanics",
    "physics of the very small",
    "climate change",
    "global warming",
    "war",
    "armed conflict",
    "peace",
    "harmony"
]
embeddings = model.encode(concepts, convert_to_tensor=True, show_progress_bar=False)
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
# %%
embeddings.shape
# %%
# continuous array of float64
embeddings = np.ascontiguousarray(embeddings, dtype=np.float64)
embeddings.shape
# %%
concept_to_embedding = {concept: emb for concept, emb in zip(concepts, embeddings)}
# %%concept_to_embedding["heart attack"]
# %%
# lets doing a umap reduction
from umap import UMAP
umap_model = UMAP(n_components=3, n_neighbors=5, metric='euclidean', random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)
# %%
embeddings_2d.shape
# %%
continuous_2d = np.ascontiguousarray(embeddings_2d, dtype=np.float64)
continuous_2d.shape
# %%
concept_to_embedding_2d = {concept: emb for concept, emb in zip(concepts, continuous_2d)}
# %%
concept_to_embedding_2d["heart attack"]
# %%
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=2, metric='cosine', copy=True)
labels = clusterer.fit_predict(continuous_2d)
# %%
labels
# %%
for concept, label in zip(concepts, labels):
    print(f"{concept}: Cluster {label}")
# %%
# putting similar concepts together
clusters = {}
for concept, label in zip(concepts, labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(concept)
# %%
for label, concepts_in_cluster in clusters.items():
    print(f"Cluster {label}: {concepts_in_cluster}")
# %%
import pandas as pd
import model_utils_clustering as model_utils
import model_utils_shared
import config
import warnings
# %%
models_to_test = config.MODELS
all_clustering_results = []
for model_name, model_conf in models_to_test.items():
    display_name = model_conf['display_name']
    print(f"\n{'='*50}")
    print(f" Processing: {display_name}")
    print(f"{'='*50}")
    print(model_conf['type'])
    print(config.CLUSTERING_PARAMS) 
# %%
