
import pandas as pd
import re
import numpy as np

# Replicate utilities
TOKEN_RE = re.compile(r"[a-z0-9]+")

def simple_tokens_word(text):
    return set(TOKEN_RE.findall(str(text).lower()))

def letter_ngrams(text, n=3):
    """Generate character n-grams for letter-level similarity."""
    text = str(text).lower()
    # Keep only alphanumeric chars
    text = "".join(c for c in text if c.isalnum())
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_sim(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union) if union else 0.0

# Load data
neg_path = "datasets/processed_datasets/test_negative_pairs.csv"
pos_path = "datasets/processed_datasets/test_positive_pairs.csv"

def load_data(path):
    print(f"Loading {path}...")
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
        return df
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

df_neg = load_data(neg_path)
df_pos = load_data(pos_path)

def analyze_similarity(df, label_name):
    if df is None: return
    
    print(f"\n{'='*20} ANALYSIS: {label_name.upper()} PAIRS {'='*20}")
    print(f"Total rows: {len(df)}")
    
    similarities = []
    pairs = []
    
    for idx, row in df.iterrows():
        t1 = str(row["term1"])
        t2 = str(row["term2"])
        s1 = letter_ngrams(t1, n=3)
        s2 = letter_ngrams(t2, n=3)
        
        sim = jaccard_sim(s1, s2)
        similarities.append(sim)
        pairs.append((t1, t2, sim))
            
    similarities = np.array(similarities)
    
    print(f"\n--- Statistics ({label_name}) ---")
    print(f"Min: {similarities.min():.4f}")
    print(f"Max: {similarities.max():.4f}")
    print(f"Mean: {similarities.mean():.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n--- Count exceeding thresholds ---")
    for t in thresholds:
        count = (similarities > t).sum()
        pct = (count / len(similarities)) * 100
        print(f"Count > {t:.1f}: {count:6d} ({pct:.1f}%)")
        
    # Examples
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n--- Top 10 High Similarity Examples ({label_name}) ---")
    for t1, t2, sim in pairs[:10]:
        print(f"{sim:.4f} | '{t1}' vs '{t2}'")

    pairs.sort(key=lambda x: x[2])
    print(f"\n--- Top 10 Low Similarity Examples ({label_name}) ---")
    for t1, t2, sim in pairs[:10]:
        print(f"{sim:.4f} | '{t1}' vs '{t2}'")

analyze_similarity(df_pos, "Positive (Synonyms)")
analyze_similarity(df_neg, "Negative (Unrelated)")

