import pandas as pd

# Paths to the CSV files
neg_path = "datasets/processed_datasets/train_negative_pairs.csv"
pos_path = "datasets/processed_datasets/train_positive_pairs.csv"

# Load the data
neg_df = pd.read_csv(neg_path)
pos_df = pd.read_csv(pos_path)

# Extract all terms from both columns in both files
neg_terms = pd.unique(neg_df[['term1', 'term2']].values.ravel('K'))
pos_terms = pd.unique(pos_df[['term1', 'term2']].values.ravel('K'))

# Combine and deduplicate
all_terms = set([str(term) for term in neg_terms if pd.notna(term)] + [str(term) for term in pos_terms if pd.notna(term)])

print(f"Total unique terms/concepts (by text): {len(all_terms)}")
