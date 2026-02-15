# %%
import re
TOKEN_RE = re.compile(r"[a-z0-9]+")
def simple_tokens(text):
    """Tokenise text into lowercase alphanumeric words."""
    return TOKEN_RE.findall(str(text).lower())


def letter_ngrams(text, n=3):
    """Generate character n-grams for letter-level similarity."""
    text = "".join(c for c in str(text).lower() if c.isalnum())
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_sim(a_tokens, b_tokens):
    """Jaccard similarity between two token/n-gram sets."""
    if not a_tokens and not b_tokens:
        return 0.0
    inter = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(inter) / len(union) if union else 0.0

# %%
test_text = "This is a test. 1234! Does it work?"
tokens = simple_tokens(test_text)
print(tokens)
# %%
ngrams = letter_ngrams(test_text, n=3)
print(ngrams)
# %%
