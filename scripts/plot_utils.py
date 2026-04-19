import glob
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


TECHNIQUE_MAP = {
    "clustering": "Clustering",
    "pairwise": "Pairwise",
    "seeded": "Seeded",
    "Clustering": "Clustering",
    "Pairwise": "Pairwise",
    "Seeded": "Seeded",
}

MODEL_MAP = {
    "SciBERT": "SciBERT (SciVocab)",
    "SciBERT (SciVocab)": "SciBERT (SciVocab)",
    "allenai/scibert_scivocab_uncased": "SciBERT (SciVocab)",
    "All MPNet Base v2": "All-MPNet-Base-v2",
    "All-MPNet-Base-v2": "All-MPNet-Base-v2",
    "all-mpnet-base-v2": "All-MPNet-Base-v2",
    "MPNet Personality": "MPNet-Personality",
    "MPNet-Personality": "MPNet-Personality",
    "dwulff/mpnet-personality": "MPNet-Personality",
    "BERT Base": "BERT Base",
    "bert-base-uncased": "BERT Base",
}


def read_csv_optional(path: str) -> Tuple[pd.DataFrame, bool]:
    if os.path.exists(path):
        return pd.read_csv(path), False
    return pd.DataFrame(), True


def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Technique" in out.columns and "technique" not in out.columns:
        out = out.rename(columns={"Technique": "technique"})
    if "Model" in out.columns and "model" not in out.columns:
        out = out.rename(columns={"Model": "model"})
    if "technique" in out.columns:
        out["technique"] = out["technique"].map(lambda v: TECHNIQUE_MAP.get(v, v))
    if "model" in out.columns:
        out["model"] = out["model"].map(lambda v: MODEL_MAP.get(v, v))
    return out


def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def sanitize_model_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def list_csv(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))


def minmax_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    lo = s.min()
    hi = s.max()
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - lo) / (hi - lo)


def template_text(ax, title: str, detail: str = "") -> None:
    ax.set_axis_off()
    msg = "[TEMPLATE — run analysis notebooks first]"
    if detail:
        msg = f"{msg}\n{detail}"
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
    ax.set_title(title, fontsize=10, weight="bold")


def ensure_ordered_categorical(df: pd.DataFrame, column: str, order: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if column in out.columns:
        out[column] = pd.Categorical(out[column], categories=list(order), ordered=True)
        out = out.sort_values(column)
    return out


def row_normalize_confusion(tp: float, fp: float, fn: float, tn: float) -> np.ndarray:
    mat = np.array([[tp, fn], [fp, tn]], dtype=float)
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return mat / row_sum
