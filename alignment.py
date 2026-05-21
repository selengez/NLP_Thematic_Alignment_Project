"""
alignment.py
------------
Computes cosine similarity between paper embeddings and the scope embedding,
classifies papers by alignment level, and aggregates yearly statistics.

METHODOLOGY:
    1. Each paper embedding is compared to the single scope_embedding vector
       (768d, representing the journal's Aims & Scope) using cosine similarity.
    2. Z-score outlier detection identifies papers significantly below or above
       the mean — same method as the medical-AI example project.
    3. Yearly aggregation feeds the thematic-drift visualisation (RQ2).
"""

from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import numpy as np
import pandas as pd
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def compute_alignment_scores(
    paper_embeddings: np.ndarray,
    scope_embedding: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between each paper and the scope vector.

    Parameters
    ----------
    paper_embeddings : np.ndarray, shape (N, 768)
        Stacked paper embedding matrix.
    scope_embedding : np.ndarray, shape (768,)
        The embedded Aims & Scope text.

    Returns
    -------
    np.ndarray, shape (N,)
        Cosine similarity scores in [−1, 1]; practically in [0, 1] for text.
    """
    scope_vec = scope_embedding.reshape(1, -1)
    scores = cosine_similarity(paper_embeddings, scope_vec).flatten()
    return scores


# ---------------------------------------------------------------------------
# Paper classification (Z-score based)
# ---------------------------------------------------------------------------

def classify_papers(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """
    Classify papers into three categories based on alignment score z-scores.

    Categories
    ----------
    high_alignment : z > +1.0  — well above average; tightly on-scope
    aligned        : −1.5 ≤ z ≤ +1.0  — within normal range
    outlier        : z < −1.5  — significantly off-scope

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 'alignment_score' column.

    Returns
    -------
    (df, mean_score, std_score)
    """
    mean = df["alignment_score"].mean()
    std  = df["alignment_score"].std()

    df = df.copy()
    df["z_score"]  = stats.zscore(df["alignment_score"])
    df["category"] = "aligned"
    df.loc[df["z_score"] < -1.5, "category"] = "outlier"
    df.loc[df["z_score"] >  1.0, "category"] = "high_alignment"

    counts = df["category"].value_counts()
    n = len(df)
    print(
        f"[alignment] Mean score: {mean:.4f} | Std: {std:.4f}\n"
        f"           high_alignment: {counts.get('high_alignment', 0)} "
        f"({counts.get('high_alignment', 0)/n*100:.1f}%)\n"
        f"           aligned:        {counts.get('aligned', 0)} "
        f"({counts.get('aligned', 0)/n*100:.1f}%)\n"
        f"           outlier:        {counts.get('outlier', 0)} "
        f"({counts.get('outlier', 0)/n*100:.1f}%)"
    )
    return df, mean, std


# ---------------------------------------------------------------------------
# Yearly statistics (for RQ2 drift analysis)
# ---------------------------------------------------------------------------

def compute_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate alignment scores by publication year.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'year' and 'alignment_score' columns.

    Returns
    -------
    pd.DataFrame
        Columns: year, mean_score, std_score, count, min_score, max_score.
    """
    yearly = (
        df.groupby("year")["alignment_score"]
        .agg(
            mean_score="mean",
            std_score="std",
            count="count",
            min_score="min",
            max_score="max",
        )
        .reset_index()
    )
    return yearly


# ---------------------------------------------------------------------------
# Silhouette-based k selection helper (used in Section 7)
# ---------------------------------------------------------------------------

def evaluate_k_range(
    embeddings: np.ndarray,
    k_values: list = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit KMeans for several k values and return silhouette scores.
    Used to justify the final choice of k=3.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    k_values : List[int]
        Values of k to test (default: [2, 3, 4, 5]).
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: k, silhouette_score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k_values is None:
        k_values = [2, 3, 4, 5]

    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=min(5000, len(embeddings)))
        rows.append({"k": k, "silhouette_score": round(sil, 4)})
        print(f"[alignment] k={k} → silhouette score = {sil:.4f}")

    return pd.DataFrame(rows)
