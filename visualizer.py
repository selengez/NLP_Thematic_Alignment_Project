"""
visualizer.py
-------------
Creates and saves the main figures used in the TACL thematic alignment project.

Figures
-------
paper_volume_by_year.png           : TACL article counts per year
abstract_length_distribution.png   : title + abstract word-count distribution
alignment_distribution.png         : cosine-similarity alignment scores
alignment_inspection_table.png     : high- and low-alignment article examples
drift_over_time.png                : mean yearly alignment with ±1 std band
umap_cluster_map.png               : 2D UMAP view of KMeans clusters
topic_market_share.png             : yearly topic-share plot by cluster
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
OUTPUT_DIR = "visualizations"   # all PNGs saved to a dedicated folder


def _savefig(fig: plt.Figure, filename: str, dpi: int = 150) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[visualizer] Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Paper Volume by Year
# ---------------------------------------------------------------------------

def plot_paper_volume(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Bar chart: number of journal articles per year, annotated with counts.
    """
    counts = df.groupby("year").size().reset_index(name="count")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        counts["year"],
        counts["count"],
        color=PALETTE[0],
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, val in zip(bars, counts["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts["count"].max() * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    year_min = int(counts['year'].min())
    year_max = int(counts['year'].max())
    ax.set_title(
        f"Distribution of TACL Journal Papers by Year ({year_min}–{year_max})",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.set_xticks(counts["year"])
    ax.set_ylim(0, counts["count"].max() * 1.15)
    fig.tight_layout()

    if save:
        _savefig(fig, "paper_volume_by_year.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Alignment Score Distribution
# ---------------------------------------------------------------------------

def plot_alignment_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Histogram of cosine similarity scores, colour-coded by category.
    """
    COLOR_MAP = {
        "high_alignment": "#55A868",
        "aligned":        "#4C72B0",
        "outlier":        "#C44E52",
    }
    mean_score = df["alignment_score"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    for cat, color in COLOR_MAP.items():
        subset = df[df["category"] == cat]["alignment_score"]
        if len(subset) == 0:
            continue
        ax.hist(subset, bins=50, color=color, alpha=0.7, label=cat.replace("_", " ").title())

    ax.axvline(mean_score, color="red", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_score:.3f}")

    outlier_count = (df["category"] == "outlier").sum()
    high_count    = (df["category"] == "high_alignment").sum()
    ax.text(0.02, 0.95,
            f"Outliers: {outlier_count}   High-alignment: {high_count}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_title("Alignment Score Distribution vs Journal Scope", fontweight="bold", pad=14)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Number of Papers")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    if save:
        _savefig(fig, "alignment_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 2b — Alignment Inspection Table

def plot_inspection_table(
    df: pd.DataFrame,
    filename: str = "alignment_inspection_table.png",
    save: bool = True,
) -> plt.Figure:
    """
    Render the qualitative inspection table as a PNG for inclusion in reports.
    """
    display_cols = ["Type", "title", "year", "Score", "Inspection note"]
    table_df = df[display_cols].copy()

    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        colLoc="center",
        loc="center",
        cellColours=[[("#f8f8f8" if row == 0 else "white") for _ in table_df.columns] for row in range(len(table_df))],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9d9d9")
        elif col == 0:
            cell.set_text_props(weight="bold")
        if col in [2, 3]:
            cell.set_text_props(ha="center")

    fig.tight_layout()
    if save:
        _savefig(fig, filename, dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Thematic Drift Over Time
# ---------------------------------------------------------------------------

def plot_drift_over_time(yearly_stats: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Line chart: mean alignment per year + ±1 std shaded band.
    Annotates the year with the lowest and highest mean score.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    years      = yearly_stats["year"].values
    means      = yearly_stats["mean_score"].values
    stds       = yearly_stats["std_score"].values
    overall    = means.mean()

    ax.plot(years, means, marker="o", color=PALETTE[0], linewidth=2.2,
            markersize=7, label="Mean Alignment Score")
    ax.fill_between(years, means - stds, means + stds,
                    color=PALETTE[0], alpha=0.15, label="±1 Std Dev")
    ax.axhline(overall, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall Mean = {overall:.3f}")

    # Annotate min and max years
    min_idx = np.argmin(means)
    max_idx = np.argmax(means)
    for idx, label, va_pos in [(min_idx, "Min", "top"), (max_idx, "Max", "bottom")]:
        ax.annotate(
            f"{label}\n{years[idx]}",
            xy=(years[idx], means[idx]),
            xytext=(years[idx], means[idx] + (0.005 if va_pos == "bottom" else -0.005)),
            ha="center", va=va_pos, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    year_min = int(years.min())
    year_max = int(years.max())
    ax.set_title(
        f"Thematic Drift: Mean Alignment Score by Year ({year_min}–{year_max})",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_xticks(years)
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    if save:
        _savefig(fig, "drift_over_time.png")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — UMAP Cluster Map
# ---------------------------------------------------------------------------

def plot_umap_clusters(
    df: pd.DataFrame,
    cluster_labels: dict,
    save: bool = True,
) -> plt.Figure:
    """
    2D scatter of UMAP embeddings coloured by KMeans cluster, with centroid labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'umap_x', 'umap_y', 'cluster' columns.
    cluster_labels : dict
        Maps cluster id (int) → human-readable label string,
        e.g. {0: "Traditional NLP", 1: "Deep Learning NLP", 2: "LLMs / Generative AI"}.
    """
    n_clusters = df["cluster"].nunique()
    colors = sns.color_palette("tab10", n_clusters)

    fig, ax = plt.subplots(figsize=(9, 6))

    for cid in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cid]
        label  = cluster_labels.get(cid, f"Topic {cid}")
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=40,
            color=colors[cid],
            alpha=0.65,
            label=label,
            rasterized=True,
        )
        # Centroid label
        cx = subset["umap_x"].mean()
        cy = subset["umap_y"].mean()
        ax.text(
            cx, cy,
            f"Topic {cid}\n{label}",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor=colors[cid]),
        )

    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    ax.set_title(
        f"The Semantic Landscape of TACL Research ({year_min}–{year_max})",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="upper right", markerscale=4, framealpha=0.9)
    fig.tight_layout()

    if save:
        _savefig(fig, "umap_cluster_map.png", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Topic Market Share (Stacked Area)
# ---------------------------------------------------------------------------

def plot_topic_market_share(
    df: pd.DataFrame,
    cluster_labels: dict,
    year_start: int = 2019,
    save: bool = True,
) -> plt.Figure:
    """
    Stacked area chart of topic market share (%) per year — normalised proportions.

    KEY DECISION: Market share (%) rather than raw counts is used so that the
    growing total publication volume does not mask compositional shifts.
    A topic rising from 5% to 40% is a paradigm shift even if all counts grew.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'year' and 'cluster' columns.
    cluster_labels : dict
        Maps cluster id → label string.
    year_start : int
        First year to include (default 2019 for the post-BERT era focus).
    """
    df_filt = df[df["year"] >= year_start].copy()

    pivot = (
        df_filt.groupby(["year", "cluster"])
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    if pivot_pct.empty:
        print(
            f"[visualizer] No data available for year_start={year_start}; "
            "plotting all available years instead."
        )
        pivot = df.groupby(["year", "cluster"]).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    colors = sns.color_palette("tab10", len(pivot_pct.columns))
    labels = [cluster_labels.get(c, f"Topic {c}") for c in pivot_pct.columns]
    values = [pivot_pct[col].values for col in pivot_pct.columns]

    ax.stackplot(
        pivot_pct.index,
        *values,
        labels=labels,
        colors=colors,
        alpha=0.85,
    )

    year_min = int(pivot_pct.index.min())
    year_max = int(pivot_pct.index.max())
    ax.set_title(
        f"The Paradigm Shift: Market Share of Research Pillars ({year_min}–{year_max})",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Market Share (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(pivot_pct.index)

    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], lbls[::-1], loc="upper left", framealpha=0.9)
    fig.tight_layout()

    if save:
        _savefig(fig, "topic_market_share.png")
    return fig


# ---------------------------------------------------------------------------
# Abstract Length Distribution (Figure 0)
# ---------------------------------------------------------------------------

def plot_abstract_length_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Histogram of word counts per paper with a mean line.
    Justifies the decision not to apply hard length truncation manually.
    """
    wc = df["word_count"] if "word_count" in df.columns else df["input_text"].apply(lambda x: len(x.split()))
    mean_wc = wc.mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(wc, bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(mean_wc, color="red", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_wc:.0f} words")
    ax.set_title("Abstract Length Distribution (Word Count per Paper)", fontweight="bold", pad=14)
    ax.set_xlabel("Word Count (title + abstract)")
    ax.set_ylabel("Number of Papers")
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    if save:
        _savefig(fig, "abstract_length_distribution.png")
    return fig
