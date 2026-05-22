import os
import pandas as pd
from text_processing import preprocess_dataframe
from embedder import TextEmbedder
from alignment import (
    compute_alignment_scores,
    classify_papers,
    compute_yearly_stats,
    evaluate_k_range,
    build_alignment_inspection_table,
)
from visualizer import (
    plot_paper_volume,
    plot_abstract_length_distribution,
    plot_alignment_distribution,
    plot_drift_over_time,
    plot_umap_clusters,
    plot_topic_market_share,
    plot_inspection_table,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Heuristic cluster labels for visualization.
# These are manually assigned topic names for the three KMeans clusters
# and should be treated as interpretive labels rather than canonical categories.
CLUSTER_LABELS = {
    0: 'Traditional NLP',
    1: 'Deep Learning NLP',
    2: 'LLMs / Generative AI',
}

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


JOURNAL_NAME = 'Transactions of the Association for Computational Linguistics (TACL)'
AIMS_AND_SCOPE = (
    'Transactions of the Association for Computational Linguistics (TACL) publishes papers across the full spectrum of computational linguistics and natural language processing. '
    'The journal emphasizes high-quality computational methods, empirical evaluation, and linguistic analysis for tasks such as syntax, semantics, discourse, pragmatics, information extraction, machine translation, summarization, question answering, dialogue systems, and language generation. '
    'TACL prioritizes work that advances the theoretical foundations, architectures, datasets, resources, and applications of language technologies within an editorially curated journal scope.'
)


def main():
    if not os.path.exists('dataset_nlp_cl.csv'):
        raise SystemExit('dataset_nlp_cl.csv not found; cannot continue.')

    df_raw = pd.read_csv('dataset_nlp_cl.csv', sep='|')
    print(f'[main] Using journal-scoped dataset: {JOURNAL_NAME}')
    print(f'[main] Loaded dataset with {len(df_raw)} rows; years {df_raw["year"].min()}-{df_raw["year"].max()}')
    print(f'[main] Dataset source: OpenAlex journal articles for {JOURNAL_NAME}')

    df = preprocess_dataframe(df_raw)
    plot_paper_volume(df, save=True)
    plot_abstract_length_distribution(df, save=True)

    embedder = TextEmbedder(device='cpu')
    scope_embedding = embedder.encode_scope(AIMS_AND_SCOPE)
    paper_embeddings = embedder.encode_or_load(
        texts=df['input_text'].tolist(),
        path='embeddings.npy',
        batch_size=8,
        resume=True,
    )

    print('Embedding complete:', paper_embeddings.shape)

    scores = compute_alignment_scores(paper_embeddings, scope_embedding)
    df['alignment_score'] = scores
    df, mean_score, std_score = classify_papers(df)
    print(f'Mean alignment: {mean_score:.4f}, std: {std_score:.4f}')

    inspection_table = build_alignment_inspection_table(df)
    plot_inspection_table(
        inspection_table,
        filename='alignment_inspection_table.png',
        save=True,
    )

    plot_alignment_distribution(df, save=True)

    yearly_stats = compute_yearly_stats(df)
    plot_drift_over_time(yearly_stats, save=True)

    sil_df = evaluate_k_range(paper_embeddings, k_values=[2, 3, 4, 5])
    print(sil_df.to_string(index=False))

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(paper_embeddings)

    if HAS_UMAP:
        umap_model = UMAP(
            n_components=2,
            n_neighbors=20,
            min_dist=0.03,
            random_state=42,
            metric='cosine',
            init='spectral',
        )
        projection = umap_model.fit_transform(paper_embeddings)
    else:
        pca_model = PCA(n_components=2, random_state=42)
        projection = pca_model.fit_transform(paper_embeddings)

    df['umap_x'] = projection[:, 0]
    df['umap_y'] = projection[:, 1]

    plot_umap_clusters(
        df,
        cluster_labels=CLUSTER_LABELS,
        save=True,
    )
    plot_topic_market_share(
        df,
        cluster_labels=CLUSTER_LABELS,
        year_start=2022,
        save=True,
    )

    print('Generated PNG files:')
    if os.path.isdir('visualizations'):
        for fname in sorted(f for f in os.listdir('visualizations') if f.endswith('.png')):
            print(' ', os.path.join('visualizations', fname))
    else:
        print('  (no visualizations directory found)')


if __name__ == '__main__':
    main()
