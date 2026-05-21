import os
import pandas as pd
from text_processing import preprocess_dataframe
from embedder import TextEmbedder
from alignment import compute_alignment_scores, classify_papers, compute_yearly_stats, evaluate_k_range
from visualizer import (
    plot_paper_volume,
    plot_abstract_length_distribution,
    plot_alignment_distribution,
    plot_drift_over_time,
    plot_umap_clusters,
    plot_topic_market_share,
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def main():
    if not os.path.exists('dataset_nlp_cl.csv'):
        raise SystemExit('dataset_nlp_cl.csv not found; cannot continue.')

    df_raw = pd.read_csv('dataset_nlp_cl.csv', sep='|')
    print(f'Loaded dataset with {len(df_raw)} rows; years {df_raw["year"].min()}-{df_raw["year"].max()}')

    df = preprocess_dataframe(df_raw)
    plot_paper_volume(df, save=True)
    plot_abstract_length_distribution(df, save=True)

    embedder = TextEmbedder(device='cpu')
    AIMS_AND_SCOPE = (
        'Computational Linguistics is the longest-running publication dedicated to the computational '
        'and mathematical aspects of language. The journal publishes research on all aspects of natural '
        'language processing and computational linguistics, including syntax, semantics, pragmatics, discourse, '
        'machine translation, information extraction, text summarization, question answering, dialogue systems, '
        'and language generation. The journal covers both theoretical foundations and practical applications, '
        'with emphasis on computational methods, empirical evaluation, and the development of language resources and tools.'
    )

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
    plot_alignment_distribution(df, save=True)

    yearly_stats = compute_yearly_stats(df)
    plot_drift_over_time(yearly_stats, save=True)

    sil_df = evaluate_k_range(paper_embeddings, k_values=[2, 3, 4, 5])
    print(sil_df.to_string(index=False))

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(paper_embeddings)

    if HAS_UMAP:
        umap_model = UMAP(n_components=2, random_state=42, metric='cosine', init='spectral')
        projection = umap_model.fit_transform(paper_embeddings)
    else:
        pca_model = PCA(n_components=2, random_state=42)
        projection = pca_model.fit_transform(paper_embeddings)

    df['umap_x'] = projection[:, 0]
    df['umap_y'] = projection[:, 1]

    plot_umap_clusters(
        df,
        cluster_labels={0: 'Traditional NLP', 1: 'Deep Learning NLP', 2: 'LLMs / Generative AI'},
        save=True,
    )
    plot_topic_market_share(
        df,
        cluster_labels={0: 'Traditional NLP', 1: 'Deep Learning NLP', 2: 'LLMs / Generative AI'},
        year_start=2022,
        save=True,
    )

    print('Generated PNG files:')
    for fname in sorted(f for f in os.listdir('.') if f.endswith('.png')):
        print(' ', fname)


if __name__ == '__main__':
    main()
