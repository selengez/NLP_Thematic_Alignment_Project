# Measuring Thematic Drift in Computational Linguistics

## Project Specification: P7 — Analyzing Thematic Alignment in Scientific Journals

This project investigates whether articles published in a scientific journal align with the journal's stated **Aims & Scope**. The main goal is to measure thematic coherence, detect possible thematic drift over time, and identify articles that are unusually distant from the journal's declared research focus.

In this study, the target journal is **Transactions of the Association for Computational Linguistics (TACL)**. The project analyzes TACL articles published between **2015 and 2025**.

---

## AI Usage Disclaimer

Parts of this project were developed with the assistance of OpenAI's ChatGPT. The AI was used to help fix code errors, structure the methodology, improve explanatory text, revise visualizations, and support debugging during implementation.

The dataset collection, analysis design, figures, interpretation, and final project content were reviewed, edited, and validated by me. I take full responsibility for the final content, including its accuracy, relevance, and academic integrity.

---

## Objective

The main objective is to quantitatively assess whether articles published in TACL align with the journal's official **Aims & Scope**.

The analysis addresses three research questions:

1. How well do TACL articles align with the journal's stated Aims & Scope?
2. Has this alignment changed from 2015 to 2025?
3. What are the main research topics in the corpus, and how have their shares shifted over time?

---

## Reference Concept: Aims & Scope as Thematic Benchmark

The official TACL **Aims & Scope** text is used as the fixed thematic reference for this project.

In this study, the Aims & Scope is treated as a **thematic benchmark**, not as a manually labelled ground-truth dataset. Each article is compared against this reference using semantic similarity.

---

## Dataset

The dataset was collected using the **OpenAlex API**.

Final dataset:

| Item | Description |
|---|---|
| Journal | Transactions of the Association for Computational Linguistics |
| Years | 2015–2025 |
| Number of articles | 620 |
| Text used | Title + abstract |
| Source | OpenAlex |
| Metadata | Article ID, title, abstract, publication date, year, journal category |
| Cleaning | Duplicate IDs, missing abstracts, and very short abstracts were removed |

The dataset is stored as:

`dataset_nlp_cl.csv`

---

## Core Pipeline

The project follows this pipeline:

1. **Data collection**
   - Collect TACL article metadata and abstracts using OpenAlex.

2. **Text preprocessing**
   - Combine title and abstract.
   - Apply minimal cleaning: unicode normalization, lowercasing, URL removal, and whitespace cleanup.
   - Stop words and stemming are not applied because transformer models benefit from full sentence structure.

3. **Embedding generation**
   - Encode each article using `allenai-specter`.
   - Encode the TACL Aims & Scope text using the same model.
   - Store article embeddings in `embeddings.npy`.

4. **Alignment scoring**
   - Compute cosine similarity between each article embedding and the Aims & Scope embedding.
   - This produces one alignment score per article.

5. **Classification**
   - Use z-scores to classify articles into high alignment, aligned, and outlier categories.

6. **Thematic drift analysis**
   - Aggregate alignment scores by year.
   - Visualize yearly mean alignment and standard deviation.

7. **Topic analysis**
   - Apply KMeans clustering to article embeddings.
   - Use UMAP only for 2D visualization.
   - Track topic-share changes over time.

8. **Qualitative inspection**
   - Inspect the highest- and lowest-scoring articles to check whether the alignment metric makes sense.

---

## Methods

### Embedding Model

The project uses `allenai-specter`.

SPECTER is a scientific document embedding model trained on paper titles, abstracts, and citation context. It is suitable for this project because the dataset consists of academic NLP articles.

### Alignment Metric

Alignment is measured with cosine similarity:

`alignment_score = cosine(article_embedding, scope_embedding)`

Higher scores indicate stronger thematic similarity to the journal's Aims & Scope.

### Classification Rule

Articles are classified using z-scores:

| Category | Criterion | Meaning |
|---|---|---|
| High alignment | z > +1.0 | Strongly on-scope |
| Aligned | -1.5 ≤ z ≤ +1.0 | Normal alignment range |
| Outlier | z < -1.5 | Significantly lower alignment |

---

## Main Results

The final analysis produced the following results:

| Metric | Value |
|---|---:|
| Mean alignment score | 0.793 |
| High-alignment articles | 94 |
| Aligned articles | 476 |
| Outlier articles | 50 |

The results suggest that TACL articles are broadly aligned with the journal's stated scope. The yearly alignment scores remain relatively stable from 2015 to 2025, so there is no strong evidence of major thematic drift.

The topic analysis identifies three broad research clusters:

1. Traditional NLP
2. Deep Learning NLP
3. LLMs / Generative AI

From 2022 to 2025, Deep Learning NLP increases in share, while Traditional NLP decreases slightly. LLMs / Generative AI remain visible but do not dominate the journal.

---

## Visualizations

The project generates the following figures:

| File | Description |
|---|---|
| `paper_volume_by_year.png` | TACL article counts per year |
| `abstract_length_distribution.png` | Title + abstract word-count distribution |
| `alignment_distribution.png` | Cosine-similarity alignment score distribution |
| `alignment_inspection_table.png` | Examples of high- and low-alignment articles |
| `drift_over_time.png` | Mean yearly alignment with ±1 standard deviation |
| `umap_cluster_map.png` | 2D UMAP visualization of KMeans clusters |
| `topic_market_share.png` | Yearly topic-share plot by cluster |

These figures are saved in the `visualizations/` folder.

---

## Project Structure

NLP_Alignment_Project/
│
├── data_loader.py
├── text_processing.py
├── embedder.py
├── alignment.py
├── visualizer.py
├── main.py
├── requirements.txt
│
├── dataset_nlp_cl.csv
├── embeddings.npy
│
└── visualizations/
    ├── paper_volume_by_year.png
    ├── abstract_length_distribution.png
    ├── alignment_distribution.png
    ├── alignment_inspection_table.png
    ├── drift_over_time.png
    ├── umap_cluster_map.png
    └── topic_market_share.png

---

## File Descriptions

| File | Purpose |
|---|---|
| `data_loader.py` | Collects TACL articles from OpenAlex |
| `text_processing.py` | Cleans and prepares title + abstract text |
| `embedder.py` | Generates SPECTER embeddings |
| `alignment.py` | Computes alignment scores and classifications |
| `visualizer.py` | Creates and saves figures |
| `main.py` | Runs the full pipeline |
| `requirements.txt` | Lists Python dependencies |
| `dataset_nlp_cl.csv` | Final TACL article dataset |
| `embeddings.npy` | Cached article embeddings |

---

## How to Run

Install dependencies:

`pip install -r requirements.txt`

Run the full pipeline:

`python main.py`

The script will:

1. Load the dataset.
2. Preprocess article text.
3. Load or generate embeddings.
4. Compute alignment scores.
5. Classify articles by alignment level.
6. Create yearly drift statistics.
7. Run KMeans clustering.
8. Generate and save visualizations.

---

## Requirements

Main libraries:

- pandas
- numpy
- sentence-transformers
- scikit-learn
- scipy
- umap-learn
- matplotlib
- seaborn
- requests
- tqdm
- psutil

---

## Important Note on Cached Embeddings

The file `embeddings.npy` stores cached article embeddings. This avoids recomputing SPECTER embeddings every time the project is run.

If the dataset changes, delete `embeddings.npy` and rerun the pipeline so that embeddings are recomputed for the new dataset.

---

## Limitations

This project uses article titles and abstracts rather than full paper texts. This is common in large-scale bibliometric analysis, but some articles may be represented only partially.

The Aims & Scope text is short and general, so it should be understood as a thematic benchmark rather than a perfect ground truth.

The cluster labels are manually assigned after inspecting representative articles. A more detailed future version could use BERTopic or automatic keyword extraction for more interpretable topic labels.

The drift analysis is descriptive. A future extension could add formal statistical testing, such as regression over yearly alignment scores.

---

## Conclusion

This project builds a reproducible NLP pipeline for measuring thematic alignment between journal articles and a journal's stated Aims & Scope.

Using 620 TACL articles from 2015 to 2025, SPECTER embeddings, cosine similarity, z-score classification, and clustering, the analysis shows that TACL remains broadly aligned with its official scope.

The journal's thematic profile evolves over time, especially with changes in deep learning and LLM-related work, but the results do not show strong evidence that TACL has drifted away from its declared research mission.

