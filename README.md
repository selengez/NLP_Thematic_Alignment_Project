# Measuring Thematic Drift in Computational Linguistics
**Alignment of TACL Articles with Journal Aims & Scope (2015–2025)**

Selen Gezginci · 32204A · Master in Data Science for Economics

---

## What It Does

Measures whether TACL articles stay aligned with the journal's Aims & Scope over time, using SPECTER embeddings and cosine similarity.

**Key results:** mean alignment = 0.793 · stable across 2015–2025 · no significant drift detected

---

## Project Structure

```
main.py              # Entry point — runs the full pipeline
data_loader.py       # Fetches TACL articles via OpenAlex API
text_processing.py   # Text cleaning and input_text construction
embedder.py          # SPECTER embedding model wrapper
alignment.py         # Cosine similarity scoring and z-score classification
visualizer.py        # All figures (saved to visualizations/)
requirements.txt     # Python dependencies
dataset_nlp_cl.csv   # Cached dataset (pipe-separated)
embeddings.npy       # Cached embeddings (auto-generated on first run)
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Embeddings are cached to `embeddings.npy` after the first run. To re-fetch the dataset:

```bash
python data_loader.py
```

---

## Output

Six PNG figures saved to `visualizations/`: paper volume, abstract length, alignment distribution, thematic drift, UMAP cluster map, topic market share.

---

## Limitations

- Scope embedded as a single vector — scores are approximate, not editorial judgements
- Abstracts only, no full paper text
- Cluster labels assigned manually after one fixed run (`random_state=42`)
-No strong evidence of thematic drift is visible from the yearly mean scores.
