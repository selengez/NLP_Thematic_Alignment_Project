"""
text_processing.py
------------------
Cleans and prepares TACL article text for embedding.

KEY DECISION:
    Each article is represented as title + abstract. Cleaning is kept minimal:
    unicode normalization, lowercasing, URL removal, and whitespace cleanup.

    Stop words and stemming are not applied because SPECTER is a transformer
    model and works best with full sentence structure.
"""
import unicodedata
import re
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Low-level cleaning helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalize unicode, lowercase, collapse whitespace, strip URLs.

    Parameters
    ----------
    text : str
        Raw string (title or abstract).

    Returns
    -------
    str
        Cleaned string.
    """
    text = unicodedata.normalize("NFKC", text)   # normalize unicode (e.g. ligatures)
    text = text.lower()                           # lowercase
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"\s+", " ", text)             # collapse multiple spaces / newlines
    text = text.strip()
    return text


def build_input_text(row) -> str:
    """
    Concatenate cleaned title and abstract into a single string.
    The period separator mirrors the SPECTER pre-training format (title. abstract).

    Parameters
    ----------
    row : pd.Series
        DataFrame row with 'title' and 'abstract' columns.

    Returns
    -------
    str
        Combined, cleaned text ready for embedding.
    """
    return clean_text(str(row["title"]) + ". " + str(row["abstract"]))


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the full DataFrame and add the 'input_text' column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 'title' and 'abstract' columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with an additional 'input_text' column.
    """
    df = df.copy()

    tqdm.pandas(desc="Cleaning text")
    df["input_text"] = df.progress_apply(build_input_text, axis=1)

    # Drop rows where cleaning produced empty / very short text
    df = df.dropna(subset=["input_text"])
    df = df[df["input_text"].str.len() > 100]
    df = df.reset_index(drop=True)

    # Convenience columns for analysis
    df["word_count"] = df["input_text"].apply(lambda x: len(x.split()))

    print(
        f"[text_processing] Preprocessed {len(df)} papers. "
        f"Mean word count: {df['word_count'].mean():.1f}"
    )
    return df
