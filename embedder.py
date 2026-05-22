"""
embedder.py
-----------
Encodes TACL article texts and the journal Aims & Scope using allenai-specter.

MODEL CHOICE:
    SPECTER is a scientific document embedding model trained on paper titles,
    abstracts, and citation context. I use it because the dataset contains
    academic NLP articles, so domain-specific scientific embeddings are more
    suitable than generic sentence embeddings.

    The TACL Aims & Scope text is embedded with the same model and used as the
    reference vector for cosine-similarity alignment scoring.

CACHING:
    Article embeddings are saved to 'embeddings.npy' after the first run and
    loaded directly on later runs. If the dataset changes, this cache should be
    deleted and recomputed.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import math
import time
from typing import List, Optional

try:
    import psutil
except Exception:
    psutil = None


class TextEmbedder:
    """Wrapper around allenai-specter for batch encoding of scientific texts."""

    MODEL_NAME = "allenai-specter"
    FALLBACK_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 768

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        name = model_name or self.MODEL_NAME
        self.device = device
        print(f"[embedder] Loading model '{name}' …")
        try:
            if self.device:
                self.model = SentenceTransformer(name, device=self.device)
            else:
                self.model = SentenceTransformer(name)
            print(f"[embedder] Model loaded. Output dim: {self.EMBEDDING_DIM}d")
        except Exception as e:
            print(f"[embedder] WARNING: failed to load '{name}': {e}")
            if name != self.FALLBACK_MODEL:
                print(f"[embedder] Falling back to lightweight model '{self.FALLBACK_MODEL}'.")
                self.model = SentenceTransformer(self.FALLBACK_MODEL)
                print(f"[embedder] Fallback model loaded. Output dim may differ.")
            else:
                raise

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
        path: Optional[str] = None,
        resume: bool = False,
        batch_save_interval: int = 10,
        max_batches: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode a list of strings into 768-dimensional vectors.

        Parameters
        ----------
        texts : list[str]
            Cleaned input texts (title + abstract).
        batch_size : int
            Batch size for GPU/CPU inference.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        np.ndarray, shape (N, 768)
            L2-normalised dense embeddings.
        """
        # If no persistence requested, fall back to the library encode (small inputs)
        n_texts = len(texts)
        if n_texts == 0:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        # If user did not request incremental saving, but input is small, use direct encode
        if path is None:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        # Use memmap-backed storage to avoid holding full matrix in RAM
        progress_path = path + ".progress"
        n_batches = math.ceil(n_texts / batch_size)
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
            n_texts = min(n_texts, n_batches * batch_size)

        # Create or open memmap file
        if not os.path.exists(path):
            print(f"[embedder] Creating memmap at '{path}' for {n_texts} x {self.EMBEDDING_DIM} (float32)")
            mm = np.memmap(path, dtype='float32', mode='w+', shape=(n_texts, self.EMBEDDING_DIM))
            mm.flush()
            completed = 0
        else:
            mm = np.memmap(path, dtype='float32', mode='r+', shape=(n_texts, self.EMBEDDING_DIM))
            completed = 0
            if resume and os.path.exists(progress_path):
                try:
                    with open(progress_path, 'r') as f:
                        completed = int(f.read().strip())
                        print(f"[embedder] Resuming from batch index {completed}")
                except Exception:
                    completed = 0

        # Batch loop
        for bidx in range(completed, n_batches):
            start = bidx * batch_size
            end = min(start + batch_size, n_texts)
            batch_texts = texts[start:end]

            try:
                emb = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as e:
                print(f"[embedder] ERROR encoding batch {bidx} ({start}:{end}): {e}")
                raise

            if emb.dtype != np.float32:
                emb = emb.astype(np.float32)

            mm[start:end, :] = emb
            mm.flush()

            # write progress
            try:
                with open(progress_path, 'w') as f:
                    f.write(str(bidx + 1))
            except Exception:
                pass

            # optional memory log
            if psutil is not None:
                p = psutil.Process()
                rss_mb = p.memory_info().rss / 1024 ** 2
                print(f"[embedder] Batch {bidx+1}/{n_batches} saved ({start}:{end}) — RSS: {rss_mb:.1f} MB")
            else:
                print(f"[embedder] Batch {bidx+1}/{n_batches} saved ({start}:{end})")

            # small sleep to yield IO
            time.sleep(0.01)

        # cleanup progress file
        try:
            if os.path.exists(progress_path):
                os.remove(progress_path)
        except Exception:
            pass

        # return a regular ndarray view (load into memory) — caller can keep memmap if desired
        return np.array(mm)

    def encode_scope(self, scope_text: str) -> np.ndarray:
        """
        Encode the journal's Aims & Scope text into a single 768d reference vector.

        This vector acts as the fixed 'centre of gravity' of the field.
        It is embedded ONCE and reused for all cosine similarity comparisons.

        Parameters
        ----------
        scope_text : str
            The official Aims & Scope passage.

        Returns
        -------
        np.ndarray, shape (768,)
        """
        return self.model.encode([scope_text])[0]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_embeddings(self, embeddings: np.ndarray, path: str = "embeddings.npy") -> None:
        """Save embedding matrix to a .npy file."""
        np.save(path, embeddings)
        print(f"[embedder] Embeddings saved to '{path}' — shape {embeddings.shape}.")

    def load_embeddings(self, path: str = "embeddings.npy") -> np.ndarray:
        """Load embedding matrix from a .npy file."""
        arr = np.load(path)
        print(f"[embedder] Embeddings loaded from '{path}' — shape {arr.shape}.")
        return arr

    # ------------------------------------------------------------------
    # encode or load
    # ------------------------------------------------------------------

    def encode_or_load(
        self,
        texts: List[str],
        path: str = "embeddings.npy",
        batch_size: int = 16,
        resume: bool = True,
        batch_save_interval: int = 10,
        max_batches: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load embeddings from disk if cache exists; otherwise encode and save.

        Parameters
        ----------
        texts : list[str]
            Input texts (used only when re-encoding).
        path : str
            Path to the .npy cache file.
        batch_size : int
            Encoding batch size.

        Returns
        -------
        np.ndarray, shape (N, 768)
        """
        if os.path.exists(path):
            # if a memmap exists we assume it's the final .npy; load it
            print(f"[embedder] Cache found at '{path}' — loading.")
            try:
                return self.load_embeddings(path)
            except Exception:
                # fallback to memmap read
                mm = np.memmap(path, dtype='float32', mode='r')
                arr = np.array(mm)
                print(f"[embedder] Loaded memmap cache — shape {arr.shape}")
                return arr

        print(f"[embedder] No cache found. Encoding {len(texts)} texts …")
        embeddings = self.encode(
            texts,
            batch_size=batch_size,
            path=path,
            resume=resume,
            batch_save_interval=batch_save_interval,
            max_batches=max_batches,
        )

        # ensure a standard .npy cache exists for downstream code
        try:
            self.save_embeddings(embeddings, path)
        except Exception:
            # if saving as .npy fails (e.g., because path is memmap), skip
            pass

        return embeddings
