"""
data_loader.py
--------------
Fetches computational linguistics papers from arXiv (cat:cs.CL).
No API key required. Uses the official `arxiv` Python library.

If the CSV already exists on disk, it is loaded directly — no re-fetching.
"""

import arxiv
import datetime
import pandas as pd
import os
from tqdm import tqdm
from typing import Optional


class ArxivLoader:
    """Loads cs.CL papers from arXiv and saves them to a pipe-separated CSV."""

    def __init__(
        self,
        query: str = "cat:cs.CL",
        limit: int = 5000,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        year_span: Optional[int] = None,
    ):
        self.query = query
        self.limit = limit
        current_year = datetime.datetime.now().year

        if year_span is not None:
            self.end_year = end_year if end_year is not None else current_year
            self.start_year = start_year if start_year is not None else self.end_year - year_span + 1
        else:
            self.start_year = start_year if start_year is not None else 2015
            self.end_year = end_year if end_year is not None else 2024
        self.year_span = year_span
        self.client = arxiv.Client(
            page_size=200,
            delay_seconds=3.0,
            num_retries=3,
        )

    def fetch_data(
        self,
        save_path: str = "dataset_nlp_cl.csv",
        sample_yearly: bool = True,
        sample_years: Optional[int] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch papers or load from CSV if it already exists.

        Parameters
        ----------
        save_path : str
            Path to the pipe-separated CSV cache file.
        sample_yearly : bool
            Whether to sample evenly across the year range.
        sample_years : Optional[int]
            If set, only sample this many years starting from start_year.
        force_refresh : bool
            If True, ignore the existing CSV cache and re-fetch from arXiv.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: id, title, abstract, published, year, categories.
        """
        if save_path and os.path.exists(save_path) and not force_refresh:
            print(f"[data_loader] CSV found at '{save_path}' — loading from disk.")
            df = pd.read_csv(save_path, sep="|")
            print(
                f"[data_loader] Loaded {len(df)} papers | "
                f"{df['year'].min()}–{df['year'].max()}"
            )
            return df
        elif save_path and os.path.exists(save_path) and force_refresh:
            print(f"[data_loader] CSV found at '{save_path}' but force_refresh=True, re-fetching.")

        print(
            f"[data_loader] Fetching up to {self.limit} papers "
            f"from arXiv (query='{self.query}') …"
        )

        results = []
        if sample_yearly:
            years = list(range(self.start_year, self.end_year + 1))
            if sample_years is not None:
                years = years[-sample_years:]
            base_limit, extra = divmod(self.limit, len(years))

            with tqdm(total=self.limit, desc="Fetching papers", unit="paper") as pbar:
                for index, year in enumerate(years):
                    if len(results) >= self.limit:
                        break

                    year_query = (
                        f"{self.query} AND submittedDate:[{year}01010000 TO {year}12312359]"
                    )
                    year_limit = min(
                        base_limit + (1 if index < extra else 0),
                        self.limit - len(results),
                    )
                    search = arxiv.Search(
                        query=year_query,
                        max_results=year_limit,
                        sort_by=arxiv.SortCriterion.SubmittedDate,
                        sort_order=arxiv.SortOrder.Ascending,
                    )

                    for r in self.client.results(search):
                        if len(results) >= self.limit:
                            break
                        try:
                            results.append(
                                {
                                    "id": r.entry_id,
                                    "title": r.title,
                                    "abstract": r.summary.replace("\n", " "),
                                    "published": r.published,
                                    "year": r.published.year,
                                    "categories": str(r.categories),
                                }
                            )
                            pbar.update(1)
                        except Exception as e:
                            print(f"[data_loader] Skipping a record due to error: {e}")
                            continue
        else:
            search = arxiv.Search(
                query=f"{self.query} AND submittedDate:[{self.start_year}01010000 TO {self.end_year}12312359]",
                max_results=self.limit,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Ascending,
            )

            with tqdm(total=self.limit, desc="Fetching papers", unit="paper") as pbar:
                for r in self.client.results(search):
                    try:
                        results.append(
                            {
                                "id": r.entry_id,
                                "title": r.title,
                                "abstract": r.summary.replace("\n", " "),
                                "published": r.published,
                                "year": r.published.year,
                                "categories": str(r.categories),
                            }
                        )
                        pbar.update(1)
                    except Exception as e:
                        print(f"[data_loader] Skipping a record due to error: {e}")
                        continue

        df = pd.DataFrame(results)

        # ---------- Cleaning ----------
        df = df[df["abstract"].str.len() > 80]            # drop near-empty abstracts
        df = df.drop_duplicates(subset=["id"])             # remove duplicate papers
        df = df[df["year"].between(self.start_year, self.end_year)]
        df = df.reset_index(drop=True)

        print(
            f"[data_loader] Fetched {len(df)} papers | "
            f"{df['year'].min()}–{df['year'].max()}"
        )

        if save_path:
            df.to_csv(save_path, sep="|", index=False)
            print(f"[data_loader] Dataset saved to '{save_path}'.")

        return df
