"""
data_loader.py
--------------
Fetches journal articles from OpenAlex for a target journal while preserving the
existing dataset schema.

Output columns:
id | title | abstract | published | year | categories
"""
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List, Any


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    Reconstruct OpenAlex abstract text from abstract_inverted_index.
    """
    if not inverted_index:
        return None

    position_to_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_to_word[pos] = word

    return " ".join(position_to_word[i] for i in sorted(position_to_word))


class JournalLoader:
    """
    Loads articles from a specific journal using OpenAlex.


    Output DataFrame columns:
    id, title, abstract, published, year, categories
    """
    def __init__(
        self,
        journal_name: str = "Transactions of the Association for Computational Linguistics",
        issn: str = "2307-387X",
        limit: int = 5000,
        start_year: int = 2015,
        end_year: int = 2025,
        mailto: Optional[str] = None,
    ):
        self.journal_name = journal_name
        self.issn = issn
        self.limit = limit
        self.start_year = start_year
        self.end_year = end_year
        self.mailto = mailto
        self.base_url = "https://api.openalex.org"

    def _add_mailto(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.mailto:
            params["mailto"] = self.mailto
        return params

    def get_source_id(self) -> str:
        url = f"{self.base_url}/sources/issn:{self.issn}"
        response = requests.get(url, params=self._add_mailto({}), timeout=30)
        response.raise_for_status()

        source = response.json()
        source_id = source["id"].replace("https://openalex.org/", "")

        print(f"[journal_loader] Journal found: {source.get('display_name')}")
        print(f"[journal_loader] Source ID: {source_id}")

        return source_id

    def fetch_data(
        self,
        save_path: str = "dataset_nlp_cl.csv",
        force_refresh: bool = False,
    ) -> pd.DataFrame:

        if save_path and os.path.exists(save_path) and not force_refresh:
            print(f"[journal_loader] CSV found at '{save_path}' — loading from disk.")
            df = pd.read_csv(save_path, sep="|")
            print(
                f"[journal_loader] Loaded {len(df)} papers | "
                f"{df['year'].min()}–{df['year'].max()}"
            )
            return df

        source_id = self.get_source_id()

        filters = ",".join(
            [
                f"primary_location.source.id:{source_id}",
                f"from_publication_date:{self.start_year}-01-01",
                f"to_publication_date:{self.end_year}-12-31",
                "has_abstract:true",
                "type:article",
            ]
        )

        rows = []
        cursor = "*"

        print(
            f"[journal_loader] Fetching up to {self.limit} articles from "
            f"'{self.journal_name}' for {self.start_year}–{self.end_year}..."
        )

        with tqdm(total=self.limit, desc="Fetching journal articles", unit="article") as pbar:
            while cursor and len(rows) < self.limit:
                url = f"{self.base_url}/works"
                params = self._add_mailto(
                    {
                        "filter": filters,
                        "per-page": 200,
                        "cursor": cursor,
                        "sort": "publication_date:asc",
                        "select": ",".join(
                            [
                                "id",
                                "doi",
                                "title",
                                "publication_date",
                                "publication_year",
                                "abstract_inverted_index",
                                "primary_location",
                            ]
                        ),
                    }
                )

                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    if len(rows) >= self.limit:
                        break

                    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

                    if not abstract or len(abstract) <= 80:
                        continue

                    rows.append(
                        {
                            "id": work.get("doi") or work.get("id"),
                            "title": work.get("title"),
                            "abstract": abstract,
                            "published": work.get("publication_date"),
                            "year": work.get("publication_year"),
                            "categories": str([self.journal_name]),
                        }
                    )

                    pbar.update(1)

                cursor = data.get("meta", {}).get("next_cursor")
                time.sleep(0.5)

        df = pd.DataFrame(rows)

        df = df.dropna(subset=["id", "title", "abstract", "published", "year"])
        df = df[df["abstract"].str.len() > 80]
        df = df.drop_duplicates(subset=["id"])
        df = df[df["year"].between(self.start_year, self.end_year)]
        df = df.reset_index(drop=True)

        print(
            f"[journal_loader] Final dataset: {len(df)} articles | "
            f"{df['year'].min()}–{df['year'].max()}"
        )

        if save_path:
            df.to_csv(save_path, sep="|", index=False)
            print(f"[journal_loader] Dataset saved to '{save_path}'.")

        return df


if __name__ == "__main__":
    loader = JournalLoader(
        journal_name="Transactions of the Association for Computational Linguistics",
        issn="2307-387X",
        limit=5000,
        start_year=2015,
        end_year=2025,
    )

    df = loader.fetch_data(
        save_path="dataset_nlp_cl.csv",
        force_refresh=True,
    )

    print(df.head())
