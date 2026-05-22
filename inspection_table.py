import pandas as pd

from alignment import build_alignment_inspection_table
from visualizer import plot_inspection_table


if __name__ == "__main__":
    df = pd.read_csv("dataset_nlp_cl.csv", sep="|")
    if "alignment_score" not in df.columns:
        raise SystemExit(
            "dataset_nlp_cl.csv must include an 'alignment_score' column. "
            "Run the main pipeline or supply a scored dataframe before using this script."
        )
    inspection_table = build_alignment_inspection_table(df)
    print(inspection_table.to_string(index=False))
    plot_inspection_table(
        inspection_table,
        filename="alignment_inspection_table.png",
        save=True,
    )
    print("Saved alignment_inspection_table.png")
