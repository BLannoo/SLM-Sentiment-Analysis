import os
import re
import pandas as pd

from src.consts import DATA_FOLDER
from src.logger import logger


def load_subset(start_index: int = 0, end_index: int = 100) -> pd.DataFrame:
    imdb_subsets_path = DATA_FOLDER / "imdb_subset"
    pattern = re.compile(r"N=(\d+)\.csv")

    subset_files = [
        (int(match.group(1)), imdb_subsets_path / f)
        for f in os.listdir(imdb_subsets_path)
        if (match := pattern.match(f)) and int(match.group(1)) > end_index
    ]

    if not subset_files:
        raise ValueError(f"No subset found with N > {end_index}.")

    smallest_subset_file = min(subset_files, key=lambda x: x[0])[1]
    df = pd.read_csv(smallest_subset_file)

    total_rows = len(df)

    if start_index < 0 or end_index > total_rows or start_index >= end_index:
        raise ValueError(
            f"Invalid indices: start_index = {start_index}, end_index = {end_index}, total_rows = {total_rows}"
        )

    logger.info(
        f"Loaded subset from {smallest_subset_file} with rows from {start_index} to {end_index}"
    )
    return df.iloc[start_index:end_index]


if __name__ == "__main__":
    data = load_subset(start_index=100, end_index=200)
    logger.info(type(data))
    logger.info(data.head())
