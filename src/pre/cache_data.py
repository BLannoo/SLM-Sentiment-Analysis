import pandas as pd
from datasets import load_dataset
import plac

from src.consts import DATA_FOLDER, SEED
from src.logger import logger


def sort_reviews_by_label_alternating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort reviews in alternating order by label to balance positive and negative samples.
    """
    df["rank"] = df.groupby("label").cumcount()
    df_sorted = df.sort_values(["rank", "label"]).reset_index(drop=True)
    return df_sorted[["label", "review"]]


@plac.annotations(subset_size=("The size of the subset to store", "option", "s", int))
def cache_subset(subset_size: int = 2000):
    """
    Cache a balanced subset of IMDB movie reviews to a CSV file.
    """
    try:
        dataset = load_dataset("ajaykarthick/imdb-movie-reviews", split="train")
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}")
        return

    subset = dataset.shuffle(seed=SEED)
    df = pd.DataFrame(subset)[["label", "review"]]
    df_sorted = sort_reviews_by_label_alternating(df).reset_index(drop=True)
    df_sorted.insert(0, "id", df_sorted.index + 1)

    subset_folder_name = DATA_FOLDER / "imdb_subset"
    subset_folder_name.mkdir(parents=True, exist_ok=True)
    subset_file_name = subset_folder_name / f"N={subset_size}.csv"

    df_sorted[:subset_size].to_csv(subset_file_name, index=False)
    logger.info(f"Subset of {subset_size} samples saved to {subset_file_name}")


if __name__ == "__main__":
    plac.call(cache_subset)
