import pandas as pd
import numpy as np
from pathlib import Path

from src.post.consts import (
    EXPERIMENT_IDENTIFIERS,
    LOG_NORMAL_CONFIDENCE_TITLE,
    Z_SCORE_95_4,
)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the CSV file into a DataFrame and preprocess initial columns."""
    df = pd.read_csv(file_path)

    df["Configuration"] = (
        df["Device"] + " / " + df["Model"] + " / T=" + df["Temperature"].astype(str)
    )

    sentiment_mapping = {
        "very positive": 0,
        "positive": 0,
        "mixed positive": 0,
        "mixed negative": 1,
        "negative": 1,
        "very negative": 1,
        "no sentiment found": None,
    }

    df["Mapped Execution Sentiment"] = (
        df["Execution Sentiment"].str.lower().map(sentiment_mapping)
    )
    df["No Sentiment Found"] = df["Execution Sentiment"] == "No sentiment found"
    df["Correctly Classified"] = df["Label"] == df["Mapped Execution Sentiment"]

    return df


def calculate_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate precision metrics for each experiment configuration."""
    precision_df = (
        df.groupby(EXPERIMENT_IDENTIFIERS + ["Execution Sentiment"])
        .agg(
            Total_Cases=("Correctly Classified", "size"),
            Correct_Cases=("Correctly Classified", "sum"),
        )
        .reset_index()
    )
    precision_df["Precision"] = (
        precision_df["Correct_Cases"] / precision_df["Total_Cases"]
    )

    pivot_precision = (
        precision_df.pivot_table(
            index=EXPERIMENT_IDENTIFIERS,
            columns="Execution Sentiment",
            values="Precision",
        )
        .reset_index()
        .fillna(float("nan"))
    )

    pivot_precision.columns = [
        f"Precision({col})" if col not in EXPERIMENT_IDENTIFIERS else col
        for col in pivot_precision.columns
    ]
    return pivot_precision


def calculate_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate accuracy and 95.4% performance range for a log-normal distribution."""
    grouped = df.groupby(EXPERIMENT_IDENTIFIERS)

    accuracy_df = grouped.agg(
        Accuracy=("Correctly Classified", "mean"),
        No_Sentiment_Found=("No Sentiment Found", "sum"),
        performance_in_minutes=("Execution Time (minutes)", "mean"),
        performance_std_dev=("Execution Time (minutes)", "std"),
    ).reset_index()

    accuracy_df[LOG_NORMAL_CONFIDENCE_TITLE] = accuracy_df.apply(
        lambda row: calculate_log_normal_95_4_range(
            row["performance_in_minutes"], row["performance_std_dev"]
        ),
        axis=1,
    )

    return accuracy_df


def calculate_log_normal_95_4_range(mean: float, std_dev: float) -> list:
    """Calculate the 95.4% range for a log-normal distribution."""
    if pd.isna(std_dev) or std_dev == 0:
        return [mean, mean]

    log_mean = np.log(mean)
    log_std_dev = std_dev / mean
    min_log_bound = log_mean - Z_SCORE_95_4 * log_std_dev
    max_log_bound = log_mean + Z_SCORE_95_4 * log_std_dev
    min_bound = np.exp(min_log_bound)
    max_bound = np.exp(max_log_bound)

    return [min_bound, max_bound]
