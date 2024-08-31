import pandas as pd
import numpy as np
import textwrap
from pathlib import Path

from src.consts import DATA_FOLDER

# ANSI color codes for terminal output
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

EXPERIMENT_IDENTIFIERS = ["Model", "Temperature", "Prompt Template", "Device"]

LOG_NORMAL_CONFIDENCE_TITLE = "95.4% Performance Range (Log-Normal)"
Z_SCORE_95_4 = 2  # Z-score for 95.4% confidence in a normal distribution (±2 sigma)


class GeneralMetrics:
    def __init__(
        self,
        num_reviews: int,
        num_positive: int,
        num_negative: int,
        reviews_correct_all: int,
        reviews_correct_none: int,
    ):
        self.num_reviews = num_reviews
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.reviews_correct_all = reviews_correct_all
        self.reviews_correct_none = reviews_correct_none

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{BLUE}Number of Movie Reviews:{RESET} {self.num_reviews}",
                f"{BLUE}Number of Positive Targets:{RESET} {self.num_positive}",
                f"{BLUE}Number of Negative Targets:{RESET} {self.num_negative}",
                f"{BLUE}Reviews Correctly Classified by All Experiments:{RESET} {self.reviews_correct_all}",
                f"{BLUE}Reviews Correctly Classified by None of the Experiments:{RESET} {self.reviews_correct_none}",
            ]
        )


class Report:
    def __init__(
        self,
        final_df: pd.DataFrame,
        general_metrics: GeneralMetrics,
        review_dfs: list,
        terminal_width: int = 215,
    ):
        self.final_df = final_df
        self.general_metrics = general_metrics
        self.review_dfs = review_dfs
        self.terminal_width = terminal_width

    def __repr__(self) -> str:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", self.terminal_width)

        sorted_final_df = self.final_df.sort_values(by="Accuracy", ascending=False)
        report_str = f"{self.general_metrics}\n{sorted_final_df}\n"

        if self.review_dfs:
            report_str += f"{BLUE}\nReviews with Failed Experiments:{RESET}\n"
            report_str += self._format_failed_reviews()

        # Add general metrics both at beginning and end for easy reference
        report_str += f"\n{self.general_metrics}\n{sorted_final_df}\n"
        return report_str

    def _format_failed_reviews(self) -> str:
        formatted_reviews = ""
        for i, review_df in enumerate(self.review_dfs, 1):
            formatted_reviews += self._format_single_review(i, review_df)
        return formatted_reviews

    def _format_single_review(self, index: int, review_df: pd.DataFrame) -> str:
        review_text = review_df["Review"].iloc[0]
        formatted_review = f"\nReview {index}:\n{self._soft_wrap_text(review_text)}\n"

        for _, row in review_df.iterrows():
            identifier_values = ", ".join(
                [f"{id_name}: {row[id_name]}" for id_name in EXPERIMENT_IDENTIFIERS]
            )
            if row["Correctly Classified"]:
                correctness = f"{GREEN}Correctly Classified as {row['Execution Sentiment']}{RESET}"
            else:
                correctness = f"{RED}Incorrectly Classified as {row['Execution Sentiment']}{RESET}"
            reasoning = self._soft_wrap_text(row["Reasoning"])
            formatted_review += (
                f"\n{BLUE}Experiment - {identifier_values}, {correctness} {RESET}\n"
            )
            formatted_review += f"Reasoning: {reasoning}\n"
        return formatted_review

    def _soft_wrap_text(self, text: str) -> str:
        return "\n".join(textwrap.wrap(text, self.terminal_width))


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the CSV file into a DataFrame and preprocess initial columns."""
    df = pd.read_csv(file_path)
    sentiment_mapping = {
        "Very positive": 0,
        "Positive": 0,
        "Mixed positive": 0,
        "Mixed negative": 1,
        "Negative": 1,
        "Very negative": 1,
        "No sentiment found": None,
    }

    df["Mapped Execution Sentiment"] = df["Execution Sentiment"].map(sentiment_mapping)
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


def compute_general_metrics(df: pd.DataFrame) -> GeneralMetrics:
    """Compute general metrics for correctly classified reviews."""
    num_experiments = len(df.drop_duplicates(subset=EXPERIMENT_IDENTIFIERS))
    num_reviews = len(df) // num_experiments
    num_positive = (df["Label"] == 0).sum() // num_experiments
    num_negative = (df["Label"] == 1).sum() // num_experiments

    grouped_reviews = df.groupby("Review")
    reviews_correct_all = (
        grouped_reviews.filter(lambda x: x["Correctly Classified"].all()).shape[0]
        // num_experiments
    )
    reviews_correct_none = (
        grouped_reviews.filter(lambda x: not x["Correctly Classified"].any()).shape[0]
        // num_experiments
    )

    return GeneralMetrics(
        num_reviews,
        num_positive,
        num_negative,
        reviews_correct_all,
        reviews_correct_none,
    )


def extract_failed_reviews(
    df: pd.DataFrame, excluded_templates: list[str] = None
) -> list:
    """
    Extract reviews where at least one experiment failed, excluding specified prompt templates.

    Args:
    - df (pd.DataFrame): The input DataFrame containing reviews and experiment results.
    - excluded_templates (list[str]): List of prompt template names to exclude from the failure check.

    Returns:
    - List of DataFrames representing failed reviews.
    """
    excluded_templates = excluded_templates or []

    # Filter out the excluded prompt templates
    filtered_df = df[~df["Prompt Template"].isin(excluded_templates)]

    # Identify failed reviews based on the filtered data
    failed_reviews = filtered_df.groupby("Review").filter(
        lambda x: not x["Correctly Classified"].all()
    )

    return [group for _, group in failed_reviews.groupby("Review")]


def perform_analysis(file_path: Path, excluded_templates: list[str] = None) -> Report:
    df = load_data(file_path)
    precision_df = calculate_precision(df)
    accuracy_df = calculate_accuracy(df)
    final_df = pd.merge(accuracy_df, precision_df, on=EXPERIMENT_IDENTIFIERS)

    general_metrics = compute_general_metrics(df)
    review_dfs = extract_failed_reviews(df, excluded_templates=excluded_templates)

    return Report(
        final_df,
        general_metrics,
        review_dfs,
    )


if __name__ == "__main__":
    print(
        perform_analysis(
            file_path=DATA_FOLDER
            / "collab"
            / "start=2024-08-31T_merged-QWEN-T=02-index=1-100.csv",
            # excluded_templates=["006-priming-with-key-word"],
        )
    )
