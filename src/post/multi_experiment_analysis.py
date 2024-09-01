import pandas as pd
from pathlib import Path

from src.consts import DATA_FOLDER
from src.post.consts import EXPERIMENT_IDENTIFIERS
from src.post.data_post_processing import (
    load_data,
    calculate_accuracy,
    calculate_precision,
)
from src.post.plotting import render_as_1_figure
from src.post.report import Report, GeneralMetrics


def perform_analysis(
    file_path: Path, excluded_templates: list[str] = None, keep_only_gpu: bool = True
) -> Report:
    df = load_data(file_path)

    if keep_only_gpu:
        # GPU's performance is so much better, that it becomes near 0 in comparison to MPS or CPU
        df = df.loc[lambda _df: _df["Device"] == "GPU"]

    accuracy_df = calculate_accuracy(df)
    precision_df = calculate_precision(df)
    final_df = pd.merge(accuracy_df, precision_df, on=EXPERIMENT_IDENTIFIERS)

    general_metrics = compute_general_metrics(df)
    review_dfs = extract_failed_reviews(df, excluded_templates=excluded_templates)

    return Report(
        original_df=df,
        final_df=final_df,
        general_metrics=general_metrics,
        review_dfs=review_dfs,
    )


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


if __name__ == "__main__":
    _report = perform_analysis(
        file_path=DATA_FOLDER / "gold" / "ALL-index=1-100.csv",
        keep_only_gpu=True,
    )
    print(_report)

    render_as_1_figure(_report)
    # render_as_multiple_figures(_report)
