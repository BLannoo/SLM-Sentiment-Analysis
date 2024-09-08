import textwrap

import pandas as pd

from src.post.consts import BLUE, RESET, EXPERIMENT_IDENTIFIERS, GREEN, RED


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
        original_df: pd.DataFrame,
        final_df: pd.DataFrame,
        general_metrics: GeneralMetrics,
        review_dfs: list,
        terminal_width: int = 280,
    ):
        self.original_df = original_df
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
        review_id = review_df["Review ID"].iloc[0]
        formatted_review = (
            f"\n{BLUE}Review {review_id}:\n{self._soft_wrap_text(review_text)}{RESET}\n"
        )

        for _, row in review_df.iterrows():
            identifier_values = ", ".join(
                [f"{id_name}: {row[id_name]}" for id_name in EXPERIMENT_IDENTIFIERS]
            )
            if row["Correctly Classified"]:
                correctness = f"{GREEN}Correctly Classified as {row['Execution Sentiment']}{RESET}"
            else:
                correctness = f"{RED}Incorrectly Classified as {row['Execution Sentiment']}{RESET}"
            if row["Label"] == 0:
                actual = "(actual: positive)"
            else:
                actual = "(actual: negative)"
            reasoning = self._soft_wrap_text(row["Reasoning"])
            formatted_review += f"\n{BLUE}Experiment - {identifier_values}, {correctness} {actual} {RESET}\n"
            formatted_review += f"Reasoning: {reasoning}\n"
        return formatted_review

    def _soft_wrap_text(self, text: str) -> str:
        return "\n".join(textwrap.wrap(text, self.terminal_width))
