import re
import time
from functools import partial
from pathlib import Path

from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts.prompt import PromptTemplate

from src.consts import ModelName, REPO_ROOT
from src.model.custom_slm import CustomSLM
from src.model.file_review_sentiment_output import (
    FilmReviewSentimentOutput,
    SentimentEnum,
)
from src.logger import logger


class CustomFilmReviewSentimentParser(BaseOutputParser):
    def parse(self, text: str) -> FilmReviewSentimentOutput:
        execution_sentiment = self._extract_sentiment_primary_strategy(text)

        if execution_sentiment == "No sentiment found":
            execution_sentiment = self._extract_sentiment_backup_strategy(text)

        return FilmReviewSentimentOutput(
            reasoning=text, execution_sentiment=execution_sentiment
        )

    def _extract_sentiment_primary_strategy(self, text: str) -> str:
        """Primary strategy: Extract sentiment using a direct regex search."""
        allowed_sentiments = [sentiment.value for sentiment in SentimentEnum]
        sentiment_regex = re.compile(
            rf"(?i)Execution\s*sentiment:\s*-?\s*({'|'.join(allowed_sentiments)})\b",
            re.DOTALL,
        )
        match = sentiment_regex.search(text)
        return match.group(1) if match else "No sentiment found"

    def _extract_sentiment_backup_strategy(self, text: str) -> str:
        """
        Backup strategy: Find the last occurrence of any allowed sentiment,
        with adjustments to prioritize specific sentiments over general ones.
        """
        allowed_sentiments = [sentiment.value for sentiment in SentimentEnum]
        sentiment_regex = re.compile(rf"(?i)({'|'.join(allowed_sentiments)})")
        matches = list(sentiment_regex.finditer(text))
        penalty = 5  # Penalty to adjust ranking for general sentiments

        if matches:
            ranked_matches = [
                (
                    (
                        match.start() - penalty
                        if match.group(1).lower() in ["positive", "negative"]
                        else match.start()
                    ),
                    match.start(),
                    match.group(1).capitalize(),
                )
                for match in matches
            ]

            # Sort matches: prioritize by adjusted index (highest) and then by original start index
            ranked_matches.sort(key=lambda x: (-x[0], -x[1]))

            return ranked_matches[0][2]

        return "No sentiment found"


def create_sentiment_parser(prompt_file: Path, slm: CustomSLM, temperature: float):
    sentiment_prompt_template = PromptTemplate(
        input_variables=["review"],
        template=prompt_file.read_text(),
    )
    slm_with_temperature = partial(slm.invoke, temperature=temperature)
    sentiment_parser = CustomFilmReviewSentimentParser()
    return sentiment_prompt_template | slm_with_temperature | sentiment_parser


if __name__ == "__main__":
    _sentiment_parser = create_sentiment_parser(
        prompt_file=REPO_ROOT / "prompts" / "003-weighted-sentiment-evaluation.txt",
        slm=CustomSLM(model_name=ModelName.QWEN),
        temperature=0.2,
    )

    review = "I absolutely loved the movie! The plot was engaging and the characters were relatable."

    start_time = time.time()
    result = _sentiment_parser.invoke({"review": review})
    logger.info(f"Execution time: {(time.time() - start_time) / 60} minutes")
    logger.info(result.dict())
