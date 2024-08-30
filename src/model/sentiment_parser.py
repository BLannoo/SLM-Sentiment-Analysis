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
        allowed_sentiments = [sentiment.value for sentiment in SentimentEnum]
        sentiment_regex = re.compile(
            rf"(?i)Execution\s*sentiment:\s*-?\s*({'|'.join(allowed_sentiments)})\b",
            re.DOTALL,
        )
        match = sentiment_regex.search(text)
        execution_sentiment = match.group(1) if match else "No sentiment found"

        return FilmReviewSentimentOutput(
            reasoning=text, execution_sentiment=execution_sentiment
        )


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
        prompt_file=REPO_ROOT / "prompts" / "001-naive.txt",
        slm=CustomSLM(model_name=ModelName.QWEN),
        temperature=0.2,
    )

    review = "I absolutely loved the movie! The plot was engaging and the characters were relatable."

    start_time = time.time()
    result = _sentiment_parser.invoke({"review": review})
    logger.info(f"Execution time: {(time.time() - start_time) / 60} minutes")
    logger.info(result.dict())
