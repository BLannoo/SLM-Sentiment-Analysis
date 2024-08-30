import time
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from src.consts import DATA_FOLDER
from src.model.custom_slm import CustomSLM
from src.model.file_review_sentiment_output import FilmReviewSentimentOutput
from src.model.sentiment_parser import create_sentiment_parser
from src.logger import logger


@dataclass
class ReviewData:
    review_index: int
    review_text: str
    actual_label: int
    review_id: int


@dataclass
class Experiment:
    prompt_file: Path
    temperature: float
    slm: CustomSLM
    run_time_identifier: str


def analyze_and_save_to_csv(
    data: pd.DataFrame,
    slm: CustomSLM,
    temperatures: list[float],
    prompt_folder: Path,
):
    run_time_identifier = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    output_folder = DATA_FOLDER / "sentiment_analysis_results"
    output_file = output_folder / f"start={run_time_identifier}.csv"

    logger.info(
        f"Starting batch analysis with the following configuration:\n"
        f"Run Time: {run_time_identifier}\n"
        f"Model: {slm.model_name.name}\n"
        f"Device: {slm.device_type.name}\n"
        f"Output File: {output_file}\n"
        f"Temperatures: {temperatures}\n"
        f"Prompt Files: {[p.stem for p in prompt_folder.glob('*.txt')]}"
    )
    prompt_files = list(prompt_folder.glob("*.txt"))
    experiments = [
        Experiment(
            prompt_file=prompt_file,
            temperature=temp,
            slm=slm,
            run_time_identifier=run_time_identifier,
        )
        for prompt_file in prompt_files
        for temp in temperatures
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Run Time",
                "Review ID",
                "Model",
                "Temperature",
                "Prompt Template",
                "Device",
                "Execution Time (minutes)",
                "Label",
                "Execution Sentiment",
                "Reasoning",
                "Review",
            ]
        )

        for i, review_text in enumerate(data["review"]):
            review_data = ReviewData(
                review_index=i,
                review_text=review_text,
                actual_label=data["label"].iloc[i],
                review_id=data["id"].iloc[i],
            )
            analyze_review(
                review_data,
                experiments,
                writer,
                file,
                total_reviews=len(data),
            )

    logger.info("Finished analysis for all reviews.")


def analyze_review(
    review_data: ReviewData,
    experiments: list[Experiment],
    writer: csv.writer,
    file: TextIO,
    total_reviews: int,
):
    logger.info(
        f"Starting analysis for Review ID {review_data.review_id} ({review_data.review_index + 1}/{total_reviews}): "
        f"Label = {review_data.actual_label} (0=Positive; 1=Negative)"
    )

    total_experiments = len(experiments)
    for exp_index, experiment in enumerate(experiments):
        logger.info(
            f"Starting experiment {exp_index + 1}/{total_experiments} with Prompt: {experiment.prompt_file.stem}, "
            f"Temperature: {experiment.temperature}, Model: {experiment.slm.model_name.name}, "
            f"Device: {experiment.slm.device_type.name}"
        )
        process_sentiment_analysis(review_data, experiment, writer, file, total_reviews)


def process_sentiment_analysis(
    review_data: ReviewData,
    experiment: Experiment,
    writer: csv.writer,
    file: TextIO,
    total_reviews: int,
):
    sentiment_parser = create_sentiment_parser(
        prompt_file=experiment.prompt_file,
        slm=experiment.slm,
        temperature=experiment.temperature,
    )

    start_time = time.time()
    sentiment_data: FilmReviewSentimentOutput = sentiment_parser.invoke(
        {"review": review_data.review_text}
    )
    execution_time = (time.time() - start_time) / 60

    logger.info(
        f"Completed analysis for Review ID {review_data.review_id} ({review_data.review_index + 1}/{total_reviews}): "
        f"Prompt {experiment.prompt_file.stem}, Temperature {experiment.temperature}, "
        f"Execution Time: {execution_time:.2f} minutes, Label: {review_data.actual_label}, "
        f"Execution Sentiment: {sentiment_data.execution_sentiment}, "
        f"Reasoning: {sentiment_data.escaped_reasoning()}, "
        f"Review: {review_data.review_text}"
    )

    writer.writerow(
        [
            experiment.run_time_identifier,
            review_data.review_id,
            experiment.slm.model_name.name,
            experiment.temperature,
            experiment.prompt_file.stem,
            experiment.slm.device_type.name,
            f"{execution_time:.2f}",
            review_data.actual_label,
            sentiment_data.execution_sentiment,
            sentiment_data.escaped_reasoning(),
            review_data.review_text,
        ]
    )
    file.flush()
