import re
import csv
import plac
from pathlib import Path


def parse_logs_to_csv(log_file_path: Path, csv_output_path: Path):
    config_pattern = re.compile(
        r"Starting batch analysis with the following configuration:\n"
        r"Run Time: (.+)\n"
        r"Model: (.+)\n"
        r"Device: (.+)\n"
        r"Output File:",
        re.DOTALL,
    )
    review_text_pattern = re.compile(r"^Review Text: (.+)$")
    completed_analysis_pattern_part_1 = re.compile(
        r"Completed analysis for Review ID (\d+). Experiment \((\d+\/\d+)\): "
        r"Prompt (.+), Temperature ([\d.]+)$"
    )
    completed_analysis_pattern_part_2 = re.compile(
        r"^Execution Time: ([\d.]+) minutes, Label: (\d), Execution Sentiment: (.+?)$"
    )
    completed_analysis_pattern_part_3 = re.compile(r"^Reasoning: (.+)$")

    with open(log_file_path, "r", encoding="utf-8") as log_file, open(
        csv_output_path, "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
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

        log_content = log_file.read()

        config_match = config_pattern.search(log_content)
        if config_match:
            run_time = config_match.group(1).strip()
            model_name = config_match.group(2).strip()
            device = config_match.group(3).strip()
        else:
            raise ValueError("Configuration block not found in logs.")

        review_text = None

        for line in log_content.splitlines():
            review_text_match = review_text_pattern.search(line)
            if review_text_match:
                review_text = review_text_match.group(1)

            completed_analysis_match_part_1 = completed_analysis_pattern_part_1.search(
                line
            )
            if completed_analysis_match_part_1:
                review_id = completed_analysis_match_part_1.group(1)
                prompt_template = completed_analysis_match_part_1.group(3)
                temperature = completed_analysis_match_part_1.group(4)

            completed_analysis_match_part_2 = completed_analysis_pattern_part_2.search(
                line
            )
            if completed_analysis_match_part_2:
                execution_time = completed_analysis_match_part_2.group(1)
                label = completed_analysis_match_part_2.group(2)
                execution_sentiment = completed_analysis_match_part_2.group(3)

            completed_analysis_match_part_3 = completed_analysis_pattern_part_3.search(
                line
            )
            if completed_analysis_match_part_3:
                reasoning = completed_analysis_match_part_3.group(1)

                writer.writerow(
                    [
                        run_time,
                        review_id,
                        model_name,
                        temperature,
                        prompt_template,
                        device,
                        execution_time,
                        label,
                        execution_sentiment,
                        reasoning,
                        review_text,
                    ]
                )


@plac.annotations(
    log_file=("Path to the log file", "positional", None, Path),
    csv_output=("Path to the output CSV file", "positional", None, Path),
)
def main(log_file: Path, csv_output: Path):
    parse_logs_to_csv(log_file, csv_output)


if __name__ == "__main__":
    plac.call(main)
