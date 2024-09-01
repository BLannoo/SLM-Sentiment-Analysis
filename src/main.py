import plac

from src.consts import ModelName, REPO_ROOT, DEFAULT_TEMPERATURES
from src.model.custom_slm import CustomSLM
from src.pre.load_cached_data import load_subset
from src.logger import logger
from src.model.sentiment_analysis_batch_runner import analyze_and_save_to_csv


@plac.annotations(
    model_name=("Model to use ['QWEN', 'PHI']", "option", "m", str),
    prompt_folder_name=("Relative path to the prompt folder", "option", "p", str),
    start_index=("Index of the first review to analyze", "option", "s", int),
    end_index=("Index of the last review to analyze", "option", "e", int),
    temperatures=(
        "Comma-separated list of temperatures (e.g., '0.01,0.2,0.8')",
        "option",
        "t",
        str,
    ),
)
def main(
    model_name: str = ModelName.QWEN.name,
    prompt_folder_name: str = "./prompts/strategies/",
    start_index: int = 0,
    end_index: int = 100,
    temperatures: str = None,
):
    """
    Run batch sentiment analysis on reviews using different models, temperatures, and prompt templates.

    For full usage instructions, parameters, and examples, see `docs/004_running_experiments.md`.

    Usage examples:
        PYTHONPATH=$(pwd):$PYTHONPATH caffeinate python src/main.py -m QWEN -p ./prompts/strategies/
        PYTHONPATH=$(pwd):$PYTHONPATH caffeinate python src/main.py -s 0 -e 100 -t 0.2,0.8

    Parameters:
        -m model_name (str): Model to use ('QWEN' or 'PHI').
        -p prompt_folder_name (str): Path to the folder containing prompt files.
        -s start_index (int): Index of the first review to analyze.
        -e end_index (int): Index of the last review to analyze.
        -t temperatures (str): Comma-separated list of temperatures (e.g., '0.2,0.8').

    Refer to `docs/004_running_experiments.md` for more details.
    """

    if temperatures is None:
        temperatures_list = DEFAULT_TEMPERATURES
    else:
        try:
            temperatures_list = [
                float(temp.strip()) for temp in temperatures.split(",")
            ]
        except ValueError:
            logger.error(
                "Invalid format for temperatures. Must be a comma-separated list of numbers."
            )
            return

    logger.info("Starting model analysis process.")

    try:
        model_name_enum = ModelName[model_name.upper()]
    except KeyError:
        logger.error(
            f"Invalid model_name: {model_name}. Must be one of {list(ModelName)}."
        )
        return

    prompt_folder = REPO_ROOT / prompt_folder_name
    if not prompt_folder.exists() or not prompt_folder.is_dir():
        logger.error(
            f"Invalid prompt folder path: {prompt_folder}. Please check the path."
        )
        return

    analyze_and_save_to_csv(
        data=load_subset(start_index=start_index, end_index=end_index),
        slm=CustomSLM(model_name=model_name_enum),
        temperatures=temperatures_list,
        prompt_folder=prompt_folder,
    )

    logger.info("Sentiment analysis process completed.")


if __name__ == "__main__":
    plac.call(main)
