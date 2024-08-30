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
    prompt_folder_name: str = "./prompts/",
    start_index: int = 0,
    end_index: int = 100,
    temperatures: str = None,
):
    """
    Run batch sentiment analysis on reviews using different model temperatures and prompt templates.

    Usage:
        You can use either positional or named arguments to run this script.

    Positional:
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py QWEN ./prompts/ 0 100

    Named:
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py \
            -m QWEN -p ./prompts/ -s 0 -e 100 -t 0.01,0.2,0.8

    Parameters:
        model_name (str): The model to use for sentiment analysis. Options are:
                    - 'QWEN': Qwen/Qwen2-1.5B-Instruct
                    - 'PHI': microsoft/Phi-3-mini-4k-instruct
                    Default is 'QWEN'.

        prompt_folder_name (str): Relative path to the folder containing prompt files to be used during model analysis.
                    Default is './prompts/'.

        start_index (int): The index of the first review to analyze.
                    Default is 0.

        end_index (int): The index of the last review to analyze.
                    Default is 100.

        temperatures (str): A comma-separated list of temperatures for the model runs (e.g., '0.01,0.2,0.8').
                    This controls the randomness of the model's outputs;
                    lower values make the output more deterministic.
                    Default is '0.01,0.2,0.8'.

    Functionality:
        This script performs sentiment analysis over a range of temperatures specified by the user
        or default (0.01, 0.2, 0.8) for each prompt template found in the specified prompt folder.
        The results of each analysis are saved to a CSV file, including columns for
        the temperature, prompt template, model, device, and run time.

    Examples:
        Positional:
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py QWEN ./prompts/ 0 100
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py PHI ./prompts/ 0 100

        Named:
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py -m PHI
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py -p ./prompts/
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py -s 100 -e 200
        PYTHONPATH=$(pwd):$PYTHONPATH python src/main.py -t 0.01,0.2,0.8
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
