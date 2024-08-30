import logging
import os
from datetime import datetime
from src.consts import DATA_FOLDER

logs_folder = DATA_FOLDER / "logs"
logs_folder.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
log_file_path = logs_folder / f"log_{timestamp}.log"

log_format = "%(asctime)s - %(message)s"

logging.basicConfig(
    format=log_format,
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path),
    ],
)

logger = logging.getLogger()

IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if IN_COLAB:
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(logging.Formatter(log_format))
    logger.addHandler(logging.FileHandler(log_file_path))
