import os
import sys
from enum import Enum
from pathlib import Path

SRC_FOLDER = Path(__file__).parent
REPO_ROOT = SRC_FOLDER.parent
DATA_FOLDER = REPO_ROOT / "data"


class ModelName(str, Enum):
    QWEN = "Qwen/Qwen2-1.5B-Instruct"
    PHI = "microsoft/Phi-3-mini-4k-instruct"


class DeviceType(str, Enum):
    MPS = "mps"
    CPU = "cpu"
    GPU = "cuda"

    @staticmethod
    def detect() -> "DeviceType":
        if sys.platform == "darwin":
            return DeviceType.MPS
        if "COLAB_GPU" in os.environ:
            colab_gpu_value = os.environ["COLAB_GPU"]
            if colab_gpu_value is not None and colab_gpu_value != "":
                return DeviceType.GPU
        return DeviceType.CPU


SEED = 42
