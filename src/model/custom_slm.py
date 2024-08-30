import os
import time
from typing import Optional, List, Any

from src.consts import ModelName, DeviceType

# These environment variables need to be set before importing `torch`
# PYTORCH_ENABLE_MPS_FALLBACK allows the CPU to take over when MPS operations are unsupported.
# PYTORCH_MPS_HIGH_WATERMARK_RATIO is set to 0 to maximize MPS memory usage without a predefined limit.
if DeviceType.detect() != DeviceType.GPU:  # Not running on Colab GPU
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"

import torch  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: E402
from langchain_core.language_models import LLM  # noqa: E402
from langchain_core.pydantic_v1 import Field  # noqa: E402

from src.logger import logger  # noqa: E402


class CustomSLM(LLM):
    model_name: ModelName = Field(default=ModelName.QWEN)
    device_type: DeviceType = Field(default=DeviceType.detect())
    custom_pipeline: Optional[Any] = Field(default=None)

    def __init__(
        self,
        model_name: ModelName,
        device_type: DeviceType = DeviceType.detect(),
    ):
        super().__init__()

        self._validate_device_availability(device_type)
        self.model_name = model_name
        self.device_type = device_type

        logger.info(f"Using {device_type.value} backend.")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name.value,
            torch_dtype=self._determine_data_type(device_type),
        ).to(torch.device(device_type.value))

        tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
        self.custom_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self._translate_device_type(device_type),
            torch_dtype=self._determine_data_type(device_type),
            do_sample=True,
        )

    @staticmethod
    def _validate_device_availability(device_type: DeviceType) -> None:
        if device_type == DeviceType.MPS and not torch.backends.mps.is_available():
            raise ValueError("MPS selected, but MPS is not available on this machine.")
        elif device_type == DeviceType.GPU and not torch.cuda.is_available():
            raise ValueError("GPU selected, but no GPU is available.")

    @staticmethod
    def _determine_data_type(device_type: DeviceType) -> torch.dtype:
        if device_type == DeviceType.GPU:
            return torch.float16
        return torch.float32

    @staticmethod
    def _translate_device_type(device_type: DeviceType) -> int | str:
        if device_type == DeviceType.GPU:
            return 0
        return device_type.value

    def _call(
        self,
        prompt: str,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        output = self.custom_pipeline(
            messages,
            max_new_tokens=512,
            return_full_text=False,
            temperature=temperature,
        )
        return output[0]["generated_text"]

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name.value}

    @property
    def _llm_type(self) -> str:
        return "custom_chat_llm"


if __name__ == "__main__":
    llm = CustomSLM(model_name=ModelName.QWEN)
    # llm = CustomSLM(model_name=ModelName.PHI)
    start_time = time.time()
    logger.info(llm.invoke("Can you help me solve 2x + 3 = 7 for x?"))
    logger.info(f"Execution time: {(time.time() - start_time) / 60} minutes")
