import logging
from typing import NamedTuple
from enum import Enum

import requests
from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema

logger = logging.getLogger(__name__)

class LlmModel(str, Enum):
    CUSTOM_OLLAMA_MODEL = "Llama 3"

class ModelMetadata(NamedTuple):
    provider: str
    context_window: int

MODEL_METADATA = {
    LlmModel.CUSTOM_OLLAMA_MODEL: ModelMetadata("ollama", 8192),
}

class CustomLlmCallBlock(Block):
    class Input(BlockSchema):
        prompt: str
        model: LlmModel = LlmModel.CUSTOM_OLLAMA_MODEL
        sys_prompt: str = ""
        retry: int = 3

    class Output(BlockSchema):
        response: str
        error: str

    def __init__(self):
        super().__init__(
            id="3f7b2dcb-4a78-4e3f-b0f1-88132e1b8df9",
            description="Call a custom Large Language Model (LLM) hosted on a specific URL to generate a response based on the given prompt.",
            categories={BlockCategory.AI},
            input_schema=CustomLlmCallBlock.Input,
            output_schema=CustomLlmCallBlock.Output,
            test_input={"prompt": "User prompt"},
            test_output={"response": "Response text", "error": ""},
        )

    def call_ollama_llm(self, prompt: str) -> str:
        try:
            response = requests.post(
                "https://ollama.chargedcloud.com.br/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            json_response = response.json()
            if json_response.get("done"):
                return json_response.get("response", "")
            else:
                raise ValueError("Model did not finish processing.")
        except Exception as e:
            logger.error(f"Error calling Ollama LLM: {e}")
            raise ValueError(f"Failed to get a response from the LLM: {e}")

    def run(self, input_data: Input) -> BlockOutput:
        prompt = input_data.sys_prompt + "\n" + input_data.prompt if input_data.sys_prompt else input_data.prompt

        for _ in range(input_data.retry):
            try:
                response_text = self.call_ollama_llm(prompt)
                yield "response", response_text
                return
            except Exception as e:
                error_message = f"Attempt failed: {e}"
                logger.warning(error_message)

        yield "error", "Failed to generate a response after multiple attempts."
