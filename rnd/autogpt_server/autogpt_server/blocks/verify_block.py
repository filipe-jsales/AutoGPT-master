import logging
from typing import NamedTuple
from enum import Enum

import requests
from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema

logger = logging.getLogger(__name__)

# Adicionando os modelos disponíveis
class LlmModel(str, Enum):
    FILIPEE = "filipee:latest"
    MARIO = "mario:latest"
    LUIGI = "luigi:latest"
    LLAMA_3_1 = "llama3.1:latest"
    GAMER = 'gamer:latest'

class ModelMetadata(NamedTuple):
    provider: str
    context_window: int

MODEL_METADATA = {
    LlmModel.FILIPEE: ModelMetadata("ollama", 8192),
    LlmModel.MARIO: ModelMetadata("ollama", 8192),
    LlmModel.LUIGI: ModelMetadata("ollama", 8192),
    LlmModel.LLAMA_3_1: ModelMetadata("ollama", 8192),
    LlmModel.GAMER: ModelMetadata("ollama", 8192),
}

class ClassificationLlmCallBlock(Block):
    class Input(BlockSchema):
        prompt: str
        model: LlmModel = LlmModel.LLAMA_3_1  # Modelo padrão
        sys_prompt: str = ""
        retry: int = 3

    class Output(BlockSchema):
        positive_response: str
        negative_response: str
        error: str

    def __init__(self):
        super().__init__(
            id="a892b8d9-343e-4e9c-9c1e-75f8efcf1bfa",
            description="Call a custom LLM to classify the response as positive or negative and route accordingly.",
            categories={BlockCategory.AI},
            input_schema=ClassificationLlmCallBlock.Input,
            output_schema=ClassificationLlmCallBlock.Output,
            test_input={"prompt": "User prompt"},
            test_output={"positive_response": "", "negative_response": "", "error": ""},
        )

    def call_ollama_llm(self, prompt: str, model: str) -> str:
        try:
            response = requests.post(
                "https://ollama.chargedcloud.com.br/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
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

    def classify_response(self, response_text: str) -> bool:
        # Aqui você pode adicionar a lógica para determinar se a resposta é positiva ou negativa.
        # Por exemplo, vamos supor que a classificação seja baseada em uma palavra-chave:
        return "positive" in response_text.lower()

    def run(self, input_data: Input) -> BlockOutput:
        prompt = input_data.sys_prompt + "\n" + input_data.prompt if input_data.sys_prompt else input_data.prompt

        for _ in range(input_data.retry):
            try:
                response_text = self.call_ollama_llm(prompt, input_data.model)
                if self.classify_response(response_text):
                    yield "positive_response", response_text
                else:
                    yield "negative_response", response_text
                return
            except Exception as e:
                error_message = f"Attempt failed: {e}"
                logger.warning(error_message)

        yield "error", "Failed to generate a response after multiple attempts."
