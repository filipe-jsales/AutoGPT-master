import logging
from typing import NamedTuple
from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
import requests

logger = logging.getLogger(__name__)

class CustomModelCreationBlock(Block):
    class Input(BlockSchema):
        model_name: str = "filipee"
        modelfile: str = "FROM llama3.1\nSYSTEM You are a volleyball player."

    class Output(BlockSchema):
        status: str
        error: str

    def __init__(self):
        super().__init__(
            id="4d7b3eeb-5b89-4f8e-a0c6-8a735dd40f3a",
            description="Create a custom model on Ollama using the provided model name and modelfile.",
            categories={BlockCategory.AI},
            input_schema=CustomModelCreationBlock.Input,
            output_schema=CustomModelCreationBlock.Output,
            test_input={"model_name": "filipee", "modelfile": "FROM llama3.1\nSYSTEM You are a volleyball player."},
            test_output={"status": "success", "error": ""},
        )

    def create_ollama_model(self, model_name: str, modelfile: str) -> str:
        try:
            response = requests.post(
                "https://ollama.chargedcloud.com.br/api/create",
                json={"name": model_name, "modelfile": modelfile},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            json_response = response.json()
            return json_response.get("status", "unknown status")
        except Exception as e:
            logger.error(f"Error creating Ollama model: {e}")
            raise ValueError(f"Failed to create the model: {e}")

    def run(self, input_data: Input) -> BlockOutput:
        model_name = input_data.model_name
        modelfile = input_data.modelfile

        try:
            status = self.create_ollama_model(model_name, modelfile)
            yield "status", status
        except Exception as e:
            yield "error", str(e)
