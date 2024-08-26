from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class Input(BlockSchema):
    value: str  # The input value

class Output(BlockSchema):
    message: str  # The output message

class EchoBlock(Block):
    def __init__(self):
        super().__init__(
            id="f7a8b9c0-1d2e-3f4g-5h6i-7j8k9l0m1n2o",
            input_schema=Input,
            output_schema=Output,
            test_input={"value": "Hello, AutoGPT!"},
            test_output=("message", "Hello, AutoGPT!"),
            test_mock=None,  # No network calls to mock
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "message", input_data.value
