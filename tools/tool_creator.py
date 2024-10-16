import json
from typing import List, Dict


class PropertiesVariable:
    def __init__(self, name: str, type: str, required: bool):
        self.name = name
        self.type = type
        self.required = required

    def __str__(self) -> str:
        return f'"{self.name}": {{"type": "{self.type}"}}'


class Properties:
    def __init__(self, properties: List[PropertiesVariable]):
        self.properties = properties

    def __str__(self) -> str:
        properties_str = ", ".join([str(prop) for prop in self.properties])
        required_fields = [prop.name for prop in self.properties if prop.required]
        required_str = ", ".join([f'"{name}"' for name in required_fields])
        return f'"properties": {{{properties_str}}}, "required": [{required_str}]'


class Parameters:
    def __init__(self, param_type: str, properties: Properties):
        self.type = param_type
        self.properties = properties

    def __str__(self):
        return f'"type": "{self.type}", {str(self.properties)}'


class Function:
    def __init__(self, name: str, description: str, parameters: Parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def __str__(self):
        return f'"name": "{self.name}", "description": "{self.description}", "parameters": {{{str(self.parameters)}}}'


class OpenAITool:
    def __init__(self, type: str, function: Function):
        self.type = type
        self.function = function

    def __str__(self):
        return f'{{"type": "{self.type}", "function": {{{str(self.function)}}}}}'


if __name__ == "__main__":
    openai_tool = OpenAITool(
        "function",
        Function(
            "get_weather",
            "Get the current weather for a location",
            Parameters(
                "object",
                Properties(
                    [
                        PropertiesVariable("location", "string", True),
                        PropertiesVariable("unit", "string", True),
                    ]
                ),
            ),
        ),
    )
    openai_tool_json = str(openai_tool)
    print(openai_tool_json)
