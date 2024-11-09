from inferencing import action_call
from tools import tool_creator


if __name__ == "__main__":

    weather_tool = tool_creator.OpenAITool(
        "function",
        tool_creator.Function(
            "get_weather",
            "Get the current weather for a location",
            {
                "location": "string",
                "unit": "string",
            },
        ),
    )

    tools = [weather_tool]
    action_call.stream_output("What is the weather in New York", tools)

    action_call.stream_output()
    print("Done")
