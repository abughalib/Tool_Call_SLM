from llama_cpp import Llama
from typing import List
import json

llm = Llama(model_path="./XLAM_Model/xlam_1b_f16.gguf", n_ctx=1024)

task_instruction = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.
""".strip()

format_instruction = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make tool_calls an empty list '[]'.
```
{
    "tool_calls": [
    {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
    ... (more tool calls as required)
    ]
}
```
""".strip()


# Define the input query and available tools
def convert_to_xlam_tool(tools):
    """"""
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {
                k: v for k, v in tools["parameters"].get("properties", {}).items()
            },
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    else:
        return tools


def build_prompt(
    task_instruction: str, format_instruction: str, tools: list, query: str
):
    xlam_format_tools = convert_to_xlam_tool(tools)
    content = build_prompt(
        task_instruction, format_instruction, xlam_format_tools, query
    )

    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_format_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt


def stream_output(query: str, tools: List[str]):
    # Build the input and start the inference

    content = build_prompt(task_instruction, format_instruction, tools, query)
    print(content)

    output = llm(
        json.dumps(content),
        temperature=1,
        seed=12345,
        max_tokens=8196,
        stop=["<|EOT|>"],
        echo=False,
    )

    output_json = json.dumps(output)
    json_output = json.loads(output_json)

    print(f"\n\nResponse:\n{json_output}")
