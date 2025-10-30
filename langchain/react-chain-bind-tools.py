from typing import List
from dotenv import load_dotenv
from langchain.agents import tool

from langchain_openai import AzureChatOpenAI

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, ToolMessage, ToolCall
import os


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by counting the number of characters."""

    print(f"get_text_length received input: {text=}")
    text = text.strip("\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found.")


tools = [get_text_length]


model = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0,
)

llm_with_tools = model.bind_tools(tools)

message = [HumanMessage(content="What is the length of the text 'Hello, world!'?")]

while True:
    ai_message = llm_with_tools.invoke(input=message)

    # if agent decides to call tool , execute them and return results.
    tool_calls: list[ToolCall] = ai_message.tool_calls
    if len(tool_calls) > 0:
        message.append(ai_message)
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_input = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            tool_to_use = find_tool_by_name(tools, tool_name)
            observation = tool_to_use.invoke(tool_input)
            print(f"Observation={observation}")
            message.append(
                ToolMessage(
                    tool_call_id=tool_call_id,
                    content=str(observation),
                )
            )
            continue

    print(f"AI Message: {ai_message.content}")
    break
