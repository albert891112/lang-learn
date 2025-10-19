from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_openai import AzureChatOpenAI
import os

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by counting the number of characters."""
    text = text.strip("\n").strip('"')
    return len(text)


if __name__ == "__main__":
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    model = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        temperature=0,
        stop=["\nObservation:"],
    )

    agent = prompt | model

    res = agent.invoke(
        input={"input": 'What is the length of the text "Hello, world!"?'}
    )
    print(res)
