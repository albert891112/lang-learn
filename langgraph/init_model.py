import operator
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages

import os


load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Define the structure of the state using TypedDict
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define an extended state that supports message accumulation
class AllState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    IsComplete: bool
    todo: list[str]
