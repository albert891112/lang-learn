from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.

    Attributes:
    question (str): User question
    generation (str): LLM generation
    web_search (str): Binary decision to run web search
    max_retries (int): Max number of retries for answer generation
    answers (int): Number of answers generated
    loop_step (int): Loop step counter
    documents (List[str]): List of retrieved documents
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[Document]  # List of retrieved documents
