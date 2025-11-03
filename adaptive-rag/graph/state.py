from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from graph.schemas import BinaryScore
import operator


class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.

    Attributes:
    question (str): User question
    generation (str): LLM generation
    web_search (bool): Binary decision to run web search
    max_retries (int): Max number of retries for answer generation
    from_answer (bool): Flag indicating if the flow is from answers node
    answer (str): answer from answers node
    loop_step (int): Loop step counter
    documents (List[str]): List of retrieved documents
    """

    question: str  # User question
    score: BinaryScore
    generation: AIMessage  # LLM generation
    web_search: bool  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]
    documents: List[Document]  # List of retrieved documents
