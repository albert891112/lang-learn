from pydantic import Field, BaseModel
from utility.init_model import LLM
from graph.state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from graph.instructions import ROUTER_INSTRUCTION
from graph.schemas import Route


def router(state: GraphState):
    """
    Routes questions to either 'RAG' or 'websearch'.

    Args:
    state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")

    question = state["question"]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=ROUTER_INSTRUCTION),
            HumanMessage(content=question),
        ]
    )
    router = prompt | LLM.with_structured_output(Route)

    return {"route": router.invoke({"question": question})}
