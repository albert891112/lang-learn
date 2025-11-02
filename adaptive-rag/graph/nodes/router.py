from pydantic import Field, BaseModel
from utility.init_model import LLM
from graph.state import GraphState
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from graph.instructions import ROUTER_INSTRUCTION


# Define the structured output model for routing
class Route(BaseModel):
    dataSource: str = Field(
        ..., description="Data source to route to: 'RAG' or 'websearch'"
    )


def route_question(state: GraphState):
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

    route_result: Route = router.invoke({"question": question})  # type: ignore[assignment]

    source = route_result.dataSource
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
