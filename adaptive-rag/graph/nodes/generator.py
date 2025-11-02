from graph.state import GraphState
from utility.init_model import LLM
from langchain_core.messages import HumanMessage
from graph.instructions import RAG_PROMPT
from utility.formatter import format_docs


def generate_answer(state: GraphState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    docs_content = format_docs(documents)
    loop_step = state["loop_step"]

    rag_prompt_formatted = RAG_PROMPT.format(context=docs_content, question=question)
    generation = LLM.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}
