from graph.state import GraphState
from langgraph.graph import END


def websearch_or_vectorstore(state: GraphState) -> str:
    """
    Determines whether to run web search or retrieve from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ROUTE QUESTION---")
    source = state["route"].dataSource
    if source == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"


def is_doc_relevant(state: GraphState) -> str:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate_answer"


def is_hallucination(state: GraphState) -> str:
    """
    Determines whether to end the workflow or re-attempt answer generation

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS HALLUCINATIONS---")
    grade = state["score"].binary_score
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided
    loop_step = state["loop_step"]

    if "yes" in grade.lower():
        print("---GRADE: NO HALLUCINATIONS DETECTED---")
        # supported
        return "grade_answer"
    elif loop_step <= max_retries:
        print("---GRADE: HALLUCINATIONS DETECTED, RETRYING---")
        # not supported
        return "generate_answer"
    else:
        # max retries exceeded
        return END


def is_answer_useful(state: GraphState) -> str:
    """
    Determines whether to end the workflow or re-attempt web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS HALLUCINATIONS---")
    grade = state["score"].binary_score
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided
    loop_step = state["loop_step"]

    print("---ANSWER USEFULNESS GRADE:", grade, "---")

    if "yes" in grade.lower():
        print("---GRADE: ANSWER MEETS CRITERIA---")
        # useful
        return END
    elif loop_step <= max_retries:
        print("---GRADE: ANSWER DOES NOT MEET CRITERIA, RETRYING---")
        return "web_search"
    else:
        print("---GRADE: ANSWER DOES NOT MEET CRITERIA---")
        # max retries exceeded
        return END
