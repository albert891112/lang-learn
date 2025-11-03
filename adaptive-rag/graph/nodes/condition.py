from graph.state import GraphState


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


def is_answer_useful(state: GraphState) -> str:
    """
    Determines whether to end the workflow or re-attempt web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED ANSWER---")
    web_search = state["web_search"]

    if web_search:
        # The answer is not useful, we will re-attempt web search
        print("---DECISION: ANSWER NOT USEFUL, RE-ATTEMPT WEB SEARCH---")
        return "web_search"
    else:
        # The answer is useful, we will end the workflow
        print("---DECISION: ANSWER USEFUL, END WORKFLOW---")
        return "end"
