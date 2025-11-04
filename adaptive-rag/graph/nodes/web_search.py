from langchain_core.documents import Document
from graph.state import GraphState
from langchain_tavily import TavilySearch

web_search_tool = TavilySearch()


def web_searcher(state: GraphState):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})["results"]
    print(f"Web search returned docs : {docs}")

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents}
