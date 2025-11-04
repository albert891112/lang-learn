from langgraph.graph import StateGraph
from graph.state import GraphState
from graph.nodes.grade_documents import doc_grader
from graph.nodes.check_hallucination import hallucination_checker
from graph.nodes.grade_answer import answer_grader
from graph.nodes.retrieve import retriever
from graph.nodes.generate_answer import answer_generator
from graph.nodes.route_question import router
from graph.nodes.web_search import web_searcher
from graph.nodes.condition import (
    websearch_or_vectorstore,
    is_doc_relevant,
    is_answer_useful,
    is_hallucination,
)
from langgraph.graph import END

workflow = StateGraph(GraphState)
workflow.add_node("route_question", router)
workflow.add_node("retrieve", retriever)
workflow.add_node("web_search", web_searcher)
workflow.add_node("grade_documents", doc_grader)
workflow.add_node("generate_answer", answer_generator)
workflow.add_node("grade_answer", answer_grader)
workflow.add_node("check_hallucination", hallucination_checker)

# Build graph
workflow.set_entry_point("route_question")

# conditional edges for routing to web_search or retrieve
workflow.add_conditional_edges("route_question", websearch_or_vectorstore)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("web_search", "generate_answer")

# conditional edges for document grading to generate_answer or web_search
workflow.add_conditional_edges("grade_documents", is_doc_relevant)
workflow.add_edge("generate_answer", "check_hallucination")

# conditional edges for hallucination check to grade_answer or generate_answer
workflow.add_conditional_edges("check_hallucination", is_hallucination)


workflow.add_conditional_edges("grade_answer", is_answer_useful)

# 編譯 workflow
RAG_GRAPH = workflow.compile()
