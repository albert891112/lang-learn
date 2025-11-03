from langgraph.graph import StateGraph
from graph.state import GraphState
from graph.nodes.grade_documents import doc_grader
from graph.nodes.hallucanation_check import hallucination_checker
from graph.nodes.grade_answer import answer_grader
from graph.nodes.retrieve import retriever
from graph.nodes.generate_answer import answer_generator
from graph.nodes.route_question import router
from graph.nodes.web_search import web_searcher
from graph.nodes.is_doc_relevant import is_doc_relevant
from langgraph.graph import END

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retriever)
workflow.add_node("web_search", web_searcher)
workflow.add_node("grade_documents", doc_grader)
workflow.add_node("generate_answer", answer_generator)
workflow.add_node("grade_answer", answer_grader)


# Build graph
workflow.set_conditional_entry_point(
    router,
    {
        "web_search": "web_search",
        "vector_store": "retrieve",
    },
)
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    is_doc_relevant,
    {
        "web_search": "web_search",
        "generate_answer": "generate_answer",
    },
)
workflow.add_conditional_edges(
    "generate_answer",
    hallucination_checker,
    {
        "supported": "grade_answer",
        "not supported": "generate_answer",
        "max_retries_exceeded": END,
    },
)
workflow.add_conditional_edges(
    "grade_answer",
    answer_grader,
    {"useful": END, "not useful": "web_search", "max_retries_exceeded": END},
)

# 編譯 workflow
app = workflow.compile()
