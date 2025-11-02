from graph.nodes.router import route_question
from graph.nodes.retriever import retrieve
from graph.state import GraphState


if __name__ == "__main__":

    state = GraphState(
        {
            "question": "What is the capital of France?",
            "generation": "",
            "web_search": "No",
            "max_retries": 3,
            "answers": 0,
            "loop_step": 0,
            "documents": [],
        }
    )
    print(route_question(state))
    print(retrieve(state))
