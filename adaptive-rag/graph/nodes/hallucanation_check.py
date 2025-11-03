from graph.state import GraphState
from graph.schemas import BinaryScore
from langchain_core.messages import HumanMessage, SystemMessage
from utility.init_model import LLM
from pydantic import BaseModel, Field
from utility.formatter import format_docs
from graph.instructions import (
    HALLUCINATION_GRADER_INSTRUCTIONS,
    HALLUCINATION_GRADER_PROMPT,
)


def hallucination_checker(state: GraphState):

    print("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = HALLUCINATION_GRADER_PROMPT.format(
        documents=format_docs(documents), generation=generation.content
    )
    result: BinaryScore = LLM.with_structured_output(BinaryScore).invoke(
        [SystemMessage(content=HALLUCINATION_GRADER_INSTRUCTIONS)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )  # type: ignore[assignment]
    grade = result.binary_score

    if grade.lower() == "yes":
        print("---GRADE: NO HALLUCINATIONS DETECTED---")
        return "supported"
    elif state["loop_step"] <= max_retries:
        print("---GRADE: HALLUCINATIONS DETECTED, RETRYING---")
        return "not supported"
    else:
        return "max_retries_exceeded"
