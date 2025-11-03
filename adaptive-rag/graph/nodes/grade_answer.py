from graph.state import GraphState
from langchain_core.messages import HumanMessage, SystemMessage
from utility.init_model import LLM
from pydantic import BaseModel, Field
from utility.formatter import format_docs
from graph.schemas import BinaryScore
from graph.instructions import (
    ANSWER_GRADER_INSTRUCTIONS,
    ANSWER_GRADER_PROMPT,
)


class AnswerScore(BaseModel):
    """Structured output model for answer grading."""

    binary_score: str = Field(..., description="Answer relevance score ('yes' or 'no')")


def answer_grader(state: GraphState):
    print("---GRADE ANSWER---")
    question = state["question"]
    generation = state["generation"]
    loop_step = state["loop_step"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    answer_grader_prompt_formatted = ANSWER_GRADER_PROMPT.format(
        question=question, generation=generation.content
    )
    result: BinaryScore = LLM.with_structured_output(BinaryScore).invoke(
        [SystemMessage(content=ANSWER_GRADER_INSTRUCTIONS)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )  # type: ignore[assignment]
    grade = result.binary_score

    if grade.lower() == "yes":
        print("---GRADE: ANSWER MEETS CRITERIA---")
        return "useful"
    elif loop_step <= max_retries:
        print("---GRADE: ANSWER DOES NOT MEET CRITERIA, RETRYING---")
        return "not useful"
    else:
        print("---GRADE: ANSWER DOES NOT MEET CRITERIA---")
        return "max_retries_exceeded"
