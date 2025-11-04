from graph.state import GraphState
from langchain_core.messages import HumanMessage, SystemMessage
from utility.init_model import LLM
from pydantic import BaseModel, Field
from utility.formatter import format_docs
from graph.schemas import AnswerScore
from graph.instructions import (
    ANSWER_GRADER_INSTRUCTIONS,
    ANSWER_GRADER_PROMPT,
)


def answer_grader(state: GraphState):
    print("---GRADE ANSWER---")
    question = state["question"]
    generation = state["generation"]

    answer_grader_prompt_formatted = ANSWER_GRADER_PROMPT.format(
        question=question, generation=generation.content
    )
    result: AnswerScore = LLM.with_structured_output(AnswerScore).invoke(
        [SystemMessage(content=ANSWER_GRADER_INSTRUCTIONS)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )  # type: ignore[assignment]
    return {"score": result}
