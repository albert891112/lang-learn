from pydantic import BaseModel, Field
from graph.state import GraphState
from graph.instructions import DOC_GRADER_INSTRUCTION, DOC_GRADER_PROMPT
from utility.init_model import LLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate


class BinaryScore(BaseModel):
    """Structured output model for document relevance grading."""

    binary_score: str = Field(
        ..., description="Document relevance score ('yes' or 'no')"
    )


def grade_documents(state: GraphState):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    relevant_documents = []
    web_search = "No"
    for doc in documents:
        grade_prompt_content_formatted = DOC_GRADER_PROMPT.format(
            document=doc.page_content, question=question
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=DOC_GRADER_INSTRUCTION),
                HumanMessage(content=grade_prompt_content_formatted),
            ]
        )

        grader = prompt | LLM.with_structured_output(BinaryScore)

        grade_result: BinaryScore = grader.invoke({"document": doc.page_content, "question": question})  # type: ignore[assignment]

        if grade_result.binary_score.lower() == "yes":
            relevant_documents.append(doc)
            print("---GRADE: DOCUMENT RELEVANT---")
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": relevant_documents, "web_search": web_search}
