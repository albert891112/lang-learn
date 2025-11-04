from pydantic import BaseModel, Field


class BinaryScore(BaseModel):
    """Structured output model for document relevance grading."""

    binary_score: str = Field(
        ..., description="Document relevance score ('yes' or 'no')"
    )


class AnswerScore(BinaryScore):
    """Structured output model for answer relevance grading."""

    explanation: str = Field(
        ..., description="Explanation for the answer relevance score"
    )


class Route(BaseModel):
    """Structured output model for routing."""

    dataSource: str = Field(
        ..., description="Data source to route to: 'vectorstore' or 'websearch'"
    )
