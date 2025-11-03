from pydantic import BaseModel, Field


class BinaryScore(BaseModel):
    """Structured output model for document relevance grading."""

    binary_score: str = Field(
        ..., description="Document relevance score ('yes' or 'no')"
    )
