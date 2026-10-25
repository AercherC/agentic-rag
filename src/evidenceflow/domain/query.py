"""Domain Model for a user query request."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A query request carrying one non-empty user question."""

    query: str = Field(min_length=1, description="送入 RAG 流程的用户问题。")
