"""用户查询请求的 Domain Model。"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """包含一个非空用户问题的查询请求。"""

    query: str = Field(min_length=1, description="送入 RAG 流程的用户问题。")
