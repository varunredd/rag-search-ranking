from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    retriever: str = Field(default='tfidf')
    reranker: str = Field(default='none')
    top_k: int = Field(default=10, ge=3, le=50)


class RankedResult(BaseModel):
    doc_id: int
    text: str
    score: float
    rank: int
    previous_rank: Optional[int] = None
    rank_delta: Optional[int] = None
    is_ground_truth: bool = False


class SearchResponse(BaseModel):
    query: str
    retriever: str
    reranker: str
    dataset_mode: str
    note: str
    ground_truths: List[str]
    retrieval_results: List[RankedResult]
    reranked_results: List[RankedResult]
