# schemas.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Source(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = None 
    page: Optional[int] = None
    score: Optional[float] = None
    text: Optional[str] = None
    content: Optional[str] = None
    snippet: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "ignore" 

class ChatRequest(BaseModel):
    message: str
    scope: Optional[str] = "uploaded"
    debug: bool = False

    class Config:
        extra = "ignore"

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    class Config:
        extra = "ignore"
