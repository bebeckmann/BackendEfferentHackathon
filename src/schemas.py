from typing import Literal, Optional
from pydantic import BaseModel, Field


class EvidenceImage(BaseModel):
    url: str
    caption: str
    source: Optional[str] = None
    kind: Literal["uploaded", "generated", "retrieved"] = "uploaded"


class AgentResponse(BaseModel):
    answer: str
    images: list[EvidenceImage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str