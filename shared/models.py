from enum import Enum
from typing import Optional
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM


class RoleEnum(str, Enum):
    Coach = "coach"
    Client = "client"


class Message(BaseModel):
    role: str  
    content: str


class AskRequest(BaseModel):
    coach_id: str
    client_id: str
    question: str
    history: Optional[list[Message]] = []  
    role: str 

