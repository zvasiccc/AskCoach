from enum import Enum

from pydantic import BaseModel
from typing import Optional
from deepeval.models.base_model import DeepEvalBaseLLM
from shared.models import RoleEnum

class Message(BaseModel):
    role: str   
    content: str

class AskRequest(BaseModel):
    coach_id: str
    client_id: str
    question: str
    history: Optional[list[Message]] = []  
    role: str = RoleEnum.Coach
    
class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return (await self.load_model().ainvoke(prompt)).content

    def get_model_name(self):
        return self.model.model_name


class RoleEnum(str, Enum):
    Coach= "coach"
    Client = "client"