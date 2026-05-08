from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    role: str   
    content: str

class AskRequest(BaseModel):
    coach_id: str
    pitanje: str
    history: Optional[list[Message]] = []  
