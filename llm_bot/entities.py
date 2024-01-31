from pydantic import BaseModel

class ChatRequest(BaseModel):
    sender:str
    message:str
    api_list:list[dict]=[]