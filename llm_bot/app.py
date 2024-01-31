from fastapi import FastAPI, HTTPException
import uvicorn

from config import settings
from entities import ChatRequest
from chat_service import OpenAIChat


app=FastAPI()

chat_service=OpenAIChat(settings.OPENAI_API_KEY)

def call_chat(chat_request:ChatRequest):
    try:
        response=chat_service.chat(chat_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail="chat error")
    
    return response



if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8082,reload=False)