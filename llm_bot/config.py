from pydantic import BaseSettings

class Config(BaseSettings):


    OPENAI_API_KEY:str

    class Config:
        env_file=".env"
        env_file_encoding="utf-8"


settings=Config()