from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_QA_URL: str = "http://qa-model:8001"
    MODEL_GEN_URL: str = "http://gen-model:8002"
    MODEL_STT_URL: str = "http://stt-model:8003"
    
settings = Settings()