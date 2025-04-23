import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BASE_DIR: str = os.path.join(os.getcwd(), "app")
    FACE_RECOGNITION_WEIGHTS_PATH: str = os.path.join(BASE_DIR, "artifacts", "FaceRecognition")
    os.makedirs(FACE_RECOGNITION_WEIGHTS_PATH, exist_ok = True) 

settings = Settings()