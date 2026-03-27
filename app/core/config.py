import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    MODEL_PATH = os.getenv("MODEL_PATH", "./model/skin_model.h5")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

settings = Settings()
