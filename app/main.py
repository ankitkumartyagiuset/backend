from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.database import engine, Base
import os
from app.core.config import settings
from fastapi.staticfiles import StaticFiles

# Create db tables
Base.metadata.create_all(bind=engine)

# Create structural directories if not exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)

app = FastAPI(title="AI-Powered Rural Skin Diagnosis & Referral System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Serve the Frontend seamlessly from root
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend") # Mount frontend HTML
else:
    @app.get("/")
    def root():
        return {"message": "Skin AI Assistant API is running (Frontend statically detached)"}
