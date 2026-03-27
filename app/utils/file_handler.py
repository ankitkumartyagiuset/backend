import os
import uuid
from app.core.config import settings

def save_uploaded_file(image_bytes: bytes, filename: str) -> str:
    ext = filename.split(".")[-1] if "." in filename else "jpg"
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    with open(filepath, "wb") as f:
        f.write(image_bytes)
        
    return filepath
