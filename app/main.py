from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import random

app = FastAPI()

# ✅ CORS (connect frontend)
origins = [
    "https://ai-4-healthcare.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Create uploads folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 🧠 Dummy AI Prediction Function (replace later with model)
def predict_disease(image_path):
    conditions = [
        ("Benign Skin Lesion", "Low", "Safe to treat locally"),
        ("Fungal Infection", "Medium", "Use antifungal cream"),
        ("Skin Cancer Risk", "High", "Refer to specialist immediately")
    ]

    condition, severity, recommendation = random.choice(conditions)

    confidence = round(random.uniform(85, 98), 2)

    explanation = "Detected patterns similar to common skin conditions."

    return {
        "condition": condition,
        "confidence": f"{confidence}%",
        "severity": severity,
        "recommendation": recommendation,
        "explanation": explanation
    }


# 📸 API: Upload & Predict
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # AI Prediction
    result = predict_disease(file_path)

    return {
        "status": "success",
        "data": result,
        "image": file.filename
    }


# 🏠 Test Route
@app.get("/")
def home():
    return {"message": "Skin AI Care Backend Running 🚀"}


# 🚀 Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
