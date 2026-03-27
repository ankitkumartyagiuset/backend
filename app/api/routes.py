from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.prediction_schema import PredictionResponse, PredictionCreate
from app.services.prediction_service import process_prediction
import traceback

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(...), 
    contribute_data: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    
    try:
        contents = await file.read()
        return process_prediction(contents, file.filename, db, contribute_data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

import zipfile
import tempfile
import os
import shutil
from app.ml.analytics_trainer import upgrade_model_pipeline

@router.post("/upload-dataset")
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    """ Endpoint to upload a .zip dataset for auto-upgrading the AI Model """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file containing your dataset.")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(zip_path, "wb") as f:
            f.write(await file.read())
            
        dataset_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(dataset_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        # Run active-learning pipeline on the extracted data
        upgrade_model_pipeline(dataset_dir)
        
        return {"status": "success", "message": "Dataset analyzed and Model Auto-Upgraded successfully!"}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")
    finally:
        # Clean up temporary zip and folder
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

